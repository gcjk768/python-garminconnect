"""Thin client for a local Ollama server (``/api/tags``, ``/api/pull``, ``/api/chat``).

The monitor only needs three things from Ollama: to know whether it is up and
has the configured model, to pull that model on first start, and to run one
chat completion that returns JSON (structured output) or plain text.

Everything goes through an injectable :class:`requests.Session` so tests can
stub the HTTP layer.  Every failure (connection refused, timeout, HTTP error,
unparseable model output) surfaces as :class:`OllamaError` so callers can fall
back to rule-based output without inspecting ``requests`` exceptions.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

from .config import OllamaConfig
from .llm import LLMError

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", re.DOTALL)
_MAX_ERROR_BODY = 300


class OllamaError(LLMError):
    """Raised for any Ollama transport, HTTP or output-format problem."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def strip_code_fences(text: str) -> str:
    """Return the inside of the first ```-fenced block, or the stripped text itself."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse ``text`` as a JSON object, tolerating code fences and surrounding prose.

    Raises ``ValueError`` when no JSON object can be found.
    """
    candidate = strip_code_fences(text)
    if not candidate:
        raise ValueError("empty model output")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object in model output") from None
        parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"model output is JSON but not an object ({type(parsed).__name__})")
    return parsed


class OllamaClient:
    """Small synchronous wrapper around the Ollama HTTP API."""

    def __init__(self, cfg: OllamaConfig, session: requests.Session | None = None):
        self.cfg = cfg
        self.base_url = str(cfg.base_url or "http://localhost:11434").rstrip("/")
        self._session = session or requests.Session()
        self.timeout = float(cfg.timeout_seconds or 180)
        # Pulling a multi-GB model on a NAS can take far longer than one chat turn.
        self.pull_timeout = max(self.timeout, 1800.0)

    # -- properties -------------------------------------------------------

    @property
    def model(self) -> str:
        return self.cfg.model

    # -- low level --------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Perform one HTTP call and return the decoded JSON body.

        Raises :class:`OllamaError` on connection problems, timeouts, HTTP status
        >= 400 and non-JSON bodies.
        """
        url = f"{self.base_url}{path}"
        timeout = self.timeout if timeout is None else timeout
        started = time.monotonic()
        try:
            if method == "GET":
                resp = self._session.get(url, timeout=timeout)
            else:
                resp = self._session.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise OllamaError(f"Ollama timed out after {timeout:.0f}s ({method} {path})") from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"Ollama request failed ({method} {path}): {exc}") from exc

        status = getattr(resp, "status_code", 0)
        if status >= 400:
            body = ""
            try:
                body = str(getattr(resp, "text", "") or "")[:_MAX_ERROR_BODY]
            except Exception:  # pragma: no cover - defensive
                body = ""
            raise OllamaError(f"Ollama HTTP {status} on {method} {path}: {body}", status=status)
        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama returned non-JSON body on {method} {path}") from exc
        logger.debug("%s %s -> %s in %.1fs", method, path, status, time.monotonic() - started)
        return data

    # -- availability / models -------------------------------------------

    def is_available(self) -> bool:
        """True when ``GET /api/tags`` succeeds.  Never raises."""
        try:
            self._request("GET", "/api/tags", timeout=min(self.timeout, 10.0))
            return True
        except OllamaError as exc:
            logger.debug("Ollama not available: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """Names of the models the server has (raises :class:`OllamaError` if unreachable)."""
        data = self._request("GET", "/api/tags")
        models = data.get("models") if isinstance(data, dict) else None
        names: list[str] = []
        for m in models or []:
            if isinstance(m, dict) and m.get("name"):
                names.append(str(m["name"]))
            elif isinstance(m, str):
                names.append(m)
        return names

    def has_model(self, name: str | None = None) -> bool:
        """Whether ``name`` (default: the configured model) is present on the server.

        ``llama3.1`` matches ``llama3.1:latest``; an explicit tag must match exactly.
        Raises :class:`OllamaError` when the server cannot be reached.
        """
        wanted = (name or self.model or "").strip()
        if not wanted:
            return False
        names = self.list_models()
        if wanted in names:
            return True
        if ":" not in wanted:
            return any(n == f"{wanted}:latest" or n.split(":", 1)[0] == wanted for n in names)
        return False

    def pull_model(self, name: str | None = None) -> None:
        """``POST /api/pull`` (non-streaming); blocks until the download finishes."""
        target = name or self.model
        logger.info("Pulling Ollama model %s (this can take a while)", target)
        data = self._request(
            "POST", "/api/pull", {"name": target, "stream": False}, timeout=self.pull_timeout
        )
        if isinstance(data, dict) and data.get("error"):
            raise OllamaError(f"Ollama could not pull {target}: {data['error']}")
        status = data.get("status") if isinstance(data, dict) else None
        logger.info("Pull of %s finished with status %r", target, status)

    def ensure_model(self) -> None:
        """Make sure the configured model exists, pulling it when ``cfg.auto_pull`` is set.

        Raises :class:`OllamaError` when the model is missing and cannot be pulled.
        """
        if self.has_model():
            return
        if self.cfg.auto_pull:
            self.pull_model()
            if not self.has_model():
                raise OllamaError(f"Model {self.model!r} still missing after pull")
            return
        raise OllamaError(
            f"Ollama model {self.model!r} is not installed. Run `ollama pull {self.model}` "
            "or set ollama.auto_pull: true"
        )

    # -- chat -------------------------------------------------------------

    def _chat_payload(self, system: str, user: str, fmt: Any | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": float(self.cfg.temperature),
                "num_ctx": int(self.cfg.num_ctx),
            },
            "keep_alive": self.cfg.keep_alive,
        }
        if fmt is not None:
            payload["format"] = fmt
        return payload

    def _chat(self, system: str, user: str, fmt: Any | None) -> str:
        """Run one chat completion and return ``message.content``."""
        data = self._request("POST", "/api/chat", self._chat_payload(system, user, fmt))
        if not isinstance(data, dict):
            raise OllamaError("Ollama chat response is not an object")
        if data.get("error"):
            raise OllamaError(f"Ollama chat error: {data['error']}")
        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise OllamaError("Ollama chat response has no message.content")
        content = message.get("content")
        if content is None:
            content = ""
        logger.debug(
            "Ollama %s: %s prompt tokens, %s output tokens, %.1fs total",
            self.model,
            data.get("prompt_eval_count", "?"),
            data.get("eval_count", "?"),
            float(data.get("total_duration") or 0) / 1e9,
        )
        return str(content)

    def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Chat with structured output and return the parsed JSON object.

        The first attempt sends the JSON schema as ``format``.  If the answer
        cannot be parsed (or the server rejects the schema with HTTP 400, as
        older Ollama releases do), one retry is made with ``"format": "json"``.
        Raises :class:`OllamaError` when neither attempt yields a JSON object.
        """
        first_error: Exception | None = None
        try:
            content = self._chat(system, user, schema)
            return extract_json_object(content)
        except ValueError as exc:
            first_error = exc
            logger.warning("Ollama output was not valid JSON (%s); retrying with format=json", exc)
        except OllamaError as exc:
            if exc.status != 400:
                raise
            first_error = exc
            logger.warning("Ollama rejected the JSON schema (%s); retrying with format=json", exc)

        content = self._chat(system, user, "json")
        try:
            return extract_json_object(content)
        except ValueError as exc:
            snippet = (content or "")[:_MAX_ERROR_BODY].replace("\n", " ")
            raise OllamaError(
                f"Ollama did not return parseable JSON after retry ({exc}; first: {first_error}); "
                f"output: {snippet!r}"
            ) from exc

    def chat_text(self, system: str, user: str) -> str:
        """Chat without structured output; returns the stripped assistant text.

        A fence wrapping the *whole* answer is removed; fences inside prose are kept.
        """
        content = self._chat(system, user, None).strip()
        if content.startswith("```") and content.endswith("```"):
            return strip_code_fences(content)
        return content
