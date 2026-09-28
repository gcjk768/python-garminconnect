"""Use the Claude Code CLI in print mode (``claude -p``) as the analysis model.

Each call spawns ``claude -p`` with:

* ``--output-format json`` so we get a machine-readable envelope
  (``result``, ``structured_output``, ``is_error``, ``total_cost_usd`` ...),
* ``--json-schema`` for structured answers (delivered via a tool call, which
  is why ``--max-turns`` must be at least 2),
* ``--system-prompt`` replacing the coding-assistant prompt with ours,
* ``--tools ""`` and ``--permission-mode dontAsk`` so the CLI cannot touch
  files or run commands: it is a pure text-in / JSON-out call,
* ``--no-session-persistence`` so thousands of tiny sessions are not written
  to disk.

``--safe-mode`` (or ``--bare`` with an API key) keeps the user's CLAUDE.md,
hooks, plugins and MCP servers out of the call.

Authentication is whatever the CLI already has: a subscription login, a
long-lived token from ``claude setup-token`` in ``CLAUDE_CODE_OAUTH_TOKEN`` or
an ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import ClaudeCliConfig
from .llm import LLMError

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

Runner = Callable[..., subprocess.CompletedProcess]


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as a JSON object, tolerating fences and leading prose."""
    candidate = _strip_fences(text)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(candidate[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class ClaudeCliClient:
    """LLM backend that shells out to ``claude -p``."""

    def __init__(
        self,
        cfg: ClaudeCliConfig,
        workdir: str | None = None,
        runner: Runner | None = None,
        env: dict[str, str] | None = None,
    ):
        self.cfg = cfg
        self.workdir = workdir or cfg.workdir or "."
        self._runner: Runner = runner or subprocess.run
        self._base_env = dict(os.environ if env is None else env)

    # -- interface ------------------------------------------------------------

    @property
    def model(self) -> str:
        return self.cfg.model

    def is_available(self) -> bool:
        exe = shutil.which(self.cfg.command) if self._runner is subprocess.run else self.cfg.command
        if not exe:
            logger.warning("Claude CLI command %r not found on PATH", self.cfg.command)
            return False
        try:
            proc = self._runner(
                [exe, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._env(),
                cwd=self._cwd(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Claude CLI not runnable: %s", exc)
            return False
        ok = proc.returncode == 0
        if ok:
            logger.info("Claude CLI available: %s", (proc.stdout or "").strip())
        return ok

    def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        envelope = self._run(system, user, schema=schema)
        structured = envelope.get("structured_output")
        if isinstance(structured, dict):
            return structured
        if isinstance(structured, str):
            parsed = _extract_json_object(structured)
            if parsed is not None:
                return parsed
        parsed = _extract_json_object(str(envelope.get("result") or ""))
        if parsed is None:
            raise LLMError("Claude CLI returned no parseable JSON object")
        return parsed

    def chat_text(self, system: str, user: str) -> str:
        envelope = self._run(system, user)
        text = str(envelope.get("result") or "").strip()
        if not text:
            raise LLMError("Claude CLI returned an empty result")
        return text

    # -- internals ------------------------------------------------------------

    def _env(self) -> dict[str, str]:
        env = dict(self._base_env)
        env.setdefault("DISABLE_AUTOUPDATER", "1")
        env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        env.update(self.cfg.env or {})
        return env

    def _cwd(self) -> str:
        try:
            Path(self.workdir).mkdir(parents=True, exist_ok=True)
            return self.workdir
        except OSError:
            return "."

    def build_args(self, system: str, schema: dict[str, Any] | None = None) -> list[str]:
        # full path so Windows finds claude.cmd / claude.exe (subprocess does not search PATHEXT)
        command = (shutil.which(self.cfg.command) if self._runner is subprocess.run else None) or self.cfg.command
        args = [
            command,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.cfg.model,
            "--max-turns",
            str(max(2, int(self.cfg.max_turns))),
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--system-prompt",
            system,
        ]
        # bare needs ANTHROPIC_API_KEY; safe mode keeps the subscription login but still drops
        # CLAUDE.md, hooks, plugins and MCP servers, which otherwise hijack the answer and
        # cost ~100x the tokens
        args.insert(1, "--bare" if self.cfg.bare else "--safe-mode")
        if self.cfg.effort:
            args += ["--effort", str(self.cfg.effort)]
        if self.cfg.max_budget_usd:
            args += ["--max-budget-usd", str(self.cfg.max_budget_usd)]
        if schema is not None:
            args += ["--json-schema", json.dumps(schema)]
        args += list(self.cfg.extra_args or [])
        return args

    def _run(self, system: str, user: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        args = self.build_args(system, schema)
        try:
            proc = self._runner(
                args,
                input=user,
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout_seconds,
                env=self._env(),
                cwd=self._cwd(),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"Claude CLI timed out after {self.cfg.timeout_seconds}s") from exc
        except FileNotFoundError as exc:
            raise LLMError(f"Claude CLI command {self.cfg.command!r} not found; install it or set llm.claude.command") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise LLMError(f"Claude CLI failed to start: {exc}") from exc

        stdout = proc.stdout or ""
        stderr = (proc.stderr or "").strip()
        envelope: dict[str, Any] | None = None
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                envelope = parsed
            elif isinstance(parsed, list):  # stream-style: last result object
                for item in reversed(parsed):
                    if isinstance(item, dict) and item.get("type") == "result":
                        envelope = item
                        break
        except json.JSONDecodeError:
            envelope = None

        if proc.returncode != 0 and envelope is None:
            raise LLMError(f"Claude CLI exited with {proc.returncode}: {stderr[-500:] or stdout[-500:]}")
        if envelope is None:
            raise LLMError(f"Claude CLI produced no JSON envelope: {stdout[-300:]!r} {stderr[-300:]!r}")
        if envelope.get("is_error"):
            errors = envelope.get("errors") or envelope.get("result") or envelope.get("subtype")
            raise LLMError(f"Claude CLI reported an error: {errors}")
        cost = envelope.get("total_cost_usd")
        if cost is not None:
            logger.debug("Claude CLI call: model=%s turns=%s cost_usd=%.4f", self.cfg.model, envelope.get("num_turns"), float(cost))
        return envelope
