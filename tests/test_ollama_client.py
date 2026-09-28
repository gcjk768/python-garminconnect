"""Tests for :mod:`garmin_health_monitor.ollama_client` using a stubbed HTTP session."""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from garmin_health_monitor.config import OllamaConfig
from garmin_health_monitor.ollama_client import (
    OllamaClient,
    OllamaError,
    extract_json_object,
    strip_code_fences,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = (
            text if text is not None else (json.dumps(payload) if payload is not None else "")
        )

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def chat_response(content: str) -> FakeResponse:
    return FakeResponse(
        200,
        {
            "model": "test-model",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
            "total_duration": 1_500_000_000,
        },
    )


class FakeSession:
    """Stub of ``requests.Session``: queued responses per path, records every call."""

    def __init__(self, routes: dict[str, list[Any]] | None = None):
        self.routes = {k: list(v) for k, v in (routes or {}).items()}
        self.calls: list[dict[str, Any]] = []

    def _dispatch(self, method: str, url: str, **kw: Any) -> FakeResponse:
        path = url.split("11434", 1)[-1] if "11434" in url else url
        self.calls.append({"method": method, "url": url, "path": path, **kw})
        queue = self.routes.get(path)
        if not queue:
            raise AssertionError(f"unexpected {method} {url}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, **kw)


def make_client(
    routes: dict[str, list[Any]] | None = None, **cfg_kw: Any
) -> tuple[OllamaClient, FakeSession]:
    cfg = OllamaConfig(base_url="http://ollama.test:11434/", model="test-model", **cfg_kw)
    session = FakeSession(routes)
    return OllamaClient(cfg, session=session), session


TAGS = {"models": [{"name": "test-model"}, {"name": "llama3.1:latest"}, {"name": "gemma:2b"}]}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_strip_code_fences_and_extract():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences("  plain  ") == "plain"
    assert extract_json_object('Sure, here it is:\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}
    assert extract_json_object('The answer: {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")
    with pytest.raises(ValueError):
        extract_json_object("no json here")
    with pytest.raises(ValueError):
        extract_json_object("")


# ---------------------------------------------------------------------------
# availability / models
# ---------------------------------------------------------------------------


def test_model_property_and_base_url_trailing_slash():
    client, _ = make_client()
    assert client.model == "test-model"
    assert client.base_url == "http://ollama.test:11434"
    assert client.timeout == 180


def test_is_available_true_and_false():
    client, _ = make_client({"/api/tags": [FakeResponse(200, TAGS)]})
    assert client.is_available() is True
    client, _ = make_client({"/api/tags": [requests.ConnectionError("refused")]})
    assert client.is_available() is False
    client, _ = make_client({"/api/tags": [FakeResponse(500, None, "boom")]})
    assert client.is_available() is False


def test_has_model_matching_rules():
    routes = {"/api/tags": [FakeResponse(200, TAGS)] * 6}
    client, _ = make_client(routes)
    assert client.has_model() is True  # configured model, exact
    assert client.has_model("llama3.1") is True  # bare name matches :latest
    assert client.has_model("gemma") is True  # bare name matches any tag
    assert client.has_model("gemma:7b") is False  # explicit tag must match
    assert client.has_model("mistral") is False
    assert client.has_model(None) is True  # None / "" fall back to the configured model


def test_has_model_raises_when_unreachable():
    client, _ = make_client({"/api/tags": [requests.ConnectionError("refused")]})
    with pytest.raises(OllamaError):
        client.has_model()


def test_pull_model_posts_expected_payload_and_uses_long_timeout():
    client, session = make_client({"/api/pull": [FakeResponse(200, {"status": "success"})]})
    client.pull_model()
    call = session.calls[-1]
    assert call["method"] == "POST" and call["path"] == "/api/pull"
    assert call["json"] == {"name": "test-model", "stream": False}
    assert call["timeout"] >= 180


def test_pull_model_error_paths():
    client, _ = make_client(
        {"/api/pull": [FakeResponse(200, {"error": "pull model manifest: not found"})]}
    )
    with pytest.raises(OllamaError, match="not found"):
        client.pull_model("nope")
    client, _ = make_client({"/api/pull": [FakeResponse(500, None, "server error")]})
    with pytest.raises(OllamaError):
        client.pull_model()


def test_ensure_model_present_does_not_pull():
    client, session = make_client({"/api/tags": [FakeResponse(200, TAGS)]})
    client.ensure_model()
    assert all(c["path"] != "/api/pull" for c in session.calls)


def test_ensure_model_pulls_when_auto_pull():
    missing = {"models": [{"name": "other"}]}
    client, session = make_client(
        {
            "/api/tags": [FakeResponse(200, missing), FakeResponse(200, TAGS)],
            "/api/pull": [FakeResponse(200, {"status": "success"})],
        },
        auto_pull=True,
    )
    client.ensure_model()
    assert [c["path"] for c in session.calls] == ["/api/tags", "/api/pull", "/api/tags"]


def test_ensure_model_raises_when_missing_and_no_auto_pull():
    client, session = make_client(
        {"/api/tags": [FakeResponse(200, {"models": []})]}, auto_pull=False
    )
    with pytest.raises(OllamaError, match="ollama pull test-model"):
        client.ensure_model()
    assert all(c["path"] != "/api/pull" for c in session.calls)


# ---------------------------------------------------------------------------
# chat_json
# ---------------------------------------------------------------------------


def test_chat_json_happy_path_sends_structured_payload():
    client, session = make_client(
        {"/api/chat": [chat_response('{"a": 1}')]}, temperature=0.3, num_ctx=4096, keep_alive="10m"
    )
    out = client.chat_json("sys prompt", "user prompt", SCHEMA)
    assert out == {"a": 1}
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["method"] == "POST" and call["path"] == "/api/chat"
    assert call["timeout"] == 180
    payload = call["json"]
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert payload["format"] == SCHEMA
    assert payload["options"] == {"temperature": 0.3, "num_ctx": 4096}
    assert payload["keep_alive"] == "10m"
    assert payload["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_chat_json_uses_configured_timeout():
    client, session = make_client({"/api/chat": [chat_response('{"a": 1}')]}, timeout_seconds=42)
    client.chat_json("s", "u", SCHEMA)
    assert session.calls[0]["timeout"] == 42


def test_chat_json_strips_code_fences():
    client, session = make_client(
        {"/api/chat": [chat_response('```json\n{"a": 5, "b": "x"}\n```')]}
    )
    assert client.chat_json("s", "u", SCHEMA) == {"a": 5, "b": "x"}
    assert len(session.calls) == 1


def test_chat_json_retries_once_with_json_format():
    client, session = make_client(
        {"/api/chat": [chat_response("I cannot answer in JSON, sorry."), chat_response('{"a": 7}')]}
    )
    assert client.chat_json("s", "u", SCHEMA) == {"a": 7}
    assert len(session.calls) == 2
    assert session.calls[0]["json"]["format"] == SCHEMA
    assert session.calls[1]["json"]["format"] == "json"


def test_chat_json_retries_when_schema_rejected_with_400():
    client, session = make_client(
        {"/api/chat": [FakeResponse(400, {"error": "invalid format"}), chat_response('{"a": 3}')]}
    )
    assert client.chat_json("s", "u", SCHEMA) == {"a": 3}
    assert session.calls[1]["json"]["format"] == "json"


def test_chat_json_unparseable_twice_raises():
    client, session = make_client(
        {"/api/chat": [chat_response("nope"), chat_response("still nope")]}
    )
    with pytest.raises(OllamaError, match="parseable JSON"):
        client.chat_json("s", "u", SCHEMA)
    assert len(session.calls) == 2


def test_chat_json_non_object_json_retries_then_raises():
    client, _ = make_client({"/api/chat": [chat_response("[1,2]"), chat_response('"str"')]})
    with pytest.raises(OllamaError):
        client.chat_json("s", "u", SCHEMA)


def test_chat_json_http_error_raises_without_retry():
    client, session = make_client({"/api/chat": [FakeResponse(500, None, "internal")]})
    with pytest.raises(OllamaError, match="HTTP 500"):
        client.chat_json("s", "u", SCHEMA)
    assert len(session.calls) == 1


def test_chat_json_timeout_and_connection_errors():
    client, _ = make_client({"/api/chat": [requests.Timeout("slow")]})
    with pytest.raises(OllamaError, match="timed out"):
        client.chat_json("s", "u", SCHEMA)
    client, _ = make_client({"/api/chat": [requests.ConnectionError("refused")]})
    with pytest.raises(OllamaError, match="Cannot reach"):
        client.chat_json("s", "u", SCHEMA)
    client, _ = make_client({"/api/chat": [requests.RequestException("weird")]})
    with pytest.raises(OllamaError):
        client.chat_json("s", "u", SCHEMA)


def test_chat_json_malformed_body_raises():
    client, _ = make_client(
        {"/api/chat": [FakeResponse(200, None, "not json"), FakeResponse(200, {"done": True})]}
    )
    with pytest.raises(OllamaError, match="non-JSON"):
        client.chat_json("s", "u", SCHEMA)
    client, _ = make_client({"/api/chat": [FakeResponse(200, {"done": True})]})
    with pytest.raises(OllamaError, match="message.content"):
        client.chat_json("s", "u", SCHEMA)


def test_chat_json_error_field_in_body():
    client, _ = make_client({"/api/chat": [FakeResponse(200, {"error": "model not found"})]})
    with pytest.raises(OllamaError, match="model not found"):
        client.chat_json("s", "u", SCHEMA)


# ---------------------------------------------------------------------------
# chat_text
# ---------------------------------------------------------------------------


def test_chat_text_returns_stripped_content_without_format():
    client, session = make_client({"/api/chat": [chat_response("  A short paragraph.\n")]})
    assert client.chat_text("s", "u") == "A short paragraph."
    assert "format" not in session.calls[0]["json"]
    assert session.calls[0]["json"]["stream"] is False


def test_chat_text_unwraps_whole_answer_fence_only():
    client, _ = make_client({"/api/chat": [chat_response("```\nWhole answer.\n```")]})
    assert client.chat_text("s", "u") == "Whole answer."
    client, _ = make_client({"/api/chat": [chat_response("Intro. ```x``` outro.")]})
    assert client.chat_text("s", "u") == "Intro. ```x``` outro."


def test_chat_text_error_paths():
    client, _ = make_client({"/api/chat": [FakeResponse(404, {"error": "model 'x' not found"})]})
    with pytest.raises(OllamaError, match="HTTP 404"):
        client.chat_text("s", "u")
    client, _ = make_client({"/api/chat": [requests.Timeout("slow")]})
    with pytest.raises(OllamaError):
        client.chat_text("s", "u")
