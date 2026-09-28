"""Tests for the Claude Code CLI backend (subprocess is faked)."""

from __future__ import annotations

import json
import subprocess

import pytest

from garmin_health_monitor.claude_cli_client import ClaudeCliClient, _extract_json_object
from garmin_health_monitor.config import (
    AppConfig,
    ClaudeCliConfig,
    ConfigError,
    LLMConfig,
    parse_chat_targets,
    parse_config,
)
from garmin_health_monitor.llm import LLMError, describe_backend, make_llm_client

SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}


class FakeRunner:
    """Records the subprocess call and returns canned CompletedProcess objects."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "", raise_exc: Exception | None = None):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def __call__(self, args, **kwargs):
        self.calls.append({"args": args, **kwargs})
        if self.raise_exc:
            raise self.raise_exc
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.stdout, stderr=self.stderr)


def envelope(**overrides):
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 2,
        "result": json.dumps({"summary": "from result"}),
        "structured_output": {"summary": "from structured"},
        "total_cost_usd": 0.004,
        "session_id": "abc",
    }
    base.update(overrides)
    return json.dumps(base)


def make_client(runner, **cfg_kwargs) -> ClaudeCliClient:
    cfg = ClaudeCliConfig(**cfg_kwargs)
    return ClaudeCliClient(cfg, workdir="/tmp/ghm-claude-test", runner=runner, env={"PATH": "/usr/bin"})


def test_chat_json_prefers_structured_output():
    runner = FakeRunner(stdout=envelope())
    client = make_client(runner, model="sonnet", effort="low")
    out = client.chat_json("SYS", "USER", SCHEMA)
    assert out == {"summary": "from structured"}
    call = runner.calls[0]
    args = call["args"]
    assert args[0] == "claude" and "-p" in args
    assert call["input"] == "USER"  # prompt goes over stdin
    assert args[args.index("--system-prompt") + 1] == "SYS"
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--effort") + 1] == "low"
    assert json.loads(args[args.index("--json-schema") + 1]) == SCHEMA
    assert args[args.index("--tools") + 1] == ""
    assert "--no-session-persistence" in args and "--permission-mode" in args
    assert int(args[args.index("--max-turns") + 1]) >= 2
    assert call["env"]["DISABLE_AUTOUPDATER"] == "1"
    assert call["timeout"] == client.cfg.timeout_seconds


def test_max_turns_is_raised_to_two():
    runner = FakeRunner(stdout=envelope())
    client = make_client(runner, max_turns=1)
    client.chat_json("s", "u", SCHEMA)
    args = runner.calls[0]["args"]
    assert args[args.index("--max-turns") + 1] == "2"


def test_chat_json_falls_back_to_result_text_with_fences():
    fenced = "```json\n" + json.dumps({"summary": "fenced"}) + "\n```"
    runner = FakeRunner(stdout=envelope(structured_output=None, result=fenced))
    assert make_client(runner).chat_json("s", "u", SCHEMA) == {"summary": "fenced"}


def test_chat_json_extracts_object_from_prose():
    runner = FakeRunner(stdout=envelope(structured_output=None, result='Sure! {"summary": "x"} hope this helps'))
    assert make_client(runner).chat_json("s", "u", SCHEMA) == {"summary": "x"}


def test_chat_json_raises_on_garbage():
    runner = FakeRunner(stdout=envelope(structured_output=None, result="no json here"))
    with pytest.raises(LLMError):
        make_client(runner).chat_json("s", "u", SCHEMA)


def test_chat_text_returns_result():
    runner = FakeRunner(stdout=envelope(result="Hello there", structured_output=None))
    assert make_client(runner).chat_text("s", "u") == "Hello there"
    assert "--json-schema" not in runner.calls[0]["args"]


def test_error_envelope_raises():
    runner = FakeRunner(stdout=envelope(is_error=True, subtype="error_max_turns", errors=["Reached maximum number of turns (1)"]))
    with pytest.raises(LLMError, match="maximum number of turns"):
        make_client(runner).chat_text("s", "u")


def test_nonzero_exit_without_json_raises():
    runner = FakeRunner(stdout="", stderr="Not logged in", returncode=1)
    with pytest.raises(LLMError, match="Not logged in"):
        make_client(runner).chat_text("s", "u")


def test_timeout_raises():
    runner = FakeRunner(raise_exc=subprocess.TimeoutExpired(cmd="claude", timeout=5))
    with pytest.raises(LLMError, match="timed out"):
        make_client(runner, timeout_seconds=5).chat_text("s", "u")


def test_missing_binary_raises():
    runner = FakeRunner(raise_exc=FileNotFoundError("claude"))
    with pytest.raises(LLMError, match="not found"):
        make_client(runner).chat_text("s", "u")


def test_is_available_uses_version():
    runner = FakeRunner(stdout="2.1.0 (Claude Code)\n")
    assert make_client(runner).is_available() is True
    assert runner.calls[0]["args"] == ["claude", "--version"]
    assert make_client(FakeRunner(returncode=1)).is_available() is False


def test_bare_and_extra_args():
    runner = FakeRunner(stdout=envelope())
    client = make_client(runner, bare=True, extra_args=["--fallback-model", "haiku"], max_budget_usd=0.05)
    client.chat_json("s", "u", SCHEMA)
    args = runner.calls[0]["args"]
    assert args[1] == "--bare"
    assert args[-2:] == ["--fallback-model", "haiku"]
    assert args[args.index("--max-budget-usd") + 1] == "0.05"


def test_extract_json_object_variants():
    assert _extract_json_object('{"a": 1}') == {"a": 1}
    assert _extract_json_object("```\n{\"a\": 1}\n```") == {"a": 1}
    assert _extract_json_object("[1,2]") is None
    assert _extract_json_object("nothing") is None


# --- factory / config ---------------------------------------------------------


def test_make_llm_client_backends(tmp_path):
    from tests.conftest import make_app_config

    cfg = make_app_config(tmp_path)
    cfg.llm = LLMConfig(backend="claude-cli")
    client = make_llm_client(cfg)
    assert isinstance(client, ClaudeCliClient)
    assert client.workdir.endswith("claude-workdir")
    assert "ClaudeCliClient" in describe_backend(client)

    cfg.llm = LLMConfig(backend="none")
    assert make_llm_client(cfg) is None
    assert describe_backend(None) == "disabled"

    cfg.llm = LLMConfig(backend="bogus")
    with pytest.raises(ValueError):
        make_llm_client(cfg)


def base_raw(**over):
    raw = {
        "timezone": "Asia/Singapore",
        "data_dir": "/tmp/ghm-test-data",
        "telegram": {"bot_token": "t", "admin_chat_ids": ["-1002069000031:2665", 5]},
        "profiles": [{"name": "Dad", "garmin": {"email": "a", "password": "b"}, "telegram_chat_ids": ["-1002069000031/2665"]}],
    }
    raw.update(over)
    return raw


def test_parse_config_llm_defaults_to_claude_cli():
    cfg: AppConfig = parse_config(base_raw())
    assert cfg.llm.backend == "claude-cli"
    assert cfg.llm.claude.model == "sonnet"
    assert cfg.llm.claude.workdir == "/tmp/ghm-test-data/claude-workdir"


def test_parse_config_llm_section():
    cfg = parse_config(base_raw(llm={"backend": "claude", "claude": {"model": "opus", "effort": "medium", "env": {"X": 1}}}))
    assert cfg.llm.backend == "claude-cli"
    assert cfg.llm.claude.model == "opus"
    assert cfg.llm.claude.env == {"X": "1"}
    cfg = parse_config(base_raw(llm={"backend": "ollama"}, ollama={"enabled": False}))
    assert cfg.llm.backend == "ollama" and cfg.ollama.enabled is True
    with pytest.raises(ConfigError):
        parse_config(base_raw(llm={"backend": "gpt"}))


def test_chat_targets_with_topics():
    ids, threads = parse_chat_targets(["-1002069000031:2665", "123", 456, {"chat_id": 789, "thread_id": 3}, "123"])
    assert ids == [-1002069000031, 123, 456, 789]
    assert threads == {-1002069000031: 2665, 789: 3}
    cfg = parse_config(base_raw())
    assert cfg.telegram.admin_threads == {-1002069000031: 2665}
    assert cfg.profiles[0].telegram_threads == {-1002069000031: 2665}
    assert cfg.thread_for(-1002069000031) == 2665
    assert cfg.thread_for(5) is None
    with pytest.raises(ConfigError):
        parse_chat_targets(["abc"])
