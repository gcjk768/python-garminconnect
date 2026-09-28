"""LLM backend selection.

Two interchangeable backends implement the same tiny interface:

* :class:`~garmin_health_monitor.claude_cli_client.ClaudeCliClient` runs the
  Claude Code CLI in print mode (``claude -p``) as a subprocess. This is the
  default: it uses your existing Claude subscription / login.
* :class:`~garmin_health_monitor.ollama_client.OllamaClient` talks to a local
  Ollama server over HTTP.

Both expose ``model``, ``is_available()``, ``chat_json(system, user, schema)``
and ``chat_text(system, user)`` and raise :class:`LLMError` on failure, so the
analysis code never needs to know which one is in use.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from .config import AppConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The language-model backend failed (unavailable, timeout, bad output)."""


@runtime_checkable
class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    def is_available(self) -> bool: ...

    def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...

    def chat_text(self, system: str, user: str) -> str: ...


def make_llm_client(cfg: AppConfig) -> LLMClient | None:
    """Build the configured backend, or ``None`` when analysis is disabled."""
    backend = (cfg.llm.backend or "none").lower().replace("_", "-")
    if backend in {"none", "off", "disabled", ""}:
        logger.info("LLM analysis disabled (llm.backend=%s)", backend)
        return None
    if backend in {"claude-cli", "claude", "claude-code"}:
        from .claude_cli_client import ClaudeCliClient

        return ClaudeCliClient(cfg.llm.claude, workdir=cfg.llm.claude.workdir or f"{cfg.data_dir}/claude-workdir")
    if backend == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(cfg.ollama)
    raise ValueError(f"Unknown llm.backend {cfg.llm.backend!r}; use claude-cli, ollama or none")


def describe_backend(client: LLMClient | None) -> str:
    if client is None:
        return "disabled"
    return f"{type(client).__name__} ({client.model})"
