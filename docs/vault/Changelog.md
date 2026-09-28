---
tags: [active]
updated: 2026-09-29
---
# Changelog

## 2026-09-29 — `claude -p` on a subscription (no API key)
- Calls now run with `--safe-mode`: drops CLAUDE.md/hooks/plugins/MCP (they overrode the schema; cost fell ~$0.58 → ~$0.005 per call).
- Resolve the `claude` path before spawning; on Windows point `llm.claude.command` at `claude.exe` (the `.cmd` shim mangles JSON and multi-line args).
- Red flags widened: any chest mention, or extracted chest_tightness / near_fainting, gets the 995 banner.
- Verified live: extraction returns correct vocabulary in 5 to 7 s.

## 2026-09-29 — branch `father`
- Added `extract` task: `/palp` / `/note` text → structured fields via `claude -p --json-schema`, stored beside the raw note.
- Hard-coded red-flag check (chest pain, fainting, breathlessness) → "call 995" banner, works with no LLM.
- Dad's symptom logs are copied to James's Telegram; Dad's profile targets both chats.
- Prompts: no dashes, 65-year-old reading level; weekly patterns need ≥3 occurrences with counts; doctor narrative <250 words, names red-flag dates.
