---
tags: [active]
updated: 2026-09-29
---
# Changelog

## 2026-09-29 — branch `father`
- Added `extract` task: `/palp` / `/note` text → structured fields via `claude -p --json-schema`, stored beside the raw note.
- Hard-coded red-flag check (chest pain, fainting, breathlessness) → "call 995" banner, works with no LLM.
- Dad's symptom logs are copied to James's Telegram; Dad's profile targets both chats.
- Prompts: no dashes, 65-year-old reading level; weekly patterns need ≥3 occurrences with counts; doctor narrative <250 words, names red-flag dates.
