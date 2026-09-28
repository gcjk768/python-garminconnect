---
tags: [active]
updated: 2026-09-29
---
# Changelog

## 2026-09-29 — Fitness section + schedule
- `GarminSession.fetch_fitness`: VO2 max, training status + load balance, fitness age, weekly intensity minutes, race predictions (once per message, not every poll; falls back to yesterday's training status until Garmin processes today).
- `messages.fitness_lines` → 💪 Fitness in the evening summary only (removed from morning brief at James's request).
- NAS schedule: morning 09:00, evening 22:00, weekly Sunday 22:00.
- Compose mounts `./garmin_health_monitor` so code updates need only a Restart.

## 2026-09-29 — NAS deploy + phone-sized messages
- `Dockerfile` (python:3.12-slim + native Claude CLI, uid 1000) and `docker-compose.yaml`; stack at `/volume1/docker/garmin-monitor` in Dockge (:5001). `.env` there is mode 600.
- Telegram: new bot @koh_heart_tracker_bot, member of James Channel, posts to topic 2665 "James Garmin Tracker".
- Messages: unmeasured values hidden; coaching capped at 2 do more / 2 do less / 1 watch out; bullets under 12 words.
- James's profile: status + coaching only (palpitations off). Dad's profile commented in the NAS config until his login exists.
- All 223 tests green (fixed Windows path + backfill test).

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
