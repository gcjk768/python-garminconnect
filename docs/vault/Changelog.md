---
tags: [active]
updated: 2026-10-01
---
# Changelog

## 2026-10-01
- fix: in a forum group each bot answers only in its own topic (`_authorised_chat` in `garmin_health_monitor/telegram_bot.py` checks `message_thread_id` against `thread_for`). /ask in the SG car topic made both James and Dad bots reply "I don't know that command".

## 2026-09-30 — README + architecture diagram
- New `README.md` (problem, engineering highlights, flow, stack, setup, limits; no real readings or names).
- `docs/architecture.drawio` (editable) + exported `docs/architecture.drawio.svg` / `docs/architecture.png`.

## 2026-09-29 — Safety + routine features ("do all")
- Dad alerts: `alerts.only: [no_sync, low_hr, garmin_abnormal_hr]`; watch not synced 6h; low HR < 40 bpm for 10 min awake (PLACEHOLDER, confirm with doctor; backtest on Sept: 0 days).
- Dad medicine: metoprolol 50 mg reminder 09:00 (PLACEHOLDER time) with ✅ Taken button; status in the 22:00 heart review; started 2026-09-27 marked on the calendar.
- 1st of month 09:00: Dad calendar + Claude month report (needs token) + doctor PDF/CSV; James 30-day progress picture.
- James: Thursday 19:00 workout nudge when behind "3 workouts a week".
- Both: nightly 03:30 SQLite online backup to /volume1/James/Backups/<stack>, keep 30.
- Default branch fast-forwarded to `father`.

## 2026-09-29 — Alerts explain possible links
- Alert adds "🔍 Possible links" (software rules via `analysis.episode_links`: sleep last night, high-stress minutes, Body Battery vs his 14-day median) and a fixed safe "🌿 Now" tip. No medicine advice.
- September check: nearly every episode day had short/poor sleep and/or a very stressful day (e.g. 3.4 h sleep before 24 Sep).
- Monthly calendar chart (`charts.heart_calendar`) and a one-off Claude "what September shows" report (sent manually; automation needs CLAUDE_CODE_OAUTH_TOKEN on the NAS).

## 2026-09-29 — Garmin outage handling
- 10:17 Garmin HTTP 521 outage fired raw error dumps + a misleading "run login" on the first failed poll; both stacks recovered by themselves (tokens fine).
- Now: failures are remembered; one plain alert only after 2h (login vs servers down); "✅ data is back" on recovery.
- Recovery re-checks earlier days the outage covered (an outage past midnight used to skip the end of the previous day).

## 2026-09-29 — Dad: tables + weekly summary
- Heart review (22:00) and monthly/weekly summaries use one aligned `<pre>` table (date, time, peak, minutes); only AI-ruled-out episodes hidden.
- Dad weekly summary every Sunday 22:00 (`weekly_review_text` returns `heart_month` for heart-only profiles).

## 2026-09-29 — Dad's heart tracker live
- Second NAS stack `dad-heart-monitor` (same image + mounted code as `garmin-monitor`), bot @koh_dad_heart_bot "Dad Heart Tracker" → James Channel topic 2677 "Dad Garmin Tracker".
- Features: instant palpitation alert (date, time, peak/before HR, state, AI view) + 22:00 heart review (`features.heart_review`). Nothing else.
- Thresholds tuned for Dad's resting HR ~42: rise_over_baseline 50, max_steps_in_window 60, alert_min_confidence 0.5, quiet hours 23 to 07.
- Obsidian vault `/volume1/James/Obsidian/Dad Heart` (`vault_dir`), backfilled with September: 18 flagged (12 possible, 4 unclear, 2 exertion).

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
