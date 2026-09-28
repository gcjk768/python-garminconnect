---
tags: [active]
updated: 2026-09-29
---
# App Overview

Telegram bot + scheduler on the NAS that pulls Garmin data for the family, detects at-rest heart-rate excursions for Dad, keeps a palpitation diary and builds a doctor report. English only.

- LLM backend: `claude -p` via `garmin_health_monitor/claude_cli_client.py` (schema passed with `--json-schema`); Ollama optional (`garmin_health_monitor/ollama_client.py`). Health notes leave the NAS when Claude is used.
- Prompts + schemas: `garmin_health_monitor/analysis.py` — coaching, episode assessment, doctor narrative, and `extract_symptom` (free text → symptoms / triggers / duration / red flags).
- Red flags: `analysis.has_red_flag` (hard-coded regex + extracted flags) → 995 banner in `garmin_health_monitor/messages.py::symptom_logged`.
- Symptom logging: `garmin_health_monitor/service.py::log_symptom`; raw note + `extracted` JSON stored in `symptoms` (`garmin_health_monitor/storage.py`).
- Routing: Dad's profile sends to his chat and James's (`config.example.yaml`); `/palp` from Dad's chat is copied to the admins (`garmin_health_monitor/telegram_bot.py::_log_symptom`).
