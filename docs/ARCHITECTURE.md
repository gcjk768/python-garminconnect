# Architecture

Garmin Health Monitor is a single Python process (one Docker container) that:

1. **Polls Garmin Connect** for each configured profile (`garmin_client.py`, using the
   `garminconnect` package with a persistent token store, so MFA is needed once).
2. **Normalises** the raw payloads into a `DaySnapshot` (`normalize.py`, `models.py`).
3. **Stores** normalised rows, intraday heart rate and raw JSON in SQLite (`storage.py`).
4. **Detects palpitation candidates**: at-rest heart-rate excursions (`palpitations.py`).
5. **Asks a language model** for daily "do more / do less" coaching and for a
   plain-language assessment of each episode (`analysis.py`). The default backend is the
   **Claude Code CLI in print mode** (`claude -p`, `claude_cli_client.py`); a local **Ollama**
   server is the alternative (`ollama_client.py`). `llm.py` picks one from `llm.backend`.
6. **Sends Telegram messages** on a schedule and on events (`messages.py`,
   `telegram_bot.py`, `scheduler.py`) and answers interactive commands.
7. **Generates a doctor report** (PDF + CSV, `report.py`, `charts.py`).

`service.py` (`MonitorService`) orchestrates 1-7; `cli.py` exposes it as `garmin-monitor`.

```
Garmin Connect ──▶ garmin_client ──▶ normalize ──▶ storage (SQLite)
                                        │              │
                                        ▼              ▼
                                  palpitations ──▶ analysis ◀──▶ Ollama
                                        │              │
                                        ▼              ▼
                                   messages / charts / report
                                        │
                                        ▼
                                 telegram_bot (commands, alerts, files)
```

## Module contracts

All datetimes are timezone-aware UTC internally; convert with `utils.to_local(dt, profile.timezone)`
only when rendering.  Garmin's intraday heart rate is sampled every **2 minutes**.

### `config.py`
`load_config(path) -> AppConfig`. `AppConfig` has `timezone`, `database`, `data_dir`, `telegram`
(`TelegramConfig`: `bot_token`, `admin_chat_ids`), `ollama` (`OllamaConfig`), `schedule`
(`ScheduleConfig`), `profiles` (`list[ProfileConfig]`). `ProfileConfig` has `name`, `slug`,
`garmin`, `telegram_chat_ids`, `timezone`, `persona`, `goals`, `language`, `features`
(`FeatureFlags`), `palpitations` (`PalpitationConfig`), `alerts` (`AlertConfig`).
`AppConfig.profiles_for_chat(chat_id)` / `is_authorised(chat_id)` implement access control.

### `models.py`
`DaySnapshot` (summary, hr, steps, stress, body_battery, activities, sleep, hrv, readiness,
spo2, respiration, device, raw, errors), `Episode`, `SymptomReport`, `EpisodeAssessment`,
`CoachingAdvice`, `Alert`. Constants `ASSESSMENT_*`, `EPISODE_KIND_*`, `EPISODE_SOURCE_*`.

### `storage.py`
`Storage(path)` with `save_snapshot`, `get_snapshot_row(s)`, `get_snapshot_raw`, `save_hr_samples`,
`get_hr_samples`, `hr_near`, `upsert_episode -> (Episode, is_new)`, `get_episode(s)`,
`update_episode_assessment`, `mark_episode_notified`, `set_episode_felt`, `unassessed_episodes`,
`add_symptom`, `get_symptoms`, `symptoms_without_hr`, `update_symptom_hr`, `save_analysis`,
`get_latest_analysis`, `notification_sent`, `mark_notification`, `kv_get`, `kv_set`.
Snapshot rows are dicts whose keys are the `daily_snapshots` columns (`day`, `total_steps`,
`resting_hr`, `sleep_seconds`, `sleep_score`, `hrv_last_night`, `avg_stress`, `body_battery_high`,
`abnormal_hr_alerts`, `activities` (list of dicts) ...).

### `palpitations.py`
`detect_episodes(snapshot, cfg, extra_sleep_windows=None) -> list[Episode]` and
`hr_context_at(snapshot, when, cfg) -> (hr, baseline)`.

### `llm.py` / `claude_cli_client.py`
```python
class LLMError(RuntimeError): ...            # base class; OllamaError subclasses it
class LLMClient(Protocol):                    # model, is_available(), chat_json(), chat_text()
def make_llm_client(cfg: AppConfig) -> LLMClient | None   # from cfg.llm.backend
class ClaudeCliClient:                        # runs: claude -p --output-format json --json-schema ...
    def __init__(self, cfg: ClaudeCliConfig, workdir=None, runner=subprocess.run, env=None)
```
`ClaudeCliClient` passes the user prompt on stdin, the system prompt via `--system-prompt`,
disables tools (`--tools "" --permission-mode dontAsk`), keeps `--max-turns >= 2` (structured
output is delivered through a tool call) and reads `structured_output` / `result` from the JSON
envelope. Authentication is the CLI's own (`claude setup-token` -> `CLAUDE_CODE_OAUTH_TOKEN`, or
`ANTHROPIC_API_KEY`).

### `ollama_client.py`
```python
class OllamaError(LLMError): ...
class OllamaClient:
    def __init__(self, cfg: OllamaConfig, session: requests.Session | None = None): ...
    model: str                      # property, cfg.model
    def is_available(self) -> bool  # GET /api/tags succeeds
    def has_model(self, name: str | None = None) -> bool
    def pull_model(self, name: str | None = None) -> None    # POST /api/pull {"name":..., "stream": false}
    def ensure_model(self) -> None  # pull when cfg.auto_pull and model missing
    def chat_json(self, system: str, user: str, schema: dict) -> dict
        # POST /api/chat {"model", "messages":[system,user], "stream": false, "format": schema,
        #                 "options": {"temperature", "num_ctx"}, "keep_alive"}
        # parse message.content as JSON; if that fails retry once with "format": "json";
        # raise OllamaError on HTTP errors / timeouts / unparseable output
    def chat_text(self, system: str, user: str) -> str
```

### `analysis.py`
```python
COACHING_SCHEMA: dict; EPISODE_SCHEMA: dict   # JSON schemas used with chat_json
def build_history_payload(rows: list[dict], today: DaySnapshot | None) -> dict
def daily_coaching(client, profile, rows_7d, today, episodes_today, symptoms_today) -> CoachingAdvice
def weekly_review(client, profile, rows_this_week, rows_prev_week, episodes_week) -> CoachingAdvice  # period="weekly"
def assess_episode(client, profile, episode, snapshot, recent_symptoms) -> EpisodeAssessment
def doctor_narrative(client, profile, episodes, symptoms, rows) -> str   # short factual paragraph
def rule_based_coaching(profile, rows_7d, today) -> CoachingAdvice        # fallback, model="rules"
```
Prompts must: hand the model only numbers from the data, state it is not a diagnosis, use plain
language in `profile.language`, respect `profile.persona` / `profile.goals`, and for episodes
return one of `ASSESSMENTS` with a one-line factual `doctor_note`.

### `messages.py` (Telegram HTML, every dynamic value escaped, < 4000 chars)
```python
def esc(text) -> str
def coaching_block(advice: CoachingAdvice) -> str
def morning_brief(profile, today, yesterday_row, overnight_episodes, coaching) -> str
def evening_summary(profile, today, rows_7d, episodes_today, symptoms_today, coaching) -> str
def weekly_review(profile, rows_this_week, rows_prev_week, episodes, coaching) -> str
def today_status(profile, snap, episodes_today) -> str
def sleep_message(profile, snap) -> str
def hr_message(profile, snap, episodes) -> str
def steps_message(profile, snap, rows_7d) -> str
def episode_alert(profile, ep, assessment) -> str
def episodes_list(profile, episodes, symptoms, days) -> str
def symptom_logged(profile, rep) -> str
def alert_message(alert: Alert) -> str
def doctor_report_caption(profile, days, n_episodes, n_symptoms) -> str
def help_text(is_admin: bool) -> str
def mfa_request(profile) -> str
```

### `charts.py` (matplotlib, Agg backend, returns PNG bytes)
```python
def hr_day_chart(snap, episodes, symptoms, tz) -> bytes
def episode_chart(samples: list[HRSample], ep: Episode, tz, minutes_around=45) -> bytes
def weekly_trend_chart(rows: list[dict]) -> bytes
def episode_histograms(episodes, tz) -> bytes
```

### `report.py`
```python
@dataclass
class ReportFiles: pdf_path: Path; csv_path: Path; summary: str; n_episodes: int; n_symptoms: int
def generate_doctor_report(profile, storage, start: date, end: date, out_dir: Path,
                           narrative: str | None = None) -> ReportFiles
```
PDF built with `matplotlib.backends.backend_pdf.PdfPages`: cover + summary + disclaimer, frequency
statistics (per week, hour of day, asleep vs awake, felt vs detected), episode table, symptom
table, resting-HR/sleep trend, zoomed chart per episode (max 12). CSV has one row per episode
and per symptom (`record_type` column).

### `telegram_bot.py`
```python
class MfaBroker:
    def wait_for_code(self, profile_name: str, timeout: float = 600) -> str   # blocking, worker thread
    def submit(self, profile_name: str, code: str) -> bool
    def pending(self) -> list[str]
class HealthBot:
    def __init__(self, config: AppConfig, service: MonitorService): ...
    app: telegram.ext.Application
    mfa: MfaBroker
    async def send_text(self, chat_ids, text, reply_markup=None) -> None
    async def send_photo(self, chat_ids, png: bytes, caption: str) -> None
    async def send_document(self, chat_ids, path, caption: str) -> None
    def run(self) -> None      # app.run_polling(); the scheduler registers jobs before this
```
Commands: `/start /help /profiles /today /yesterday /sleep /hr [YYYY-MM-DD] /steps /episodes [days]
/palp [HH:MM] [note] /note <text> /report [days] /analyze /status /mfa <code>`.  When a chat can see
several profiles the first argument may name the profile (`/today Dad`).  Inline buttons on an
episode alert: callback data `felt:<episode_id>:yes|no`.  Every handler checks
`config.is_authorised(chat_id)` and ignores strangers.  Blocking service calls go through
`asyncio.to_thread`.

### `service.py` (`MonitorService`)
```python
@dataclass
class PollResult: profile: str; snapshot: DaySnapshot | None; new_episodes: list[Episode];
                  alerts: list[Alert]; error: str | None = None
class MonitorService:
    config: AppConfig; storage: Storage; ollama: OllamaClient | None
    def profile_for_chat(self, chat_id: int, name: str | None) -> ProfileConfig
    def poll(self, profile, day=None, light=True) -> PollResult      # fetch, store, detect, assess
    def backfill(self, profile, days) -> None
    def morning_brief_text(self, profile) -> str
    def evening_summary_text(self, profile, day=None) -> str
    def weekly_review_text(self, profile) -> str
    def today_text(self, profile) -> str; yesterday_text; sleep_text(profile, day=None)
    def hr_chart(self, profile, day=None) -> tuple[str, bytes | None]
    def steps_text(self, profile) -> str
    def episodes_text(self, profile, days=7) -> str
    def log_symptom(self, profile, event_time: datetime | None, note: str, chat_id: int) -> str
    def set_felt(self, profile, episode_id: int, felt: bool) -> str
    def doctor_report(self, profile, days=30) -> ReportFiles
    def analyze_now(self, profile) -> str
    def status_text(self) -> str
    def episode_alert_payload(self, ep) -> tuple[str, bytes | None]  # text + zoom chart
```

### `scheduler.py`
Registers PTB `JobQueue` jobs: poll every `poll_minutes`; morning brief; evening summary;
weekly review; monthly doctor report; startup backfill. Each job iterates profiles, respects
`FeatureFlags`, dedupes alerts via `storage.notification_sent(key)`.

## Data flow for a palpitation

1. Poll fetches today's HR/steps/stress/activities (light poll every 15 min).
2. `detect_episodes` yields candidates; `storage.upsert_episode` merges overlapping windows so an
   ongoing episode is updated, not duplicated.
3. New episodes above `alert_min_confidence` are sent to Ollama for an assessment, stored, and a
   Telegram alert with a zoomed chart and **"I felt it" / "Didn't notice"** buttons is sent.
4. `/palp` records a person-reported symptom; the service attaches the HR near that time.
5. `/report 30` (or the monthly job) produces the PDF + CSV for the doctor.
