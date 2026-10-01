"""Configuration loading.

Configuration lives in a YAML file (see ``config.example.yaml``).  Any string
value may reference environment variables with ``${VAR}`` or ``${VAR:-default}``
so secrets can stay in the NAS's Docker environment instead of the file.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

logger = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


def substitute_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:-default}`` in strings."""
    env = os.environ if env is None else env
    if isinstance(value, str):

        def _sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in env and env[name] != "":
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable {name} is not set (referenced in config)")

        return _ENV_RE.sub(_sub, value)
    if isinstance(value, list):
        return [substitute_env(v, env) for v in value]
    if isinstance(value, dict):
        return {k: substitute_env(v, env) for k, v in value.items()}
    return value


@dataclass(slots=True)
class GarminConfig:
    email: str | None = None
    password: str | None = None
    tokenstore: str = ""
    is_cn: bool = False


@dataclass(slots=True)
class PalpitationConfig:
    """Thresholds for the at-rest heart-rate excursion detector.

    Garmin Connect exposes intraday heart rate at 2-minute resolution, so an
    episode has to last a few minutes to be visible at all.  The defaults are
    deliberately conservative to avoid alarm fatigue; tune them per person.
    """

    rest_hr_threshold: int = 100  # bpm while at rest -> candidate (tachycardia band)
    rise_over_baseline: int = 35  # bpm above the rolling at-rest baseline -> candidate
    spike_hr_threshold: int = 120  # single 2-min sample this high at rest -> spike episode
    spike_jump_bpm: int = 40  # jump from previous at-rest sample -> spike episode
    nocturnal_hr_threshold: int = 90  # bpm while asleep -> candidate
    nocturnal_rise: int = 25  # bpm above sleeping baseline -> candidate
    min_duration_minutes: int = 4  # sustained episode length (2 samples at 2-min spacing)
    max_steps_in_window: int = 120  # steps in the enclosing 15-min bucket to still count as "rest"
    baseline_window_minutes: int = 30
    post_activity_cooldown_minutes: int = 10
    gap_tolerance_samples: int = 1
    alert_min_confidence: float = 0.45
    notify: bool = True
    quiet_hours: tuple[int, int] | None = None  # e.g. (23, 7): buffer non-critical alerts overnight


@dataclass(slots=True)
class AlertConfig:
    rhr_above_7d_avg: int = 10  # bpm above 7-day average resting HR
    rhr_absolute_high: int = 90
    spo2_below: int = 90
    body_battery_below: int = 15
    hrv_alert_statuses: tuple[str, ...] = ("LOW", "POOR")
    no_sync_hours: int = 12
    sedentary_nudge_hour: int = 15  # local hour to check for very low steps
    sedentary_nudge_steps: int = 1500
    abnormal_hr_alerts: bool = True  # notify when Garmin's own abnormal-HR alert count rises
    low_hr_below: int | None = None  # awake heart rate under this for low_hr_minutes -> alert (off by default)
    low_hr_minutes: int = 10
    only: tuple[str, ...] | None = None  # emit only these rule kinds, e.g. [no_sync, low_hr]


@dataclass(slots=True)
class FeatureFlags:
    morning_brief: bool = True
    evening_summary: bool = True
    weekly_review: bool = True
    daily_coaching: bool = True
    palpitations: bool = False
    doctor_report: bool = False
    alerts: bool = True
    heart_review: bool = False  # evening message = palpitation list only (time + HR), not the full summary
    monthly_summary: bool = False  # 1st of month: calendar + AI report (heart) or 30-day progress picture
    workout_nudge: bool = False  # mid-week nudge when behind the weekly workout goal


@dataclass(slots=True)
class MedicationConfig:
    name: str = ""  # e.g. "Metoprolol tartrate 50 mg"
    times: tuple[str, ...] = ()  # local HH:MM reminder times; empty = no reminders
    started: str | None = None  # YYYY-MM-DD; marked on the monthly calendar and given to the AI report


@dataclass(slots=True)
class ProfileConfig:
    name: str
    garmin: GarminConfig
    telegram_chat_ids: list[int] = field(default_factory=list)
    telegram_threads: dict[int, int] = field(default_factory=dict)  # chat_id -> forum topic id
    timezone: str | None = None
    persona: str = ""  # free text handed to the model, e.g. "68-year-old, mild hypertension"
    goals: str = ""  # free text, e.g. "walk 6000 steps a day, sleep 7h"
    language: str = "English"
    features: FeatureFlags = field(default_factory=FeatureFlags)
    palpitations: PalpitationConfig = field(default_factory=PalpitationConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    medication: MedicationConfig = field(default_factory=MedicationConfig)

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or "profile"


@dataclass(slots=True)
class TelegramConfig:
    bot_token: str
    admin_chat_ids: list[int] = field(default_factory=list)
    admin_threads: dict[int, int] = field(default_factory=dict)  # chat_id -> forum topic id
    parse_mode: str = "HTML"
    # Added to every menu command (e.g. "dad_" -> /dad_today) so bots sharing one group
    # don't list the same names; the plain names keep working.
    command_prefix: str = ""


@dataclass(slots=True)
class OllamaConfig:
    enabled: bool = True
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    timeout_seconds: int = 180
    keep_alive: str = "30m"
    temperature: float = 0.2
    num_ctx: int = 8192
    auto_pull: bool = False


@dataclass(slots=True)
class ClaudeCliConfig:
    """Settings for the Claude Code CLI backend (``claude -p``)."""

    command: str = "claude"
    model: str = "sonnet"  # alias or full model id passed to --model
    timeout_seconds: int = 240
    max_turns: int = 3  # structured output needs >= 2 (it is delivered via a tool call)
    effort: str | None = None  # low | medium | high (omit = CLI default)
    max_budget_usd: float | None = None  # per-call ceiling, API-key users
    bare: bool = False  # --bare: API-key only, skips hooks/settings; not for subscription logins
    workdir: str | None = None  # empty directory the CLI runs in (default: <data_dir>/claude-workdir)
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LLMConfig:
    backend: str = "claude-cli"  # claude-cli | ollama | none
    claude: ClaudeCliConfig = field(default_factory=ClaudeCliConfig)


@dataclass(slots=True)
class ScheduleConfig:
    poll_minutes: int = 15
    morning_brief: str = "07:30"
    evening_summary: str = "21:00"
    weekly_review_day: str = "sun"  # mon..sun
    weekly_review_time: str = "19:00"
    doctor_report_day_of_month: int = 1
    doctor_report_time: str = "09:00"
    doctor_report_days: int = 30
    backfill_days: int = 7  # days of history to fetch on first start
    monthly_summary_day: int = 1  # previous month's summary (features.monthly_summary)
    monthly_summary_time: str = "09:00"
    workout_nudge_day: str = "thu"  # features.workout_nudge
    workout_nudge_time: str = "19:00"


@dataclass(slots=True)
class AppConfig:
    timezone: str
    database: str
    data_dir: str
    telegram: TelegramConfig
    ollama: OllamaConfig
    schedule: ScheduleConfig
    profiles: list[ProfileConfig]
    log_level: str = "INFO"
    llm: LLMConfig = field(default_factory=LLMConfig)
    vault_dir: str | None = None  # Obsidian vault: movement log + AI memory (optional; see vault.py)
    backup_dir: str | None = None  # nightly copies of the database (optional; keeps the newest BACKUP_KEEP)

    def thread_for(self, chat_id: int, profile: ProfileConfig | None = None) -> int | None:
        """Forum topic (message_thread_id) to post into for ``chat_id``, if configured."""
        if profile is not None and chat_id in profile.telegram_threads:
            return profile.telegram_threads[chat_id]
        for p in self.profiles:
            if chat_id in p.telegram_threads:
                return p.telegram_threads[chat_id]
        return self.telegram.admin_threads.get(chat_id)

    def profile(self, name: str) -> ProfileConfig:
        for p in self.profiles:
            if p.name.lower() == name.lower() or p.slug == name.lower():
                return p
        raise ConfigError(f"Unknown profile {name!r}; known: {[p.name for p in self.profiles]}")

    def profiles_for_chat(self, chat_id: int) -> list[ProfileConfig]:
        """Profiles a Telegram chat may query.  Admins see everything."""
        if chat_id in self.telegram.admin_chat_ids:
            return list(self.profiles)
        return [p for p in self.profiles if chat_id in p.telegram_chat_ids]

    def is_authorised(self, chat_id: int) -> bool:
        return bool(self.profiles_for_chat(chat_id))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


_CHAT_TARGET_RE = re.compile(r"^\s*(-?\d+)\s*(?:[:/]\s*(\d+))?\s*$")


def parse_chat_targets(value: Any) -> tuple[list[int], dict[int, int]]:
    """Parse Telegram destinations.

    Accepts an int, a string, a list of them, or dicts ``{chat_id, thread_id}``.
    A string may carry a forum topic: ``"-1002069000031:2665"`` (also ``/``),
    which is what a ``t.me/c/2069000031/2665`` link means for the Bot API
    (``-100`` prefix + internal id, topic 2665).
    Returns ``(chat_ids, {chat_id: thread_id})``.
    """
    if value is None:
        return [], {}
    if isinstance(value, int | str | dict):
        value = [value]
    ids: list[int] = []
    threads: dict[int, int] = {}
    for v in value:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        if isinstance(v, dict):
            chat = v.get("chat_id")
            thread = v.get("thread_id") or v.get("message_thread_id")
            try:
                chat_id = int(str(chat).strip())
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"Chat target {v!r} needs an integer chat_id") from exc
            if thread is not None:
                threads[chat_id] = int(thread)
            ids.append(chat_id)
            continue
        if isinstance(v, bool):
            raise ConfigError(f"Chat id {v!r} is not an integer")
        m = _CHAT_TARGET_RE.match(str(v))
        if not m:
            raise ConfigError(f"Chat id {v!r} is not an integer (use 123456 or -100123456:topic)")
        chat_id = int(m.group(1))
        ids.append(chat_id)
        if m.group(2):
            threads[chat_id] = int(m.group(2))
    # de-duplicate, keep order
    seen: set[int] = set()
    unique = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            unique.append(cid)
    return unique, threads


def _as_int_list(value: Any) -> list[int]:
    return parse_chat_targets(value)[0]


def _check_tz(name: str | None, where: str) -> None:
    if not name:
        return
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"{where}: unknown timezone {name!r}") from exc


def _dataclass_from(cls: type, data: dict[str, Any] | None, where: str) -> Any:
    """Build a flat dataclass from a dict, ignoring unknown keys with a warning."""
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a mapping")
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            logger.warning("Ignoring unknown config key %s.%s", where, key)
            continue
        kwargs[key] = value
    try:
        obj = cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc
    # Light coercion for tuples declared as lists in YAML
    for name, f in fields.items():
        val = getattr(obj, name)
        if isinstance(val, list) and "tuple" in str(f.type):
            setattr(obj, name, tuple(val))
    return obj


def _parse_profile(data: dict[str, Any], defaults: dict[str, Any], data_dir: str) -> ProfileConfig:
    if not isinstance(data, dict) or not data.get("name"):
        raise ConfigError("Each profile needs a 'name'")
    name = str(data["name"])
    where = f"profiles[{name}]"
    garmin_raw = data.get("garmin") or {}
    if not isinstance(garmin_raw, dict):
        raise ConfigError(f"{where}.garmin must be a mapping")
    tokenstore = garmin_raw.get("tokenstore") or str(
        Path(data_dir) / "tokens" / re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    )
    garmin = GarminConfig(
        email=garmin_raw.get("email") or None,
        password=garmin_raw.get("password") or None,
        tokenstore=str(tokenstore),
        is_cn=_as_bool(garmin_raw.get("is_cn"), False),
    )
    features_raw = {**(defaults.get("features") or {}), **(data.get("features") or {})}
    features = FeatureFlags(
        **{k: _as_bool(v, True) for k, v in features_raw.items() if k in FeatureFlags.__dataclass_fields__}
    )
    for key in features_raw:
        if key not in FeatureFlags.__dataclass_fields__:
            logger.warning("Ignoring unknown feature flag %s.%s", where, key)
    palp_raw = {**(defaults.get("palpitations") or {}), **(data.get("palpitations") or {})}
    alerts_raw = {**(defaults.get("alerts") or {}), **(data.get("alerts") or {})}
    chat_ids, threads = parse_chat_targets(data.get("telegram_chat_ids"))
    profile = ProfileConfig(
        name=name,
        garmin=garmin,
        telegram_chat_ids=chat_ids,
        telegram_threads=threads,
        timezone=data.get("timezone") or None,
        persona=str(data.get("persona") or ""),
        goals=str(data.get("goals") or ""),
        language=str(data.get("language") or defaults.get("language") or "English"),
        features=features,
        palpitations=_dataclass_from(PalpitationConfig, palp_raw, f"{where}.palpitations"),
        alerts=_dataclass_from(AlertConfig, alerts_raw, f"{where}.alerts"),
        medication=_dataclass_from(MedicationConfig, data.get("medication"), f"{where}.medication"),
    )
    _check_tz(profile.timezone, where)
    qh = profile.palpitations.quiet_hours
    if qh is not None and (len(qh) != 2 or not all(0 <= int(h) <= 23 for h in qh)):
        raise ConfigError(f"{where}.palpitations.quiet_hours must be [start_hour, end_hour]")
    return profile


def parse_config(raw: dict[str, Any]) -> AppConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a mapping")
    raw = substitute_env(raw)
    timezone = str(raw.get("timezone") or "UTC")
    _check_tz(timezone, "timezone")
    data_dir = str(raw.get("data_dir") or "/data")
    database = str(raw.get("database") or str(Path(data_dir) / "monitor.db"))

    tg_raw = raw.get("telegram") or {}
    if not isinstance(tg_raw, dict) or not tg_raw.get("bot_token"):
        raise ConfigError("telegram.bot_token is required")
    admin_ids, admin_threads = parse_chat_targets(tg_raw.get("admin_chat_ids"))
    telegram = TelegramConfig(
        bot_token=str(tg_raw["bot_token"]),
        admin_chat_ids=admin_ids,
        admin_threads=admin_threads,
        parse_mode=str(tg_raw.get("parse_mode") or "HTML"),
        command_prefix=str(tg_raw.get("command_prefix") or ""),
    )
    if not re.fullmatch(r"[a-z0-9_]{0,12}", telegram.command_prefix):
        raise ConfigError("telegram.command_prefix: use up to 12 of a-z, 0-9 and _")

    ollama = _dataclass_from(OllamaConfig, raw.get("ollama") or {}, "ollama")
    ollama.enabled = _as_bool(ollama.enabled, True)
    ollama.base_url = str(ollama.base_url).rstrip("/")

    llm_raw = raw.get("llm") or {}
    if not isinstance(llm_raw, dict):
        raise ConfigError("llm must be a mapping")
    claude_cfg = _dataclass_from(ClaudeCliConfig, llm_raw.get("claude") or {}, "llm.claude")
    claude_cfg.bare = _as_bool(claude_cfg.bare, False)
    claude_cfg.extra_args = [str(a) for a in (claude_cfg.extra_args or [])]
    claude_cfg.env = {str(k): str(v) for k, v in (claude_cfg.env or {}).items()}
    if not claude_cfg.workdir:
        claude_cfg.workdir = f"{data_dir}/claude-workdir"  # same form as llm.make_llm_client
    backend = str(llm_raw.get("backend") or ("ollama" if raw.get("ollama") and not llm_raw else "claude-cli")).lower()
    backend = backend.replace("_", "-")
    if backend in {"claude", "claude-code"}:
        backend = "claude-cli"
    if backend not in {"claude-cli", "ollama", "none"}:
        raise ConfigError("llm.backend must be claude-cli, ollama or none")
    llm = LLMConfig(backend=backend, claude=claude_cfg)
    if backend == "ollama":
        ollama.enabled = True
    schedule = _dataclass_from(ScheduleConfig, raw.get("schedule") or {}, "schedule")
    if int(schedule.poll_minutes) < 1:
        raise ConfigError("schedule.poll_minutes must be >= 1")
    from .utils import parse_hhmm  # local import to avoid a cycle at import time

    for key in ("morning_brief", "evening_summary", "weekly_review_time", "doctor_report_time"):
        parse_hhmm(getattr(schedule, key))
    if str(schedule.weekly_review_day).lower()[:3] not in {
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
    }:
        raise ConfigError("schedule.weekly_review_day must be mon..sun")

    defaults = raw.get("profile_defaults") or {}
    profiles_raw = raw.get("profiles") or []
    if not profiles_raw:
        raise ConfigError("At least one profile is required")
    profiles = [_parse_profile(p, defaults, data_dir) for p in profiles_raw]
    names = [p.name.lower() for p in profiles]
    if len(set(names)) != len(names):
        raise ConfigError("Profile names must be unique")
    for p in profiles:
        if not p.timezone:
            p.timezone = timezone
        if not p.telegram_chat_ids and not telegram.admin_chat_ids:
            logger.warning("Profile %s has no telegram_chat_ids and no admins are set", p.name)

    return AppConfig(
        timezone=timezone,
        database=database,
        data_dir=data_dir,
        telegram=telegram,
        ollama=ollama,
        schedule=schedule,
        profiles=profiles,
        log_level=str(raw.get("log_level") or "INFO").upper(),
        llm=llm,
        vault_dir=str(raw.get("vault_dir") or "") or None,
        backup_dir=str(raw.get("backup_dir") or "") or None,
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the YAML config at ``path`` (default: ``$GHM_CONFIG`` or ``config.yaml``)."""
    path = Path(path or os.environ.get("GHM_CONFIG") or "config.yaml")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = parse_config(raw)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    return cfg
