"""Prompts, schemas and validation for the Ollama-backed analysis.

Three kinds of output are produced here:

* **Coaching** (daily / weekly): a short "do more / do less" list built from the
  last days of steps, sleep, resting heart rate, stress and body battery.
* **Episode assessment**: a plain-language classification of one at-rest
  heart-rate excursion (``likely_exertion`` / ``likely_artifact`` /
  ``possible_palpitation`` / ``unclear``) plus a one-line factual note for the
  doctor.
* **Doctor narrative**: a short factual paragraph summarising the diary.

Design rules (see ``docs/ARCHITECTURE.md``):

* the model only ever sees *numbers* from Garmin (compact JSON), never guesses;
* every prompt states that this is a symptom diary / lifestyle coaching, not a
  diagnosis, and asks for calm language in ``profile.language``;
* every model answer is validated and clamped before it becomes a dataclass;
* :func:`rule_based_coaching` produces useful advice with no LLM at all, so a
  message can always be sent even when Ollama is down.

All functions that take a ``client`` propagate :class:`~.llm.LLMError` (e.g. ``OllamaError``)
so the caller can decide to fall back to the rule-based variants.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from .config import ProfileConfig
from .models import (
    ASSESSMENT_ARTIFACT,
    ASSESSMENT_EXERTION,
    ASSESSMENT_POSSIBLE,
    ASSESSMENT_UNCLEAR,
    ASSESSMENTS,
    CoachingAdvice,
    DaySnapshot,
    Episode,
    EpisodeAssessment,
    SymptomReport,
)
from .utils import as_float, as_int, clamp, mean, median, to_local

logger = logging.getLogger(__name__)

RULES_MODEL = "rules"
MAX_LIST_ITEMS = 4
MIN_LIST_ITEMS = 2
MAX_WATCH_OUTS = 6  # rule-based fallback may need to flag several things on a bad day
MAX_TEXT_LEN = 700
MAX_ITEM_LEN = 220
SYMPTOM_MATCH_MINUTES = 30
TRACE_BEFORE = timedelta(minutes=16)
TRACE_AFTER = timedelta(minutes=10)
LOCAL_FMT = "%d %b %H:%M"

# ---------------------------------------------------------------------------
# JSON schemas handed to Ollama's structured output
# ---------------------------------------------------------------------------

COACHING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Two or three plain sentences about the day/week, quoting the numbers given.",
        },
        "do_more": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": MIN_LIST_ITEMS,
            "maxItems": MAX_LIST_ITEMS,
            "description": "2-4 concrete things to do more of.",
        },
        "do_less": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": MIN_LIST_ITEMS,
            "maxItems": MAX_LIST_ITEMS,
            "description": "2-4 concrete things to do less of.",
        },
        "watch_outs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Things worth keeping an eye on or mentioning to a doctor; may be empty.",
        },
        "heart_note": {
            "type": "string",
            "description": "One calm sentence about heart-rate observations, or an empty string.",
        },
    },
    "required": ["summary", "do_more", "do_less", "watch_outs", "heart_note"],
}

EPISODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string", "enum": list(ASSESSMENTS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {
            "type": "string",
            "description": "Two or three plain sentences explaining the choice using the numbers.",
        },
        "doctor_note": {
            "type": "string",
            "description": "One factual sentence with time, context, heart-rate numbers and duration.",
        },
    },
    "required": ["assessment", "confidence", "reasoning", "doctor_note"],
}

SYMPTOM_WORDS = (
    "racing", "skipped_beats", "fluttering", "pounding", "dizziness", "chest_pain",
    "chest_tightness", "breathlessness", "sweating", "nausea", "fainting", "near_fainting", "anxiety",
)
TRIGGER_WORDS = (
    "caffeine", "alcohol", "poor_sleep", "stress", "exercise", "large_meal", "standing_up",
    "lying_down", "missed_medication", "heat", "unknown",
)
RED_FLAG_KEYS = ("chest_pain", "fainting", "severe_breathlessness")
RED_FLAG_SYMPTOMS = {"chest_pain", "chest_tightness", "fainting", "near_fainting"}

# Structured fields pulled out of a free-text /palp or /note message.
EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": ["palpitation", "activity", "other"]},
        "duration_minutes": {"type": ["number", "null"]},
        "still_ongoing": {"type": ["boolean", "null"]},
        "activity_before": {"type": ["string", "null"]},
        "symptoms": {"type": "array", "items": {"type": "string", "enum": list(SYMPTOM_WORDS)}},
        "severity_self_reported": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
        "heart_rate_bpm": {"type": ["number", "null"]},
        "possible_triggers": {"type": "array", "items": {"type": "string", "enum": list(TRIGGER_WORDS)}},
        "red_flags": {
            "type": "object",
            "properties": {k: {"type": "boolean"} for k in RED_FLAG_KEYS},
            "required": list(RED_FLAG_KEYS),
        },
    },
    "required": ["event_type", "symptoms", "possible_triggers", "red_flags"],
}

# Hard-coded red-flag check: works with no LLM and cannot be talked out of it.
# Deliberately broad: any chest mention alongside a palpitation log gets the 995 banner.
_RED_FLAG_RE = re.compile(
    r"chest|faint|passed\s*out|black(ed)?\s*out|collaps|"
    r"can'?t\s*breathe|cannot\s*breathe|short(ness)?\s*of\s*breath|breathless",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _tz(profile: ProfileConfig, fallback: str | None = None) -> str:
    return profile.timezone or fallback or "UTC"


def _local_str(dt: datetime | None, tz: str, fmt: str = LOCAL_FMT) -> str | None:
    if dt is None:
        return None
    try:
        return to_local(dt, tz).strftime(fmt)
    except Exception:  # pragma: no cover - bad tz names are caught by config
        return dt.strftime(fmt)


def _round(value: Any, digits: int = 1) -> float | int | None:
    f = as_float(value)
    if f is None:
        return None
    if digits <= 0:
        return int(round(f))
    return round(f, digits)


def _seconds_to_hours(value: Any) -> float | None:
    f = as_float(value)
    return None if f is None else round(f / 3600.0, 1)


def _seconds_to_minutes(value: Any) -> int | None:
    f = as_float(value)
    return None if f is None else int(round(f / 60.0))


def _fmt(value: Any, unit: str = "", digits: int = 0) -> str:
    """Render a number for a sentence, ``n/a`` when missing."""
    f = as_float(value)
    if f is None:
        return "n/a"
    if digits <= 0:
        text = f"{int(round(f)):,}"
    else:
        text = f"{f:.{digits}f}"
    return f"{text}{unit}"


def _fmt_hours(hours: float | None) -> str:
    if hours is None:
        return "n/a"
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h}h {m:02d}m"


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _clean_str(value: Any, max_len: int = MAX_TEXT_LEN) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = " ".join(str(v) for v in value.values() if v is not None)
    elif isinstance(value, list | tuple):
        value = " ".join(str(v) for v in value if v is not None)
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def _clean_list(value: Any, max_items: int = MAX_LIST_ITEMS) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # A single string: split on newlines / bullet markers when the model ignored the array.
        parts = [p.strip(" -*•\t") for p in re.split(r"[\n;]+", value)]
        items: Iterable[Any] = [p for p in parts if p]
    elif isinstance(value, list | tuple):
        items = value
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        text = _clean_str(item, MAX_ITEM_LEN)
        if text and text not in out:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _activity_line(act: dict[str, Any] | Any) -> str | None:
    """One compact line for an activity (storage dict or ``ActivityWindow``)."""
    if isinstance(act, dict):
        name = act.get("name") or act.get("type") or "activity"
        duration_s = as_float(act.get("duration_s"))
        avg_hr = as_int(act.get("avg_hr"))
        start = act.get("start")
        end = act.get("end")
    else:
        name = getattr(act, "name", "") or getattr(act, "type_key", "") or "activity"
        duration_s = as_float(getattr(act, "duration_s", None))
        avg_hr = as_int(getattr(act, "avg_hr", None))
        start = getattr(act, "start", None)
        end = getattr(act, "end", None)
    if duration_s is None and isinstance(start, datetime) and isinstance(end, datetime):
        duration_s = (end - start).total_seconds()
    parts = [str(name)]
    if duration_s is not None:
        parts.append(f"{int(round(duration_s / 60))} min")
    if avg_hr is not None:
        parts.append(f"avg HR {avg_hr}")
    return ", ".join(parts)


def _activity_lines_with_time(activities: Iterable[Any], tz: str) -> list[str]:
    lines = []
    for act in activities:
        line = _activity_line(act)
        start = act.get("start") if isinstance(act, dict) else getattr(act, "start", None)
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except ValueError:
                start = None
        when = _local_str(start, tz, "%H:%M") if isinstance(start, datetime) else None
        lines.append(f"{when} {line}" if when else str(line))
    return lines[:6]


# ---------------------------------------------------------------------------
# History payload (storage rows + today's snapshot -> compact numbers)
# ---------------------------------------------------------------------------


def snapshot_to_row(snap: DaySnapshot) -> dict[str, Any]:
    """Project a :class:`DaySnapshot` onto the ``daily_snapshots`` column names.

    Mirrors ``Storage.save_snapshot`` so today's live data can be handled exactly
    like stored history rows.
    """
    s = snap.summary
    sl = snap.sleep
    hrv = snap.hrv
    rd = snap.readiness
    resp = snap.respiration
    activities = [
        {
            "name": a.name,
            "type": a.type_key,
            "start": a.start,
            "end": a.end,
            "duration_s": a.duration_s,
            "avg_hr": a.avg_hr,
            "max_hr": a.max_hr,
            "distance_m": a.distance_m,
            "source": a.source,
        }
        for a in snap.activities
    ]
    return {
        "profile": snap.profile,
        "day": snap.date_str,
        "tz": snap.tz,
        "total_steps": s.total_steps,
        "step_goal": s.step_goal,
        "distance_m": s.distance_m,
        "active_kcal": s.active_kcal,
        "min_hr": s.min_hr,
        "max_hr": s.max_hr,
        "resting_hr": s.resting_hr,
        "resting_hr_7d_avg": s.resting_hr_7d_avg,
        "sleep_seconds": sl.total_seconds if sl else None,
        "sleep_score": sl.score if sl else None,
        "sleep_deep_seconds": sl.deep_seconds if sl else None,
        "sleep_awake_seconds": sl.awake_seconds if sl else None,
        "sleep_start": sl.start if sl else None,
        "sleep_end": sl.end if sl else None,
        "hrv_last_night": hrv.last_night_avg if hrv else None,
        "hrv_weekly_avg": hrv.weekly_avg if hrv else None,
        "hrv_status": hrv.status if hrv else None,
        "avg_stress": s.avg_stress,
        "max_stress": s.max_stress,
        "high_stress_seconds": s.high_stress_seconds,
        "body_battery_high": s.body_battery_high,
        "body_battery_low": s.body_battery_low,
        "body_battery_latest": s.body_battery_latest,
        "moderate_intensity_min": s.moderate_intensity_min,
        "vigorous_intensity_min": s.vigorous_intensity_min,
        "active_seconds": s.active_seconds,
        "highly_active_seconds": s.highly_active_seconds,
        "sedentary_seconds": s.sedentary_seconds,
        "avg_spo2": (
            snap.spo2.average if snap.spo2 and snap.spo2.average is not None else s.avg_spo2
        ),
        "lowest_spo2": (
            snap.spo2.lowest if snap.spo2 and snap.spo2.lowest is not None else s.lowest_spo2
        ),
        "avg_waking_respiration": resp.avg_waking if resp else s.avg_waking_respiration,
        "readiness_score": rd.score if rd else None,
        "abnormal_hr_alerts": s.abnormal_hr_alerts,
        "activities_count": len(snap.activities),
        "activities": activities,
    }


def compact_day(row: dict[str, Any] | None) -> dict[str, Any]:
    """Reduce one snapshot row to the handful of numbers a prompt needs."""
    row = row or {}
    active_s = (as_float(row.get("active_seconds")) or 0.0) + (
        as_float(row.get("highly_active_seconds")) or 0.0
    )
    has_active = (
        row.get("active_seconds") is not None or row.get("highly_active_seconds") is not None
    )
    return {
        "day": row.get("day"),
        "steps": as_int(row.get("total_steps")),
        "step_goal": as_int(row.get("step_goal")),
        "resting_hr": as_int(row.get("resting_hr")),
        "max_hr": as_int(row.get("max_hr")),
        "sleep_h": _seconds_to_hours(row.get("sleep_seconds")),
        "sleep_score": as_int(row.get("sleep_score")),
        "deep_sleep_min": _seconds_to_minutes(row.get("sleep_deep_seconds")),
        "awake_min": _seconds_to_minutes(row.get("sleep_awake_seconds")),
        "hrv": _round(row.get("hrv_last_night"), 0),
        "hrv_status": row.get("hrv_status"),
        "stress_avg": as_int(row.get("avg_stress")),
        "stress_max": as_int(row.get("max_stress")),
        "high_stress_min": _seconds_to_minutes(row.get("high_stress_seconds")),
        "body_battery_high": as_int(row.get("body_battery_high")),
        "body_battery_low": as_int(row.get("body_battery_low")),
        "body_battery_latest": as_int(row.get("body_battery_latest")),
        "moderate_min": as_int(row.get("moderate_intensity_min")),
        "vigorous_min": as_int(row.get("vigorous_intensity_min")),
        "active_min": int(round(active_s / 60.0)) if has_active else None,
        "sedentary_h": _seconds_to_hours(row.get("sedentary_seconds")),
        "spo2_avg": _round(row.get("avg_spo2"), 1),
        "spo2_low": as_int(row.get("lowest_spo2")),
        "respiration": _round(row.get("avg_waking_respiration"), 1),
        "readiness": as_int(row.get("readiness_score")),
        "abnormal_hr_alerts": as_int(row.get("abnormal_hr_alerts")),
        "activities": [_activity_line(a) for a in (row.get("activities") or [])][:5],
    }


_AVERAGE_KEYS = (
    "steps",
    "resting_hr",
    "sleep_h",
    "sleep_score",
    "hrv",
    "stress_avg",
    "high_stress_min",
    "body_battery_high",
    "body_battery_low",
    "active_min",
    "spo2_avg",
    "readiness",
)
_INT_AVERAGES = {
    "steps",
    "resting_hr",
    "sleep_score",
    "hrv",
    "stress_avg",
    "high_stress_min",
    "body_battery_high",
    "body_battery_low",
    "active_min",
    "readiness",
}


def _averages(days: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_days": len(days)}
    for key in _AVERAGE_KEYS:
        vals = [as_float(d.get(key)) for d in days]
        m = mean(v for v in vals if v is not None)
        if m is None:
            out[key] = None
        elif key in _INT_AVERAGES:
            out[key] = int(round(m))
        else:
            out[key] = round(m, 1)
    return out


def build_history_payload(rows: list[dict[str, Any]], today: DaySnapshot | None) -> dict[str, Any]:
    """Compact numeric history for a prompt.

    ``rows`` are ``daily_snapshots`` rows (ascending).  When ``today`` is given it
    replaces any stored row for the same day (it is fresher) and is reported
    separately under ``"today"``; ``"averages"`` are computed over the *other*
    days so today can be compared with its recent past (falls back to all days
    when there is no history).  Missing values are ``null`` = not measured.
    """
    today_row = snapshot_to_row(today) if today is not None else None
    today_day = today_row.get("day") if today_row else None
    history = [
        compact_day(r) for r in rows or [] if isinstance(r, dict) and r.get("day") != today_day
    ]
    history.sort(key=lambda d: str(d.get("day") or ""))
    today_compact = compact_day(today_row) if today_row else None
    base = history if history else ([today_compact] if today_compact else [])
    all_days = history + ([today_compact] if today_compact else [])
    payload: dict[str, Any] = {
        "days": all_days[-14:],
        "averages": _averages(base),
        "today": today_compact,
    }
    if today_compact is None and all_days:
        payload["latest"] = all_days[-1]
    return payload


# ---------------------------------------------------------------------------
# Episode / symptom summaries for prompts
# ---------------------------------------------------------------------------


def _episode_brief(ep: Episode, tz: str) -> dict[str, Any]:
    return {
        "start": _local_str(ep.start, tz),
        "duration_min": _round(ep.duration_min, 1),
        "peak_hr": ep.peak_hr,
        "baseline_hr": _round(ep.baseline_hr, 0),
        "delta_hr": _round(ep.delta_hr, 0),
        "max_jump_bpm": ep.max_jump_bpm,
        "steps_in_window": ep.steps_in_window,
        "asleep": bool(ep.asleep),
        "kind": ep.kind,
        "assessment": ep.llm_assessment,
        "felt": ep.felt,
    }


def _symptom_brief(rep: SymptomReport, tz: str) -> dict[str, Any]:
    return {
        "time": _local_str(rep.event_time, tz),
        "note": _clean_str(rep.note, 120) or None,
        "hr_at_time": rep.hr_at_time,
        "baseline_hr": _round(rep.baseline_hr, 0),
        "linked_episode": rep.episode_id,
    }


def _hour_band(dt: datetime, tz: str) -> str:
    h = to_local(dt, tz).hour
    if h < 6:
        return "night (00-06)"
    if h < 12:
        return "morning (06-12)"
    if h < 18:
        return "afternoon (12-18)"
    return "evening (18-24)"


def episode_stats(
    episodes: list[Episode],
    tz: str,
    span_days: int | None = None,
    symptoms: list[SymptomReport] | None = None,
) -> dict[str, Any]:
    """Frequency / timing statistics over a list of episodes (numbers only).

    ``span_days`` is the length of the reporting period used for the per-week
    rate (defaults to the span between the first and last episode).  A symptom
    report that overlaps an episode counts as "felt".
    """
    eps = [e for e in episodes or [] if e is not None]
    if not eps:
        return {"count": 0, "span_days": span_days}
    durations = [as_float(e.duration_min) for e in eps]
    peaks = [as_float(e.peak_hr) for e in eps]
    baselines = [as_float(e.baseline_hr) for e in eps]
    days = {to_local(e.start, tz).date() for e in eps}
    first = min(e.start for e in eps)
    last = max(e.start for e in eps)
    if not span_days or span_days < 1:
        span_days = max(1, (to_local(last, tz).date() - to_local(first, tz).date()).days + 1)
    felt = [_felt_status(e, symptoms or [])[0] for e in eps]
    return {
        "count": len(eps),
        "span_days": span_days,
        "days_with_episodes": len(days),
        "first": _local_str(first, tz),
        "last": _local_str(last, tz),
        "per_week": round(len(eps) * 7.0 / span_days, 1),
        "by_kind": dict(Counter(e.kind for e in eps)),
        "by_assessment": dict(Counter(e.llm_assessment or "not_assessed" for e in eps)),
        "by_time_of_day": dict(Counter(_hour_band(e.start, tz) for e in eps)),
        "asleep": sum(1 for e in eps if e.asleep),
        "felt_yes": sum(1 for f in felt if f == "yes"),
        "felt_no": sum(1 for f in felt if f == "did not notice"),
        "felt_unknown": sum(1 for f in felt if f not in ("yes", "did not notice")),
        "duration_min_median": _round(median(durations), 1),
        "duration_min_max": _round(max(d for d in durations if d is not None), 1)
        if any(durations)
        else None,
        "peak_hr_median": _round(median(peaks), 0),
        "peak_hr_max": max(int(p) for p in peaks if p is not None) if any(peaks) else None,
        "baseline_hr_mean": _round(mean(baselines), 0),
    }


def _matching_symptoms(ep: Episode, symptoms: list[SymptomReport]) -> list[SymptomReport]:
    tol = timedelta(minutes=SYMPTOM_MATCH_MINUTES)
    out = []
    for s in symptoms or []:
        if s.episode_id is not None and ep.id is not None and s.episode_id == ep.id:
            out.append(s)
        elif ep.start - tol <= s.event_time <= ep.end + tol:
            out.append(s)
    return out


def _felt_status(ep: Episode, symptoms: list[SymptomReport]) -> tuple[str, list[SymptomReport]]:
    matched = _matching_symptoms(ep, symptoms)
    if ep.felt is True or matched:
        return "yes", matched
    if ep.felt is False:
        return "did not notice", matched
    return "not asked / unknown", matched


def _hr_trace(ep: Episode, snapshot: DaySnapshot | None, tz: str) -> list[str]:
    """``HH:MM=bpm`` samples from shortly before the episode to shortly after it."""
    samples = list(snapshot.hr) if snapshot is not None and snapshot.hr else list(ep.samples)
    lo, hi = ep.start - TRACE_BEFORE, ep.end + TRACE_AFTER
    picked = sorted((s for s in samples if lo <= s.ts <= hi), key=lambda s: s.ts)
    if not picked:
        picked = sorted(ep.samples, key=lambda s: s.ts)
    return [f"{to_local(s.ts, tz).strftime('%H:%M')}={s.hr}" for s in picked[:40]]


def _rest_label(ep: Episode, profile: ProfileConfig) -> str:
    if ep.asleep:
        return "asleep"
    if ep.steps_in_window <= profile.palpitations.max_steps_in_window:
        return "at rest"
    return "moving"


def fallback_doctor_note(
    ep: Episode, profile: ProfileConfig, symptoms: list[SymptomReport] | None = None
) -> str:
    """One factual sentence built purely from the episode's numbers.

    Used when the model returns an empty ``doctor_note`` and as the seed example
    inside the prompt so the model copies the format.
    """
    tz = _tz(profile)
    felt, matched = _felt_status(ep, symptoms or [])
    when = _local_str(ep.start, tz) or "unknown time"
    onset = (
        f"sudden onset (jump {ep.max_jump_bpm} bpm)" if ep.max_jump_bpm >= 20 else "gradual onset"
    )
    dur = _round(ep.duration_min, 0)
    parts = [
        f"{when}, {_rest_label(ep, profile)} ({ep.steps_in_window} steps): "
        f"HR rose from {_fmt(ep.baseline_hr)} to {ep.peak_hr} bpm for {dur} min, {onset}"
    ]
    if felt == "yes":
        note = next((_clean_str(s.note, 60) for s in matched if _clean_str(s.note, 60)), "")
        parts.append(f"person reported {note}" if note else "person reported feeling it")
    elif felt == "did not notice":
        parts.append("person did not notice it")
    return ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Prompt text
# ---------------------------------------------------------------------------


def _persona_lines(profile: ProfileConfig) -> str:
    lines = [f"Person: {profile.name}."]
    if profile.persona:
        lines.append(f"About them: {profile.persona}.")
    if profile.goals:
        lines.append(f"Their goals: {profile.goals}.")
    lines.append(f"Language for the answer: {profile.language or 'English'}.")
    lines.append("Never use dashes in any text you write. Short plain sentences a 65 year old can read.")
    return "\n".join(lines)


def _coaching_system(profile: ProfileConfig, period: str) -> str:
    what = "one day" if period == "daily" else "one week compared with the week before"
    return (
        "You are a calm, practical wellness coach reviewing "
        f"{what} of data from a Garmin watch.\n"
        f"{_persona_lines(profile)}\n"
        "Rules:\n"
        "- Use ONLY the numbers provided. Never invent, estimate or round up values. "
        "A null value means it was not measured; say so instead of guessing.\n"
        "- This is general lifestyle coaching, not medical advice and not a diagnosis. "
        "Do not name diseases. If something looks worth a check, say 'worth mentioning to your doctor'.\n"
        "- Plain, short, friendly sentences suitable for an older adult. No jargon, no exclamation marks.\n"
        "- Compare today with the recent averages and with the goals; quote the actual numbers.\n"
        "- do_more and do_less must each contain 2 to 4 concrete, small, realistic actions.\n"
        "- heart_note: one calm sentence about the heart-rate observations, or an empty string.\n"
        "Answer with JSON only, matching the schema."
    )


def _episode_system(profile: ProfileConfig) -> str:
    return (
        "You review one heart-rate episode recorded by a Garmin wrist watch (optical sensor, "
        "one sample every 2 minutes; it measures rate, not rhythm).\n"
        f"{_persona_lines(profile)}\n"
        "Your job is to sort the episode into exactly one category using ONLY the numbers given:\n"
        f"- '{ASSESSMENT_EXERTION}': movement, steps or a recorded activity can explain the rise "
        "(steps in the window, active level, movement flags, HR that climbs and settles like exercise).\n"
        f"- '{ASSESSMENT_ARTIFACT}': the pattern looks like sensor noise, e.g. one isolated extreme "
        "jump together with movement, or values that do not fit the neighbouring samples.\n"
        f"- '{ASSESSMENT_POSSIBLE}': at rest (few or no steps, not in an activity) with an abrupt "
        "onset, especially when the person reported feeling it.\n"
        f"- '{ASSESSMENT_UNCLEAR}': anything else, or when the data is too thin to say.\n"
        "This is a symptom diary for a doctor, never a diagnosis. Do not name conditions. "
        "Do not invent numbers.\n"
        "confidence: 0 to 1, how sure you are of the category.\n"
        "reasoning: two or three plain, calm sentences the person could read.\n"
        "doctor_note: ONE factual sentence with the time, context, heart-rate numbers and duration, "
        'for example "27 Sep 10:00, at rest (0 steps): HR rose from 64 to 127 bpm for 8 min, '
        'sudden onset, person reported fluttering."\n'
        f"Write in {profile.language or 'English'}. Answer with JSON only, matching the schema."
    )


def _narrative_system(profile: ProfileConfig) -> str:
    return (
        "You write a short factual paragraph (4 to 7 sentences) for a doctor, summarising a "
        "heart-rate symptom diary recorded by a Garmin wrist watch.\n"
        f"{_persona_lines(profile)}\n"
        "Rules: use only the numbers given; state how many episodes, how often, when in the day, "
        "how long, how high the heart rate went versus baseline, how many were felt by the person, "
        "and how many were reported without a detected episode. Mention that the watch is an "
        "optical wrist sensor sampled every 2 minutes and shows rate, not rhythm. No diagnosis, no "
        "treatment suggestions, no speculation. The statistics were computed by software and are "
        "correct: present them, do not interpret or recount them. Name any red flag reports "
        "(chest pain, fainting, severe breathlessness) with their dates. End with one line inviting "
        "the doctor to see the full log. Under 250 words. Plain prose, no lists, no headings."
    )


def _extract_system() -> str:
    return (
        "You are the language layer for a private family health log. You are not a doctor. "
        "Never diagnose, never suggest causes, treatment, medication or supplements.\n"
        "Convert ONE logged message into the structured fields of the schema. Answer from the "
        "message alone; do not use tools.\n"
        "- If a value is not stated, use null. Never guess a time, duration or heart rate.\n"
        "- Only use numbers that appear in the message.\n"
        f"- symptoms only from: {', '.join(SYMPTOM_WORDS)}.\n"
        f"- possible_triggers only from: {', '.join(TRIGGER_WORDS)}.\n"
        "- Set a red_flags value to true only when the message clearly states it. Do not judge "
        "whether the episode is serious, just record what was said."
    )


def extract_symptom(client: Any, note: str, logged_at: datetime, tz: str) -> dict[str, Any]:
    """Structured fields from a free-text symptom note (raises LLMError on failure)."""
    user = f"logged_at: {_local_str(logged_at, tz, '%Y-%m-%d %H:%M')}\nmessage: {note}"
    data = client.chat_json(_extract_system(), user, EXTRACT_SCHEMA)
    data = data if isinstance(data, dict) else {}
    # the schema enforces the shape, but a non-enforcing backend (Ollama) may not
    data["symptoms"] = [s for s in data.get("symptoms") or [] if s in SYMPTOM_WORDS]
    data["possible_triggers"] = [t for t in data.get("possible_triggers") or [] if t in TRIGGER_WORDS]
    flags = data.get("red_flags") if isinstance(data.get("red_flags"), dict) else {}
    data["red_flags"] = {k: flags.get(k) is True for k in RED_FLAG_KEYS}
    return data


def has_red_flag(note: str, extracted: dict[str, Any] | None = None) -> bool:
    """True if the note or the extracted fields mention chest pain, fainting or bad breathlessness."""
    if note and _RED_FLAG_RE.search(note):
        return True
    ex = extracted or {}
    return any(ex.get("red_flags", {}).values()) or bool(RED_FLAG_SYMPTOMS & set(ex.get("symptoms") or []))


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _pad_list(items: list[str], defaults: list[str]) -> list[str]:
    out = list(items)
    for d in defaults:
        if len(out) >= MIN_LIST_ITEMS:
            break
        if d not in out:
            out.append(d)
    return out[:MAX_LIST_ITEMS]


def coaching_from_dict(
    data: dict[str, Any] | None, model: str, period: str = "daily"
) -> CoachingAdvice:
    """Validate / clamp a model answer into :class:`CoachingAdvice`."""
    data = data if isinstance(data, dict) else {}
    heart = _clean_str(data.get("heart_note"), 400)
    return CoachingAdvice(
        summary=_clean_str(data.get("summary")) or "No summary was produced.",
        do_more=_clean_list(data.get("do_more")),
        do_less=_clean_list(data.get("do_less")),
        watch_outs=_clean_list(data.get("watch_outs"), max_items=MAX_LIST_ITEMS),
        heart_note=heart or None,
        model=model,
        period=period,
    )


def assessment_from_dict(
    data: dict[str, Any] | None,
    model: str,
    episode: Episode,
    profile: ProfileConfig,
    symptoms: list[SymptomReport] | None = None,
) -> EpisodeAssessment:
    """Validate / clamp a model answer into :class:`EpisodeAssessment`."""
    data = data if isinstance(data, dict) else {}
    raw_assessment = (
        _clean_str(data.get("assessment"), 60).lower().replace(" ", "_").replace("-", "_")
    )
    assessment = raw_assessment if raw_assessment in ASSESSMENTS else ASSESSMENT_UNCLEAR
    if raw_assessment and assessment != raw_assessment:
        logger.warning(
            "Unknown assessment %r from model; using %r", raw_assessment, ASSESSMENT_UNCLEAR
        )
    conf = as_float(data.get("confidence"))
    if conf is None:
        logger.warning("Model returned no usable confidence; recording 0.0")
        conf = 0.0
    if 5.0 <= conf <= 100.0:
        conf = conf / 100.0  # a model that answered in percent; anything else is just clamped
    conf = round(clamp(conf, 0.0, 1.0), 3)
    reasoning = _clean_str(data.get("reasoning")) or "The model gave no explanation."
    note = _clean_str(data.get("doctor_note"), 400)
    if not note:
        note = fallback_doctor_note(episode, profile, symptoms)
    return EpisodeAssessment(
        assessment=assessment,
        confidence=conf,
        reasoning=reasoning,
        doctor_note=note,
        model=model,
    )


# ---------------------------------------------------------------------------
# LLM-backed entry points
# ---------------------------------------------------------------------------


def daily_coaching(
    client: Any,
    profile: ProfileConfig,
    rows_7d: list[dict[str, Any]],
    today: DaySnapshot | None,
    episodes_today: list[Episode] | None,
    symptoms_today: list[SymptomReport] | None,
) -> CoachingAdvice:
    """Ask the model for today's "do more / do less" advice."""
    tz = _tz(profile, today.tz if today else None)
    payload = build_history_payload(rows_7d, today)
    episodes = [_episode_brief(e, tz) for e in (episodes_today or [])]
    symptoms = [_symptom_brief(s, tz) for s in (symptoms_today or [])]
    user = (
        "Data (JSON, numbers only; null = not measured):\n"
        f"{_dumps(payload)}\n"
        f"Heart-rate episodes detected today while at rest: {_dumps(episodes) if episodes else 'none'}\n"
        f"Palpitations the person reported today: {_dumps(symptoms) if symptoms else 'none'}\n"
        "Write the summary, 2-4 do_more items, 2-4 do_less items, watch_outs and heart_note. "
        "Remember: not a diagnosis; plain language; only these numbers."
    )
    logger.debug("daily_coaching prompt for %s: %d chars", profile.name, len(user))
    data = client.chat_json(_coaching_system(profile, "daily"), user, COACHING_SCHEMA)
    return coaching_from_dict(data, model=getattr(client, "model", "ollama"), period="daily")


def weekly_review(
    client: Any,
    profile: ProfileConfig,
    rows_this_week: list[dict[str, Any]],
    rows_prev_week: list[dict[str, Any]],
    episodes_week: list[Episode] | None,
) -> CoachingAdvice:
    """Ask the model to compare this week with the previous one (``period="weekly"``)."""
    tz = _tz(profile)
    this_week = build_history_payload(rows_this_week, None)
    prev_week = build_history_payload(rows_prev_week, None)
    changes: dict[str, Any] = {}
    for key in _AVERAGE_KEYS:
        a, b = this_week["averages"].get(key), prev_week["averages"].get(key)
        if a is not None and b is not None:
            changes[key] = round(a - b, 1)
    payload = {
        "this_week": this_week,
        "previous_week": prev_week,
        "change_this_minus_previous": changes,
        "episodes_this_week": episode_stats(episodes_week or [], tz, span_days=7),
    }
    user = (
        "Data (JSON, numbers only; null = not measured):\n"
        f"{_dumps(payload)}\n"
        "Write a weekly review: what went well, what slipped compared with the previous week, "
        "2-4 do_more items and 2-4 do_less items for next week, watch_outs, and a heart_note "
        "about the at-rest heart-rate episodes (count and timing only, no diagnosis). "
        "Only call something a pattern when it shows up on at least 3 days, and quote the count, "
        "for example 'under 6000 steps on 4 of 7 days'. Never invent a suggestion the data does not "
        "support. If fewer than 5 days have data, say so in the summary."
    )
    data = client.chat_json(_coaching_system(profile, "weekly"), user, COACHING_SCHEMA)
    return coaching_from_dict(data, model=getattr(client, "model", "ollama"), period="weekly")


def build_episode_prompt(
    profile: ProfileConfig,
    episode: Episode,
    snapshot: DaySnapshot | None,
    recent_symptoms: list[SymptomReport] | None,
) -> str:
    """The user message for :func:`assess_episode` (exposed for tests and debugging)."""
    tz = _tz(profile, snapshot.tz if snapshot else None)
    felt, matched = _felt_status(episode, recent_symptoms or [])
    summary = snapshot.summary if snapshot is not None else None
    day_context: dict[str, Any] = {
        "day": snapshot.date_str if snapshot else _local_str(episode.start, tz, "%Y-%m-%d"),
        "resting_hr": summary.resting_hr if summary else None,
        "resting_hr_7d_avg": summary.resting_hr_7d_avg if summary else None,
        "max_hr_of_day": summary.max_hr if summary else None,
        "garmin_abnormal_hr_alerts": summary.abnormal_hr_alerts if summary else None,
        "activities_that_day": _activity_lines_with_time(snapshot.activities, tz)
        if snapshot
        else [],
        "sleep_window": (
            f"{_local_str(snapshot.sleep.start, tz, '%H:%M')}-{_local_str(snapshot.sleep.end, tz, '%H:%M')}"
            if snapshot and snapshot.sleep and snapshot.sleep.start and snapshot.sleep.end
            else None
        ),
    }
    facts = {
        "start_local": _local_str(episode.start, tz),
        "end_local": _local_str(episode.end, tz, "%H:%M"),
        "duration_min": _round(episode.duration_min, 1),
        "peak_hr": episode.peak_hr,
        "mean_hr": _round(episode.mean_hr, 0),
        "baseline_hr": _round(episode.baseline_hr, 0),
        "delta_hr": _round(episode.delta_hr, 0),
        "max_jump_bpm": episode.max_jump_bpm,
        "steps_in_window": episode.steps_in_window,
        "activity_level": episode.activity_level,
        "asleep": bool(episode.asleep),
        "movement_fraction": _round(episode.movement_fraction, 2),
        "stress_avg": _round(episode.stress_avg, 0),
        "kind": episode.kind,
        "source": episode.source,
        "detector_confidence": _round(episode.confidence, 2),
        "person_reported_feeling_it": felt,
        "reported_symptoms_near_episode": [_symptom_brief(s, tz) for s in matched],
        "hr_trace_2min": _hr_trace(episode, snapshot, tz),
        "day_context": day_context,
    }
    other = [_symptom_brief(s, tz) for s in (recent_symptoms or []) if s not in matched][:5]
    return (
        "Episode (JSON, numbers only; null = not measured):\n"
        f"{_dumps(facts)}\n"
        f"Other palpitations the person reported recently: {_dumps(other) if other else 'none'}\n"
        f"Steps threshold used for 'at rest': {profile.palpitations.max_steps_in_window} steps per 15 min.\n"
        "Decide the category, give a confidence between 0 and 1, a short calm reasoning, and the "
        "one-sentence doctor_note built from these numbers. Not a diagnosis."
    )


def assess_episode(
    client: Any,
    profile: ProfileConfig,
    episode: Episode,
    snapshot: DaySnapshot | None,
    recent_symptoms: list[SymptomReport] | None,
) -> EpisodeAssessment:
    """Ask the model to classify one episode and write the doctor note."""
    user = build_episode_prompt(profile, episode, snapshot, recent_symptoms)
    data = client.chat_json(_episode_system(profile), user, EPISODE_SCHEMA)
    result = assessment_from_dict(
        data, getattr(client, "model", "ollama"), episode, profile, recent_symptoms or []
    )
    logger.info(
        "Episode %s at %s assessed as %s (%.2f)",
        episode.id if episode.id is not None else episode.fingerprint or "?",
        _local_str(episode.start, _tz(profile)),
        result.assessment,
        result.confidence,
    )
    return result


def _narrative_facts(
    profile: ProfileConfig,
    episodes: list[Episode],
    symptoms: list[SymptomReport],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tz = _tz(profile)
    eps = list(episodes or [])
    syms = list(symptoms or [])
    matched_ids = {id(s) for e in eps for s in _matching_symptoms(e, syms)}
    days = sorted((compact_day(r) for r in rows or []), key=lambda d: str(d.get("day") or ""))
    span_days: int | None = None
    if days and days[0]["day"] and days[-1]["day"]:
        try:
            first_d = datetime.fromisoformat(str(days[0]["day"])).date()
            last_d = datetime.fromisoformat(str(days[-1]["day"])).date()
            span_days = max(len(days), (last_d - first_d).days + 1)
        except ValueError:
            span_days = len(days)
    period = {
        "first_day": days[0]["day"] if days else None,
        "last_day": days[-1]["day"] if days else None,
        "days_of_data": len(days),
        "averages": _averages(days),
    }
    notes = [
        e.doctor_note or fallback_doctor_note(e, profile, syms)
        for e in sorted(eps, key=lambda e: e.start)
    ][:12]
    return {
        "period": period,
        "episodes": episode_stats(eps, tz, span_days=span_days, symptoms=syms),
        "symptom_reports": {
            "count": len(syms),
            "with_detected_episode": sum(1 for s in syms if id(s) in matched_ids),
            "without_detected_episode": sum(1 for s in syms if id(s) not in matched_ids),
            "examples": [_symptom_brief(s, tz) for s in syms[:5]],
        },
        "episode_notes": notes,
    }


def doctor_narrative(
    client: Any,
    profile: ProfileConfig,
    episodes: list[Episode],
    symptoms: list[SymptomReport],
    rows: list[dict[str, Any]],
) -> str:
    """Short factual paragraph for the doctor report (falls back to rules when empty)."""
    facts = _narrative_facts(profile, episodes, symptoms, rows)
    user = (
        "Diary statistics (JSON, numbers only; null = not measured):\n"
        f"{_dumps(facts)}\n"
        "Write the paragraph now. Plain prose only."
    )
    text = _clean_str(client.chat_text(_narrative_system(profile), user), 2000)
    if not text:
        logger.warning("Model returned an empty narrative; using rule-based text")
        return rule_based_narrative(profile, episodes, symptoms, rows)
    return text


# ---------------------------------------------------------------------------
# Rule-based fallbacks (no LLM)
# ---------------------------------------------------------------------------

_STEP_GOAL_RE = re.compile(r"(\d[\d,\.]*)\s*(?:k\s*)?steps", re.IGNORECASE)
_SLEEP_GOAL_RE = re.compile(
    r"(?:sleep\D{0,12}(\d+(?:\.\d+)?)\s*h)|(?:(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?\s*(?:of\s+)?sleep)",
    re.IGNORECASE,
)

DEFAULT_STEP_GOAL = 6000
DEFAULT_SLEEP_GOAL_H = 7.0
LOW_SLEEP_H = 6.0
HIGH_STRESS = 50
LOW_BODY_BATTERY_HIGH = 40

_GENERIC_DO_MORE = [
    "Take a short walk after meals, even 10 minutes counts.",
    "Keep a regular bedtime and wake time.",
    "Drink water regularly through the day.",
    "Stand up and stretch once every hour.",
]
_GENERIC_DO_LESS = [
    "Sitting for more than an hour at a stretch.",
    "Screens, heavy meals and caffeine late in the evening.",
    "Rushing: spread activity through the day instead of one big effort.",
    "Skipping rest when you feel tired.",
]


def parse_goals(goals: str | None) -> tuple[int | None, float | None]:
    """Extract ``(step_goal, sleep_goal_hours)`` from free-text goals, if present."""
    text = goals or ""
    steps: int | None = None
    sleep: float | None = None
    m = _STEP_GOAL_RE.search(text)
    if m:
        raw = m.group(1).replace(",", "")
        val = as_float(raw)
        if val is not None:
            if val < 100 and "k" in m.group(0).lower():
                val *= 1000
            steps = int(val)
    m = _SLEEP_GOAL_RE.search(text)
    if m:
        sleep = as_float(m.group(1) or m.group(2))
    return steps, sleep


def rule_based_coaching(
    profile: ProfileConfig,
    rows_7d: list[dict[str, Any]],
    today: DaySnapshot | None,
    episodes_today: list[Episode] | None = None,
    symptoms_today: list[SymptomReport] | None = None,
) -> CoachingAdvice:
    """Deterministic "do more / do less" advice that needs no LLM (``model="rules"``).

    Compares today's steps, sleep, resting HR, stress and body battery with the
    7-day averages and the profile goals; flags resting HR more than
    ``alerts.rhr_above_7d_avg`` bpm over the average, sleep under 6 h, high
    stress, low body battery, Garmin abnormal-HR alerts and any episodes.
    ``episodes_today`` / ``symptoms_today`` are optional extras beyond the
    documented signature.  Text is English regardless of ``profile.language``.
    """
    payload = build_history_payload(rows_7d, today)
    t = payload.get("today") or payload.get("latest") or compact_day(None)
    avg = payload["averages"]
    alerts = profile.alerts
    goal_steps, goal_sleep = parse_goals(profile.goals)
    goal_steps = goal_steps or t.get("step_goal") or DEFAULT_STEP_GOAL
    goal_sleep = goal_sleep or DEFAULT_SLEEP_GOAL_H

    do_more: list[str] = []
    do_less: list[str] = []
    watch: list[str] = []
    heart: list[str] = []
    summary_bits: list[str] = []

    # -- steps --------------------------------------------------------------
    steps = t.get("steps")
    avg_steps = avg.get("steps")
    summary_bits.append(
        f"Steps {_fmt(steps)} (goal {_fmt(goal_steps)}, 7-day average {_fmt(avg_steps)})"
    )
    if steps is None:
        watch.append("No step data has synced yet today.")
    else:
        if steps < goal_steps * 0.5:
            do_more.append(
                f"Walking: {_fmt(steps)} steps so far against a goal of {_fmt(goal_steps)}; "
                "a 10-15 minute stroll adds about 1,000 steps."
            )
        elif steps < goal_steps:
            do_more.append(
                f"A short evening walk would close the gap of {_fmt(goal_steps - steps)} steps to the goal."
            )
        if avg_steps and steps < avg_steps * 0.7:
            do_more.append(
                f"Moving about: today is below the recent average of {_fmt(avg_steps)} steps."
            )
        if avg_steps and steps > max(avg_steps * 1.6, goal_steps * 1.3):
            do_less.append(
                f"Big jumps in activity: {_fmt(steps)} steps is well above the usual {_fmt(avg_steps)}; "
                "spread it over the week."
            )

    # -- sleep --------------------------------------------------------------
    sleep_h = t.get("sleep_h")
    avg_sleep = avg.get("sleep_h")
    summary_bits.append(f"Sleep {_fmt_hours(sleep_h)} (goal {_fmt_hours(goal_sleep)})")
    if sleep_h is not None:
        vs_avg = (
            f" and about {_fmt(avg_sleep - sleep_h, 'h', 1)} shorter than the recent average"
            if avg_sleep and sleep_h < avg_sleep - 1.0
            else ""
        )
        if sleep_h < LOW_SLEEP_H:
            watch.append(f"Short sleep: {_fmt_hours(sleep_h)} last night, under 6 hours{vs_avg}.")
            do_more.append("Rest: an earlier wind-down tonight, aiming for 7 hours in bed.")
            do_less.append("Late evenings; short sleep tends to push resting heart rate up.")
        else:
            if sleep_h < goal_sleep:
                do_more.append(
                    f"Sleep: {_fmt_hours(sleep_h)} is a little under the {_fmt_hours(goal_sleep)} goal."
                )
            if vs_avg:
                watch.append(f"Sleep was {_fmt_hours(sleep_h)}{vs_avg}.")
    awake_min = t.get("awake_min")
    if awake_min is not None and awake_min >= 60:
        do_less.append(
            f"Fluids and screens right before bed: {awake_min} min awake during the night."
        )

    # -- resting heart rate ---------------------------------------------------
    rhr = t.get("resting_hr")
    rhr_avg = avg.get("resting_hr")
    if rhr_avg is None:
        src = snapshot_to_row(today) if today else (rows_7d[-1] if rows_7d else {})
        rhr_avg = as_int((src or {}).get("resting_hr_7d_avg"))
    summary_bits.append(f"Resting HR {_fmt(rhr)} (7-day average {_fmt(rhr_avg)})")
    if rhr is not None:
        if rhr_avg is not None and rhr > rhr_avg + alerts.rhr_above_7d_avg:
            msg = (
                f"Resting heart rate {rhr} bpm is {rhr - rhr_avg} above the recent average of "
                f"{rhr_avg}. Worth a quieter day; mention it to your doctor if it stays high."
            )
            watch.append(msg)
            heart.append(msg)
            do_less.append("Strenuous effort today while the resting heart rate is raised.")
        if rhr >= alerts.rhr_absolute_high:
            msg = f"Resting heart rate {rhr} bpm is above {alerts.rhr_absolute_high}."
            if msg not in watch:
                watch.append(msg)
            heart.append(msg)

    # -- stress ---------------------------------------------------------------
    stress = t.get("stress_avg")
    avg_stress = avg.get("stress_avg")
    summary_bits.append(f"Stress {_fmt(stress)} (average {_fmt(avg_stress)})")
    if stress is not None:
        if stress >= HIGH_STRESS:
            watch.append(f"High stress: average {stress} today (Garmin scale 0-100).")
            do_more.append(
                "Slow breathing for five minutes, twice a day, and a quiet break after lunch."
            )
            do_less.append("Caffeine after midday and rushing through the afternoon.")
        elif avg_stress is not None and stress >= avg_stress + 15:
            watch.append(f"Stress {stress} is above the recent average of {avg_stress}.")
            do_more.append(
                "A calm activity you enjoy this evening; today ran more tense than usual."
            )
    high_stress_min = t.get("high_stress_min")
    if high_stress_min is not None and high_stress_min >= 90:
        do_less.append(f"Long tense stretches: {high_stress_min} min of high stress today.")

    # -- body battery ---------------------------------------------------------
    bb_low = t.get("body_battery_low")
    bb_high = t.get("body_battery_high")
    bb_latest = t.get("body_battery_latest")
    summary_bits.append(f"Body battery high {_fmt(bb_high)}, low {_fmt(bb_low)}")
    if bb_low is not None and bb_low < alerts.body_battery_below:
        watch.append(f"Body battery dropped to {bb_low}; the body is running on reserve.")
        do_more.append("Rest and an early night; energy reserves are very low.")
        do_less.append("Extra chores or outings until energy has recovered.")
    elif bb_high is not None and bb_high < LOW_BODY_BATTERY_HIGH:
        watch.append(
            f"Body battery only reached {bb_high} today, so recovery overnight was limited."
        )
        do_more.append("Rest: keep today gentle and prioritise sleep tonight.")
    if (
        bb_latest is not None
        and bb_latest < alerts.body_battery_below
        and bb_low is not None
        and bb_low >= alerts.body_battery_below
    ):
        watch.append(f"Body battery is currently {bb_latest}.")

    # -- other signals ----------------------------------------------------------
    hrv_status = str(t.get("hrv_status") or "").upper()
    if hrv_status and hrv_status in {s.upper() for s in alerts.hrv_alert_statuses}:
        watch.append(f"Overnight HRV status is {hrv_status.lower()} (value {_fmt(t.get('hrv'))}).")
        do_more.append("Recovery: a gentler day and a full night's sleep.")
    spo2_low = t.get("spo2_low")
    if spo2_low is not None and spo2_low < alerts.spo2_below:
        watch.append(
            f"Lowest blood oxygen reading {spo2_low}% is below {alerts.spo2_below}%; "
            "a single low reading is common, repeated ones are worth mentioning to your doctor."
        )
    abnormal = t.get("abnormal_hr_alerts")
    if abnormal:
        msg = f"The watch raised {abnormal} abnormal heart-rate alert(s) today."
        watch.append(msg)
        heart.append(msg)
    eps = [e for e in (episodes_today or []) if e is not None]
    if eps:
        tz = _tz(profile, today.tz if today else None)
        peak = max(e.peak_hr for e in eps)
        times = ", ".join(_local_str(e.start, tz, "%H:%M") or "?" for e in eps[:4])
        msg = (
            f"{len(eps)} at-rest heart-rate episode(s) recorded today at {times}, "
            f"highest {peak} bpm. Kept in the diary for your doctor; note how you felt."
        )
        heart.append(msg)
        watch.append(msg)
        do_less.append(
            "Caffeine, alcohol and big meals late in the day; note when episodes happen."
        )
    syms = [s for s in (symptoms_today or []) if s is not None]
    if syms:
        heart.append(
            f"You reported {len(syms)} palpitation(s) today; they are logged for the doctor."
        )

    # -- activities / intensity ---------------------------------------------------
    acts = t.get("activities") or []
    if acts:
        summary_bits.append("Activities: " + "; ".join(acts[:3]))
    if not acts and steps is not None and steps >= goal_steps:
        summary_bits.append("Step goal reached")

    do_more = _pad_list(_clean_list(do_more), _GENERIC_DO_MORE)
    do_less = _pad_list(_clean_list(do_less), _GENERIC_DO_LESS)
    watch = _clean_list(watch, max_items=MAX_WATCH_OUTS)
    if (
        steps is not None
        and steps >= goal_steps
        and sleep_h is not None
        and sleep_h >= goal_sleep
        and not watch
    ):
        lead = "A good day: step goal and sleep goal both met. "
    elif not watch:
        lead = "A steady day. "
    else:
        lead = "A few things to keep an eye on today. "
    summary = _clean_str(lead + ". ".join(summary_bits) + ".", 900)
    heart_note = " ".join(dict.fromkeys(heart)) if heart else None
    return CoachingAdvice(
        summary=summary,
        do_more=do_more,
        do_less=do_less,
        watch_outs=watch,
        heart_note=_clean_str(heart_note, 500) or None,
        model=RULES_MODEL,
        period="daily",
    )


def rule_based_narrative(
    profile: ProfileConfig,
    episodes: list[Episode],
    symptoms: list[SymptomReport],
    rows: list[dict[str, Any]],
) -> str:
    """Factual doctor paragraph assembled from numbers only (no LLM)."""
    facts = _narrative_facts(profile, episodes, symptoms, rows)
    p, e, s = facts["period"], facts["episodes"], facts["symptom_reports"]
    avg = p["averages"]
    span = (
        f"between {p['first_day']} and {p['last_day']} ({p['days_of_data']} days of data)"
        if p["first_day"]
        else "over the reporting period"
    )
    sentences = [
        f"Heart-rate diary for {profile.name} {span}, recorded by a Garmin optical wrist sensor "
        "sampled every 2 minutes (rate only, not rhythm)."
    ]
    if e.get("count"):
        by_time = ", ".join(f"{k}: {v}" for k, v in sorted(e["by_time_of_day"].items()))
        sentences.append(
            f"{e['count']} at-rest heart-rate excursions were detected on {e['days_with_episodes']} "
            f"day(s), about {e['per_week']} per week; timing {by_time}; {e['asleep']} while asleep."
        )
        sentences.append(
            f"Median duration {_fmt(e['duration_min_median'], ' min', 1)} (longest {_fmt(e['duration_min_max'], ' min', 1)}); "
            f"median peak {_fmt(e['peak_hr_median'], ' bpm')} (highest {_fmt(e['peak_hr_max'], ' bpm')}) "
            f"against a mean baseline of {_fmt(e['baseline_hr_mean'], ' bpm')}."
        )
        sentences.append(
            f"The person confirmed feeling {e['felt_yes']} of them, did not notice {e['felt_no']}, "
            f"and {e['felt_unknown']} were not answered."
        )
    else:
        sentences.append("No at-rest heart-rate excursions were detected in this period.")
    sentences.append(
        f"The person reported {s['count']} palpitation(s) themselves, {s['without_detected_episode']} "
        "of which had no matching excursion in the watch data."
    )
    sentences.append(
        f"Average resting heart rate {_fmt(avg.get('resting_hr'), ' bpm')}, average sleep "
        f"{_fmt_hours(avg.get('sleep_h'))}, average steps {_fmt(avg.get('steps'))} per day."
    )
    sentences.append("This is a symptom diary to support the consultation, not a diagnosis.")
    return " ".join(sentences)
