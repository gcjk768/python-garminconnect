"""Telegram message rendering.

Every function here returns a string for Telegram's **HTML parse mode**.  Only
``<b>``, ``<i>``, ``<code>``, ``<pre>`` and ``<a href>`` are used, and every
dynamic value (names, notes, model output, Garmin strings) goes through
:func:`esc` so a stray ``<`` in a note can never break a message.

Design rules (see ``docs/MESSAGES.md`` for the catalogue with examples):

* Data is UTC internally; times are rendered in the profile's timezone with
  :func:`~garmin_health_monitor.utils.to_local` / :func:`~garmin_health_monitor.utils.fmt_hm`.
* Any Garmin value may be missing.  Missing values render as ``n/a``; a message
  never raises because a field is ``None``.
* Where a 7-day history is available, values are shown with their delta versus
  the 7-day average, e.g. ``RHR 62 bpm (7-day avg 58, ▲4)``.
* Messages stay under :data:`MAX_LEN` characters (Telegram's limit is 4096);
  long lists are cut with ``… and N more``.
* The palpitation feature is a **symptom diary for a doctor**, never a
  diagnosis, and the wording stays plain and calm.
"""

from __future__ import annotations

import html
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

from .config import ProfileConfig
from .models import (
    ASSESSMENT_ARTIFACT,
    ASSESSMENT_EXERTION,
    ASSESSMENT_POSSIBLE,
    ASSESSMENT_UNCLEAR,
    EPISODE_KIND_NOCTURNAL,
    EPISODE_KIND_SPIKE,
    EPISODE_KIND_SUSTAINED,
    EPISODE_SOURCE_GARMIN_ALERT,
    EPISODE_SOURCE_MANUAL,
    Alert,
    CoachingAdvice,
    DaySnapshot,
    Episode,
    EpisodeAssessment,
    SymptomReport,
)
from .utils import clamp, fmt_duration, fmt_hm, mean, now_utc, to_local

logger = logging.getLogger(__name__)

MAX_LEN = 3800
"""Upper bound for a text message (Telegram allows 4096)."""

MAX_CAPTION_LEN = 1000
"""Upper bound for a photo/document caption (Telegram allows 1024)."""

NA = "n/a"
BAR_WIDTH = 10
NO_SYNC_WARN_HOURS = 3.0

ASSESSMENT_LABELS: dict[str, str] = {
    ASSESSMENT_POSSIBLE: "possible palpitation",
    ASSESSMENT_EXERTION: "likely exertion or movement",
    ASSESSMENT_ARTIFACT: "likely sensor artifact",
    ASSESSMENT_UNCLEAR: "unclear",
}

KIND_LABELS: dict[str, str] = {
    EPISODE_KIND_SUSTAINED: "sustained rise",
    EPISODE_KIND_SPIKE: "sudden spike",
    EPISODE_KIND_NOCTURNAL: "during sleep",
}

SOURCE_LABELS: dict[str, str] = {
    EPISODE_SOURCE_GARMIN_ALERT: "watch alert",
    EPISODE_SOURCE_MANUAL: "entered by hand",
}

SEVERITY_EMOJI: dict[str, str] = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}

NOT_DIAGNOSIS = (
    "<i>Wrist-sensor heart-rate readings for a symptom diary, not a diagnosis. "
    "Seek urgent care for chest pain, fainting or breathlessness.</i>"
)


# ---------------------------------------------------------------------------
# Escaping and small formatting helpers
# ---------------------------------------------------------------------------


def esc(text: Any) -> str:
    """HTML-escape a dynamic value for Telegram (``None`` renders as ``n/a``)."""
    if text is None:
        return NA
    return html.escape(str(text), quote=False)


def _n(value: Any, digits: int = 0, unit: str = "", sep: bool = False) -> str:
    """Format a number (``None`` -> ``n/a``); ``sep`` adds thousands separators."""
    if value is None:
        return NA
    try:
        v = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if digits == 0:
        text = f"{int(round(v)):,}" if sep else f"{int(round(v))}"
    else:
        text = f"{v:.{digits}f}"
    return f"{text}{unit}"


def _delta(
    value: Any,
    ref: Any,
    digits: int = 0,
    label: str = "7-day avg",
    unit: str = "",
    sep: bool = False,
) -> str:
    """`` (7-day avg 58, ▲4)`` or ``""`` when either side is missing."""
    if value is None or ref is None:
        return ""
    try:
        v, r = float(value), float(ref)
    except (TypeError, ValueError):
        return ""
    v, r = round(v, digits), round(r, digits)
    d = v - r
    tiny = 0.5 if digits == 0 else (10**-digits) / 2
    if abs(d) < tiny:
        arrow, mag = "±", 0.0
    else:
        arrow, mag = ("▲" if d > 0 else "▼"), abs(d)
    return f" ({label} {_n(r, digits, unit, sep)}, {arrow}{_n(mag, digits, unit, sep)})"


def _arrow(value: Any, ref: Any, digits: int = 0, unit: str = "", sep: bool = False) -> str:
    """``▲711`` / ``▼2`` / ``±0`` for value versus ref (``""`` when either is missing)."""
    if value is None or ref is None:
        return ""
    try:
        d = round(float(value), digits) - round(float(ref), digits)
    except (TypeError, ValueError):
        return ""
    tiny = 0.5 if digits == 0 else (10**-digits) / 2
    if abs(d) < tiny:
        return f"±{_n(0, digits, unit, sep)}"
    return ("▲" if d > 0 else "▼") + _n(abs(d), digits, unit, sep)


def _pct(value: Any, goal: Any) -> str:
    if value is None or not goal:
        return ""
    try:
        return f" ({int(round(100.0 * float(value) / float(goal)))}%)"
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


def _bar(value: Any, goal: Any, width: int = BAR_WIDTH) -> str:
    """Progress bar made of ▰▱ characters."""
    if value is None or not goal:
        return "▱" * width
    try:
        frac = clamp(float(value) / float(goal), 0.0, 1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return "▱" * width
    filled = int(round(frac * width))
    return "▰" * filled + "▱" * (width - filled)


def _hours(seconds: Any) -> str:
    """Seconds -> ``7.2h`` (``n/a`` when missing)."""
    if seconds is None:
        return NA
    try:
        return f"{float(seconds) / 3600.0:.1f}h"
    except (TypeError, ValueError):
        return NA


def _title(text: Any) -> str:
    """``HRV_BALANCED`` -> ``Hrv balanced``; ``None`` -> ``n/a``."""
    if text is None or str(text).strip() == "":
        return NA
    return esc(str(text).replace("_", " ").strip().capitalize())


def _tz(profile: ProfileConfig | None, snap: DaySnapshot | None = None) -> str:
    tz = getattr(profile, "timezone", None) or (snap.tz if snap is not None else None)
    return tz or "UTC"


def _name(profile: ProfileConfig | None) -> str:
    return esc(getattr(profile, "name", None) or "profile")


def _hm(dt: datetime | None, tz: str) -> str:
    return NA if dt is None else fmt_hm(dt, tz)


def _day_label(value: date | datetime | str | None, tz: str | None = None) -> str:
    """``Sun 27 Sep`` for a date / aware datetime / ISO string."""
    if value is None:
        return NA
    if isinstance(value, datetime):
        value = to_local(value, tz or "UTC").date()
    elif isinstance(value, str):
        try:
            value = date.fromisoformat(value[:10])
        except ValueError:
            return esc(value)
    return value.strftime("%a %d %b")


def _short_day(value: date | datetime | str | None, tz: str | None = None) -> str:
    """``Sun 27`` (used in the weekly table)."""
    if value is None:
        return "?"
    if isinstance(value, datetime):
        value = to_local(value, tz or "UTC").date()
    elif isinstance(value, str):
        try:
            value = date.fromisoformat(value[:10])
        except ValueError:
            return value[:6]
    return value.strftime("%a %d")


def _time_range(start: datetime | None, end: datetime | None, tz: str) -> str:
    if start is None:
        return NA
    s = to_local(start, tz)
    if end is None:
        return s.strftime("%H:%M")
    e = to_local(end, tz)
    suffix = " (+1d)" if e.date() != s.date() else ""
    return f"{s:%H:%M}–{e:%H:%M}{suffix}"


def _minutes(value: Any) -> str:
    if value is None:
        return NA
    try:
        return fmt_duration(float(value) * 60.0)
    except (TypeError, ValueError):
        return NA


def _row_get(row: Any, key: str) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _row_avg(rows: Iterable[Any] | None, key: str, exclude_day: str | None = None) -> float | None:
    vals: list[float] = []
    for r in rows or []:
        if exclude_day and str(_row_get(r, "day") or "")[:10] == exclude_day:
            continue
        v = _row_get(r, key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return mean(vals)


def _row_sum(rows: Iterable[Any] | None, key: str) -> float | None:
    vals: list[float] = []
    for r in rows or []:
        v = _row_get(r, key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return sum(vals) if vals else None


def _limited(items: Sequence[str], max_items: int) -> list[str]:
    """First ``max_items`` lines plus ``… and N more``."""
    items = list(items)
    if len(items) <= max_items:
        return items
    return [*items[:max_items], f"… and {len(items) - max_items} more"]


def _compact(parts: Sequence[str | None]) -> list[str | None]:
    """Hide what was not measured: drop ``· x n/a`` segments, then lines left with no number."""
    out: list[str | None] = []
    for line in parts:
        if line and NA in line:
            line = " · ".join(seg for seg in line.split(" · ") if NA not in seg)
            if not any(ch.isdigit() for ch in re.sub(r"<[^>]+>", "", line)):
                continue
        if line == "" and (not out or out[-1] == ""):
            continue  # no double blank lines
        out.append(line)
    return out


def _assemble(parts: Sequence[str | None], limit: int = MAX_LEN) -> str:
    """Join non-``None`` parts with newlines and keep the result under ``limit``.

    Parts are dropped from the end (whole lines, so HTML tags stay balanced)
    until the message fits; a marker says it was shortened.
    """
    lines = [p for p in parts if p is not None]
    while lines and lines[-1] == "":
        lines.pop()
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    marker = "… (message shortened)"
    while len(lines) > 1 and len("\n".join(lines)) + len(marker) + 1 > limit:
        lines.pop()
    lines.append(marker)
    text = "\n".join(lines)
    if len(text) > limit:  # a single oversized part: hard cut as a last resort
        logger.warning("Message part longer than %d characters; hard-truncating", limit)
        text = text[: limit - len(marker) - 1].rstrip() + "\n" + marker
    return text


def _nearest(samples: Iterable[Any], when: datetime, tolerance_min: float) -> Any | None:
    best = None
    best_diff = None
    for s in samples or []:
        ts = getattr(s, "ts", None)
        if ts is None:
            continue
        diff = abs((ts - when).total_seconds())
        if diff <= tolerance_min * 60 and (best_diff is None or diff < best_diff):
            best, best_diff = s, diff
    return best


def _latest(samples: Iterable[Any]) -> Any | None:
    latest = None
    for s in samples or []:
        if getattr(s, "ts", None) is None:
            continue
        if latest is None or s.ts > latest.ts:
            latest = s
    return latest


def _steps_today(snap: DaySnapshot) -> int | None:
    if snap.summary.total_steps is not None:
        return snap.summary.total_steps
    if snap.steps:
        return sum(int(b.steps or 0) for b in snap.steps)
    return None


def _last_sync(snap: DaySnapshot) -> datetime | None:
    if snap.summary.last_sync is not None:
        return snap.summary.last_sync
    if snap.device is not None and snap.device.last_upload is not None:
        return snap.device.last_upload
    last_hr = _latest(snap.hr)
    return last_hr.ts if last_hr is not None else None


def confidence_label(value: float | None) -> str:
    """Heuristic confidence 0..1 -> ``Low`` / ``Medium`` / ``High``."""
    if value is None:
        return NA
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NA
    if v < 0.4:
        return "Low"
    if v < 0.7:
        return "Medium"
    return "High"


def _felt_mark(ep: Episode) -> str:
    if ep.felt is True:
        return "✅ felt"
    if ep.felt is False:
        return "❌ not noticed"
    return "❓ not answered"


def _assessment_label(code: str | None) -> str:
    if not code:
        return "not assessed yet"
    return ASSESSMENT_LABELS.get(code, esc(str(code).replace("_", " ")))


def _activity_line(act: Any) -> str:
    name = _row_get(act, "name") or _row_get(act, "type_key") or _row_get(act, "type") or "Activity"
    duration = _row_get(act, "duration_s")
    avg_hr = _row_get(act, "avg_hr")
    bits = [esc(name)]
    if duration is not None:
        bits.append(fmt_duration(duration))
    bits.append(f"avg HR {_n(avg_hr, unit=' bpm')}")
    return "• " + " · ".join(bits)


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------


def _bullets(title: str, items: Any, max_items: int = 5, more: bool = True) -> list[str]:
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    try:
        clean = [str(i).strip() for i in items if i is not None and str(i).strip()]
    except TypeError:
        return []
    if not clean:
        return []
    shown = [f"• {esc(i)}" for i in clean]
    return [f"<b>{title}</b>", *(_limited(shown, max_items) if more else shown[:max_items])]


def coaching_block(advice: CoachingAdvice | None) -> str:
    """Short do-more / do-less card from the model (``""`` when absent); phone-sized."""
    if advice is None:
        return ""
    period = "this week" if getattr(advice, "period", "daily") == "weekly" else "today"
    lines: list[str] = [f"🧭 <b>Coaching for {period}</b>"]
    summary = getattr(advice, "summary", None)
    if summary:
        lines.append(f"<i>{esc(summary)}</i>")
    lines.extend(_bullets("✅ Do more", getattr(advice, "do_more", None), max_items=2, more=False))
    lines.extend(_bullets("⛔ Do less", getattr(advice, "do_less", None), max_items=2, more=False))
    lines.extend(_bullets("👀 Watch out", getattr(advice, "watch_outs", None), max_items=1, more=False))
    heart = getattr(advice, "heart_note", None)
    if heart:
        lines.append(f"❤️ {esc(heart)}")
    lines.append("<i>AI tips, not medical advice.</i>")
    return "\n".join(lines)


def _episode_line(ep: Episode, tz: str, with_date: bool, with_assessment: bool) -> str:
    when = _time_range(ep.start, ep.end, tz)
    if with_date:
        when = f"{_day_label(ep.start, tz)} {when}"
    bits = [
        when,
        _minutes(ep.duration_min),
        f"peak {_n(ep.peak_hr, unit=' bpm')}",
        _felt_mark(ep),
    ]
    if ep.asleep:
        bits.append("asleep")
    src = SOURCE_LABELS.get(ep.source)
    if src:
        bits.append(src)
    if with_assessment:
        bits.append(_assessment_label(ep.llm_assessment))
    return "• " + " · ".join(bits)


def _symptom_line(rep: SymptomReport, tz: str, with_date: bool = True) -> str:
    when = _hm(rep.event_time, tz)
    if with_date:
        when = f"{_day_label(rep.event_time, tz)} {when}"
    parts = [when]
    if rep.note:
        parts.append(f"“{esc(rep.note)}”")
    if rep.hr_at_time is not None:
        parts.append(f"HR {_n(rep.hr_at_time, unit=' bpm')}")
    if rep.episode_id is not None:
        parts.append(f"linked to episode #{_n(rep.episode_id)}")
    return "• " + " — ".join(parts[:2]) + ("" if len(parts) <= 2 else " · " + " · ".join(parts[2:]))


def _sleep_lines(snap: DaySnapshot) -> list[str]:
    """Short sleep block used by the morning brief and evening summary."""
    sl = snap.sleep
    if sl is None:
        return ["No sleep data yet (watch not synced or not worn overnight)."]
    tz = snap.tz or "UTC"
    score = _n(sl.score)
    if sl.score_qualifier:
        score += f" ({_title(sl.score_qualifier)})"
    lines = [
        f"Slept {fmt_duration(sl.total_seconds)} ({_hm(sl.start, tz)}–{_hm(sl.end, tz)}) · score {score}"
    ]
    lines.append(
        "Deep "
        + fmt_duration(sl.deep_seconds)
        + " · Light "
        + fmt_duration(sl.light_seconds)
        + " · REM "
        + fmt_duration(sl.rem_seconds)
        + " · Awake "
        + fmt_duration(sl.awake_seconds)
    )
    return lines


def _hrv_line(snap: DaySnapshot) -> str:
    hrv = snap.hrv
    last = hrv.last_night_avg if hrv else None
    weekly = hrv.weekly_avg if hrv else None
    status = hrv.status if hrv else None
    if last is None and snap.sleep is not None:
        last = snap.sleep.avg_hrv
        status = status or snap.sleep.hrv_status
    if weekly is None and snap.readiness is not None:
        weekly = snap.readiness.hrv_weekly_avg
    text = f"💓 HRV overnight: {_n(last, unit=' ms')}{_delta(last, weekly, label='weekly avg', unit=' ms')}"
    if status:
        text += f" — {_title(status)}"
    return text


def _rhr_line(snap: DaySnapshot, fallback_row: Any = None, avg_7d: float | None = None) -> str:
    s = snap.summary
    rhr = s.resting_hr
    suffix = ""
    if rhr is None and fallback_row is not None:
        rhr = _row_get(fallback_row, "resting_hr")
        suffix = " (yesterday)" if rhr is not None else ""
    ref = avg_7d if avg_7d is not None else s.resting_hr_7d_avg
    if ref is None and fallback_row is not None:
        ref = _row_get(fallback_row, "resting_hr_7d_avg")
    return f"❤️ Resting HR: {_n(rhr, unit=' bpm')}{suffix}{_delta(rhr, ref)}"


# ---------------------------------------------------------------------------
# Scheduled messages
# ---------------------------------------------------------------------------


_BALANCE_TEXT = {
    "AEROBIC_LOW_SHORTAGE": "needs more easy aerobic",
    "AEROBIC_HIGH_SHORTAGE": "needs more tempo work",
    "ANAEROBIC_SHORTAGE": "needs more hard intervals",
    "BALANCED": "balanced",
}


def _first(d: Any) -> dict:
    """First value of a ``{deviceId: {...}}`` map (Garmin keys per-device data by device id)."""
    return next(iter(d.values()), {}) if isinstance(d, dict) and d else {}


def _race_time(seconds: Any) -> str | None:
    if not isinstance(seconds, int | float) or seconds <= 0:
        return None
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fitness_lines(fit: dict[str, Any] | None) -> list[str]:
    """💪 VO2 max, fitness age, training status, weekly intensity minutes, race predictions."""
    fit = fit or {}
    ts = fit.get("training_status") or {}
    lines: list[str] = []

    vo2 = ((ts.get("mostRecentVO2Max") or {}).get("generic") or {})
    vo2_val = vo2.get("vo2MaxPreciseValue") or vo2.get("vo2MaxValue")
    age = fit.get("fitness_age") or {}
    parts = []
    if vo2_val:
        parts.append(f"VO2 max {_n(vo2_val, 1)}")
    if age.get("fitnessAge"):
        txt = f"fitness age {_n(age['fitnessAge'], 1)}"
        if age.get("chronologicalAge"):
            txt += f" (real {_n(age['chronologicalAge'])})"
        parts.append(txt)
    if parts:
        lines.append("🫀 " + " · ".join(parts))

    status = _first((ts.get("mostRecentTrainingStatus") or {}).get("latestTrainingStatusData"))
    phrase = re.sub(r"_\d+$", "", str(status.get("trainingStatusFeedbackPhrase") or ""))
    balance = _first((ts.get("mostRecentTrainingLoadBalance") or {}).get("metricsTrainingLoadBalanceDTOMap"))
    bal = str(balance.get("trainingBalanceFeedbackPhrase") or "")
    if phrase or bal:
        txt = f"📈 Training status: {esc(_title(phrase))}" if phrase else "📈 Training"
        if bal:
            txt += f" · load {esc(_BALANCE_TEXT.get(bal, _title(bal)))}"
        lines.append(txt)

    weeks = [w for w in fit.get("intensity") or [] if isinstance(w, dict)]
    if weeks:
        def total(w: dict) -> int:  # Garmin counts vigorous minutes double
            return int(w.get("moderateValue") or 0) + 2 * int(w.get("vigorousValue") or 0)

        goal = weeks[-1].get("weeklyGoal")
        txt = f"⏱️ Intensity minutes this week {total(weeks[-1])}" + (f"/{goal}" if goal else "")
        if len(weeks) > 1:
            txt += f" · last week {total(weeks[-2])}"
        lines.append(txt)

    rp = fit.get("race_predictions") or {}
    races = [(lbl, _race_time(rp.get(k))) for lbl, k in
             (("5K", "time5K"), ("10K", "time10K"), ("Half", "timeHalfMarathon"), ("Full", "timeMarathon"))]
    races = [f"{lbl} {t}" for lbl, t in races if t]
    if races:
        lines.append("🏁 Race predictions: " + " · ".join(races))

    return ["💪 <b>Fitness</b>", *lines] if lines else []


def morning_brief(
    profile: ProfileConfig,
    today: DaySnapshot,
    yesterday_row: dict[str, Any] | None,
    overnight_episodes: Sequence[Episode] | None,
    coaching: CoachingAdvice | None,
) -> str:
    """🌅 Morning brief: last night's sleep, HRV, resting HR, readiness, overnight episodes."""
    tz = _tz(profile, today)
    s = today.summary
    parts: list[str | None] = [
        f"🌅 <b>Good morning, {_name(profile)}</b> — {_day_label(today.day)}",
        "",
        "😴 <b>Last night</b>",
        *_sleep_lines(today),
        _hrv_line(today),
        _rhr_line(today, yesterday_row),
    ]

    # Body battery on waking: nearest sample to the sleep end, else today's highest.
    bb_text = NA
    if today.sleep is not None and today.sleep.end is not None:
        near = _nearest(today.body_battery, today.sleep.end, tolerance_min=45)
        if near is not None:
            bb_text = _n(near.level)
    if bb_text == NA and s.body_battery_high is not None:
        bb_text = f"{_n(s.body_battery_high)} (highest so far)"
    parts.append(f"🔋 Body battery on waking: {bb_text}")

    rd = today.readiness
    if rd is not None and (rd.score is not None or rd.level):
        ready = _n(rd.score)
        if rd.level:
            ready += f" ({_title(rd.level)})"
        parts.append(f"🎯 Training readiness: {ready}")
    else:
        parts.append(f"🎯 Training readiness: {NA}")

    spo2_low = None
    if today.spo2 is not None and today.spo2.lowest is not None:
        spo2_low = today.spo2.lowest
    elif today.sleep is not None and today.sleep.lowest_spo2 is not None:
        spo2_low = today.sleep.lowest_spo2
    elif s.lowest_spo2 is not None:
        spo2_low = s.lowest_spo2
    parts.append(f"🫁 Lowest SpO2 overnight: {_n(spo2_low, unit='%')}")

    if yesterday_row:
        parts.append(
            "📅 Yesterday: "
            f"{_n(_row_get(yesterday_row, 'total_steps'), sep=True)} steps · "
            f"RHR {_n(_row_get(yesterday_row, 'resting_hr'))} · "
            f"stress {_n(_row_get(yesterday_row, 'avg_stress'))}"
        )

    palp_on = bool(getattr(getattr(profile, "features", None), "palpitations", False))
    episodes = list(overnight_episodes or [])
    if episodes:
        parts.append("")
        parts.append(f"❤️ <b>Overnight / early-morning heart-rate excursions: {len(episodes)}</b>")
        parts.extend(_limited([_episode_line(e, tz, False, True) for e in episodes], 6))
        parts.append("Use /episodes for details, or /palp HH:MM if you felt something.")
    elif palp_on:
        parts.append("❤️ No at-rest heart-rate excursions overnight.")

    if coaching is not None:
        parts.append("")
        parts.append(coaching_block(coaching))
    return _assemble(_compact(parts))


def evening_summary(
    profile: ProfileConfig,
    today: DaySnapshot,
    rows_7d: Sequence[dict[str, Any]] | None,
    episodes_today: Sequence[Episode] | None,
    symptoms_today: Sequence[SymptomReport] | None,
    coaching: CoachingAdvice | None,
    fitness: dict[str, Any] | None = None,
) -> str:
    """🌙 Evening summary: steps, activity, stress, body battery, sleep, episodes, coaching."""
    tz = _tz(profile, today)
    s = today.summary
    day_str = today.date_str
    steps = _steps_today(today)
    goal = s.step_goal
    parts: list[str | None] = [
        f"🌙 <b>Evening summary — {_name(profile)}</b> — {_day_label(today.day)}",
        "",
        f"👟 <b>Steps</b> {_bar(steps, goal)} {_n(steps, sep=True)} / {_n(goal, sep=True)}{_pct(steps, goal)}",
    ]
    avg_steps = _row_avg(rows_7d, "total_steps", exclude_day=day_str)
    if avg_steps is not None:
        parts.append(f"7-day avg {_n(avg_steps, sep=True)} · today {_arrow(steps, avg_steps, sep=True)}")
    km = None if s.distance_m is None else s.distance_m / 1000.0
    parts.append(
        f"Distance {_n(km, 1, ' km')} · Floors {_n(s.floors_up)} · "
        f"Calories {_n(s.total_kcal, sep=True)} kcal (active {_n(s.active_kcal, sep=True)})"
    )
    active_s = None
    if s.active_seconds is not None or s.highly_active_seconds is not None:
        active_s = (s.active_seconds or 0) + (s.highly_active_seconds or 0)
    intensity = None
    if s.moderate_intensity_min is not None or s.vigorous_intensity_min is not None:
        intensity = (s.moderate_intensity_min or 0) + 2 * (s.vigorous_intensity_min or 0)
    parts.append(
        f"Active time {fmt_duration(active_s)} · Intensity minutes {_n(intensity)}"
        + (f" (weekly goal {_n(s.intensity_goal_min)})" if s.intensity_goal_min else "")
    )

    parts.append("")
    high_stress = None if s.high_stress_seconds is None else s.high_stress_seconds / 60.0
    stress_txt = f"🧘 Stress avg {_n(s.avg_stress)}{_delta(s.avg_stress, _row_avg(rows_7d, 'avg_stress', day_str))}"
    stress_txt += f" · high stress {_minutes(high_stress)}"
    if s.stress_qualifier:
        stress_txt += f" · {_title(s.stress_qualifier)}"
    parts.append(stress_txt)
    latest_bb = s.body_battery_latest
    if latest_bb is None:
        last = _latest(today.body_battery)
        latest_bb = last.level if last is not None else None
    parts.append(
        f"🔋 Body battery high {_n(s.body_battery_high)}"
        f"{_delta(s.body_battery_high, _row_avg(rows_7d, 'body_battery_high', day_str))}"
        f" / low {_n(s.body_battery_low)} · now {_n(latest_bb)}"
    )
    parts.append(_rhr_line(today, None, _row_avg(rows_7d, "resting_hr", day_str)))
    sleep_s = today.sleep.total_seconds if today.sleep is not None else None
    sleep_txt = f"😴 Sleep last night {fmt_duration(sleep_s)}"
    if today.sleep is not None and today.sleep.score is not None:
        sleep_txt += f" · score {_n(today.sleep.score)}"
    avg_sleep = _row_avg(rows_7d, "sleep_seconds", day_str)
    if sleep_s is not None and avg_sleep is not None:
        sleep_txt += _delta(sleep_s / 3600.0, avg_sleep / 3600.0, 1, unit="h")
    parts.append(sleep_txt)

    parts.append("")
    acts = list(today.activities or [])
    if acts:
        parts.append(f"🏃 <b>Activities ({len(acts)})</b>")
        parts.extend(_limited([_activity_line(a) for a in acts], 6))
    else:
        parts.append("🏃 No recorded activities today.")

    palp_on = bool(getattr(getattr(profile, "features", None), "palpitations", False))
    if palp_on:
        parts.append("")
        eps = list(episodes_today or [])
        if eps:
            parts.append(f"❤️ <b>At-rest heart-rate excursions today: {len(eps)}</b>")
            parts.extend(_limited([_episode_line(e, tz, False, True) for e in eps], 8))
        else:
            parts.append("❤️ No at-rest heart-rate excursions detected today.")
        reps = list(symptoms_today or [])
        if reps:
            parts.append(f"📝 <b>Reported symptoms today: {len(reps)}</b>")
            parts.extend(_limited([_symptom_line(r, tz, with_date=False) for r in reps], 6))
        else:
            parts.append("📝 No symptoms reported today (use /palp if you felt something).")

    fit = fitness_lines(fitness)
    if fit:
        parts.append("")
        parts.extend(fit)
    if coaching is not None:
        parts.append("")
        parts.append(coaching_block(coaching))
    return _assemble(_compact(parts))


def _week_table(rows: Sequence[Any]) -> str:
    header = f"{'Day':<6} {'Steps':>6} {'Sleep':>5} {'RHR':>4} {'Str':>4} {'BB':>4}"
    lines = [header]
    for r in rows:
        lines.append(
            f"{_short_day(_row_get(r, 'day')):<6} "
            f"{_n(_row_get(r, 'total_steps')):>6} "
            f"{_hours(_row_get(r, 'sleep_seconds')):>5} "
            f"{_n(_row_get(r, 'resting_hr')):>4} "
            f"{_n(_row_get(r, 'avg_stress')):>4} "
            f"{_n(_row_get(r, 'body_battery_high')):>4}"
        )
    return "<pre>" + esc("\n".join(lines)) + "</pre>"


def _week_label(rows: Sequence[Any]) -> str:
    days = sorted(str(_row_get(r, "day") or "")[:10] for r in rows if _row_get(r, "day"))
    if not days:
        return "this week"
    if len(days) == 1:
        return _day_label(days[0])
    return f"{_day_label(days[0])} – {_day_label(days[-1])}"


def _activities_count(rows: Sequence[Any]) -> int | None:
    total = 0
    seen = False
    for r in rows:
        acts = _row_get(r, "activities")
        cnt = _row_get(r, "activities_count")
        if isinstance(acts, list):
            total += len(acts)
            seen = True
        elif cnt is not None:
            try:
                total += int(cnt)
                seen = True
            except (TypeError, ValueError):
                pass
    return total if seen else None


def weekly_review(
    profile: ProfileConfig,
    rows_this_week: Sequence[dict[str, Any]] | None,
    rows_prev_week: Sequence[dict[str, Any]] | None,
    episodes: Sequence[Episode] | None,
    coaching: CoachingAdvice | None,
) -> str:
    """📊 Weekly review: 7-day table, totals/averages vs the previous week, episodes, coaching."""
    tz = _tz(profile)
    cur = list(rows_this_week or [])
    prev = list(rows_prev_week or [])
    parts: list[str | None] = [
        f"📊 <b>Weekly review — {_name(profile)}</b> — {_week_label(cur)}",
        "",
    ]
    if cur:
        parts.append(_week_table(cur[-7:]))
    else:
        parts.append("No daily data stored for this week yet.")
    parts.append("")
    parts.append("<b>Versus the previous week</b>")
    steps_total = _row_sum(cur, "total_steps")
    parts.append(
        f"👟 Steps total {_n(steps_total, sep=True)}"
        f"{_delta(steps_total, _row_sum(prev, 'total_steps'), label='prev', sep=True)}"
        f" · avg {_n(_row_avg(cur, 'total_steps'), sep=True)}/day"
    )
    cur_sleep = _row_avg(cur, "sleep_seconds")
    prev_sleep = _row_avg(prev, "sleep_seconds")
    parts.append(
        f"😴 Sleep avg {_hours(cur_sleep)}"
        + _delta(
            None if cur_sleep is None else cur_sleep / 3600.0,
            None if prev_sleep is None else prev_sleep / 3600.0,
            1,
            label="prev",
            unit="h",
        )
    )
    cur_rhr = _row_avg(cur, "resting_hr")
    parts.append(f"❤️ Resting HR avg {_n(cur_rhr, unit=' bpm')}{_delta(cur_rhr, _row_avg(prev, 'resting_hr'), label='prev')}")
    cur_stress = _row_avg(cur, "avg_stress")
    parts.append(f"🧘 Stress avg {_n(cur_stress)}{_delta(cur_stress, _row_avg(prev, 'avg_stress'), label='prev')}")
    cur_bb = _row_avg(cur, "body_battery_high")
    parts.append(f"🔋 Body battery high avg {_n(cur_bb)}{_delta(cur_bb, _row_avg(prev, 'body_battery_high'), label='prev')}")
    cur_hrv = _row_avg(cur, "hrv_last_night")
    if cur_hrv is not None:
        parts.append(f"💓 HRV avg {_n(cur_hrv, unit=' ms')}{_delta(cur_hrv, _row_avg(prev, 'hrv_last_night'), label='prev', unit=' ms')}")
    n_act = _activities_count(cur)
    if n_act is not None:
        parts.append(f"🏃 Activities {n_act}{_delta(n_act, _activities_count(prev), label='prev')}")

    palp_on = bool(getattr(getattr(profile, "features", None), "palpitations", False))
    eps = list(episodes or [])
    if eps or palp_on:
        parts.append("")
        if eps:
            per_day = Counter(to_local(e.start, tz).date() for e in eps if e.start is not None)
            days_txt = " · ".join(f"{_short_day(d)} ×{n}" for d, n in sorted(per_day.items()))
            felt = sum(1 for e in eps if e.felt is True)
            parts.append(f"❤️ <b>At-rest heart-rate excursions this week: {len(eps)}</b> (felt {felt})")
            parts.append(esc(days_txt) if days_txt else None)
            parts.append("See /episodes 7 for the list, /report 30 for the doctor PDF.")
        else:
            parts.append("❤️ No at-rest heart-rate excursions this week.")

    if coaching is not None:
        parts.append("")
        parts.append(coaching_block(coaching))
    return _assemble(_compact(parts))


# ---------------------------------------------------------------------------
# Interactive commands
# ---------------------------------------------------------------------------


def today_status(
    profile: ProfileConfig,
    snap: DaySnapshot,
    episodes_today: Sequence[Episode] | None,
    now: datetime | None = None,
) -> str:
    """📍 Live snapshot for ``/today``: steps, body battery, last HR, sync state."""
    tz = _tz(profile, snap)
    now = now or now_utc()
    s = snap.summary
    steps = _steps_today(snap)
    parts: list[str | None] = [
        f"📍 <b>Right now — {_name(profile)}</b> — {_day_label(now, tz)} {fmt_hm(now, tz)}",
        "",
        f"👟 Steps so far: {_bar(steps, s.step_goal)} {_n(steps, sep=True)} / {_n(s.step_goal, sep=True)}{_pct(steps, s.step_goal)}",
    ]
    last_bb = _latest(snap.body_battery)
    if last_bb is not None:
        parts.append(f"🔋 Body battery: {_n(last_bb.level)} (at {fmt_hm(last_bb.ts, tz)})")
    else:
        parts.append(f"🔋 Body battery: {_n(s.body_battery_latest)}")
    last_hr = _latest(snap.hr)
    if last_hr is not None:
        parts.append(f"❤️ Last HR: {_n(last_hr.hr, unit=' bpm')} at {fmt_hm(last_hr.ts, tz)} · resting {_n(s.resting_hr)}{_delta(s.resting_hr, s.resting_hr_7d_avg)}")
    else:
        parts.append(f"❤️ Last HR: {NA} · resting {_n(s.resting_hr)}{_delta(s.resting_hr, s.resting_hr_7d_avg)}")
    parts.append(f"🧘 Stress avg: {_n(s.avg_stress)}")
    acts = list(snap.activities or [])
    if acts:
        parts.append(f"🏃 Activities so far ({len(acts)}):")
        parts.extend(_limited([_activity_line(a) for a in acts], 4))
    else:
        parts.append("🏃 No recorded activities so far.")
    eps = list(episodes_today or [])
    palp_on = bool(getattr(getattr(profile, "features", None), "palpitations", False))
    if eps:
        parts.append(f"❤️ At-rest heart-rate excursions today: {len(eps)}")
        parts.extend(_limited([_episode_line(e, tz, False, False) for e in eps], 4))
    elif palp_on:
        parts.append("❤️ At-rest heart-rate excursions today: none")

    parts.append("")
    sync = _last_sync(snap)
    device = snap.device.name if snap.device is not None and snap.device.name else None
    sync_txt = f"📡 Last sync: {_hm(sync, tz)}"
    if sync is not None and to_local(sync, tz).date() != to_local(now, tz).date():
        sync_txt = f"📡 Last sync: {_day_label(sync, tz)} {_hm(sync, tz)}"
    if device:
        sync_txt += f" ({esc(device)})"
    parts.append(sync_txt)
    if sync is None:
        parts.append("⚠️ No sync time reported — the watch may not be connected to the phone.")
    else:
        age_h = (now - sync).total_seconds() / 3600.0
        if age_h > NO_SYNC_WARN_HOURS:
            parts.append(
                f"⚠️ Watch has not synced for {fmt_duration(age_h * 3600)} — "
                "check it is worn and Garmin Connect is open on the phone."
            )
    return _assemble(_compact(parts))


def sleep_message(profile: ProfileConfig, snap: DaySnapshot) -> str:
    """😴 Detailed sleep for ``/sleep``."""
    tz = _tz(profile, snap)
    sl = snap.sleep
    parts: list[str | None] = [
        f"😴 <b>Sleep — {_name(profile)}</b> — night to {_day_label(snap.day)}",
        "",
    ]
    if sl is None:
        parts.append("No sleep data for this night (watch not worn, or not synced yet).")
        return _assemble(parts)
    total = sl.total_seconds
    parts.append(f"Bed {_hm(sl.start, tz)} → wake {_hm(sl.end, tz)} · {fmt_duration(total)} asleep")
    score = _n(sl.score)
    if sl.score_qualifier:
        score += f" ({_title(sl.score_qualifier)})"
    parts.append(f"Score {score}")

    def stage(label: str, secs: int | None) -> str:
        txt = f"{label} {fmt_duration(secs)}"
        if secs is not None and total:
            txt += f" ({int(round(100.0 * secs / total))}%)"
        return txt

    parts.append(
        " · ".join(
            [
                stage("Deep", sl.deep_seconds),
                stage("Light", sl.light_seconds),
                stage("REM", sl.rem_seconds),
                f"Awake {fmt_duration(sl.awake_seconds)}",
            ]
        )
    )
    if sl.nap_seconds:
        parts.append(f"Nap {fmt_duration(sl.nap_seconds)}")
    parts.append(f"Woke {_n(sl.awake_count)} times · restless moments {_n(sl.restless_moments)}")
    hrv_txt = f"HRV {_n(sl.avg_hrv, unit=' ms')}"
    status = sl.hrv_status or (snap.hrv.status if snap.hrv is not None else None)
    if status:
        hrv_txt += f" ({_title(status)})"
    parts.append(f"❤️ Sleeping HR {_n(sl.resting_hr, unit=' bpm')} · {hrv_txt}")
    parts.append(
        f"🫁 SpO2 avg {_n(sl.avg_spo2, unit='%')} · lowest {_n(sl.lowest_spo2, unit='%')} · "
        f"respiration {_n(sl.avg_respiration, 1, ' brpm')}"
    )
    if sl.body_battery_change is not None:
        sign = "+" if sl.body_battery_change >= 0 else ""
        parts.append(f"🔋 Body battery {sign}{_n(sl.body_battery_change)} overnight")
    return _assemble(parts)


def hr_message(profile: ProfileConfig, snap: DaySnapshot, episodes: Sequence[Episode] | None) -> str:
    """❤️ Heart-rate summary for ``/hr`` (short enough to be a photo caption)."""
    tz = _tz(profile, snap)
    s = snap.summary
    last = _latest(snap.hr)
    parts: list[str | None] = [
        f"❤️ <b>Heart rate — {_name(profile)}</b> — {_day_label(snap.day)}",
        f"Resting {_n(s.resting_hr, unit=' bpm')}{_delta(s.resting_hr, s.resting_hr_7d_avg)}",
        f"Min {_n(s.min_hr)} · Max {_n(s.max_hr)} · {len(snap.hr)} readings"
        + (f" (last {fmt_hm(last.ts, tz)}: {_n(last.hr)} bpm)" if last is not None else ""),
    ]
    if s.abnormal_hr_alerts:
        parts.append(f"⚠️ Watch abnormal-HR alerts: {_n(s.abnormal_hr_alerts)}")
    eps = list(episodes or [])
    if eps:
        parts.append(f"At-rest excursions: {len(eps)}")
        for e in _limited(
            [
                f"• {_time_range(e.start, e.end, tz)} · {_minutes(e.duration_min)} · peak {_n(e.peak_hr)} "
                f"(base {_n(e.baseline_hr)}, +{_n(e.delta_hr)}) · {KIND_LABELS.get(e.kind, esc(e.kind))} · {_felt_mark(e)}"
                for e in eps
            ],
            5,
        ):
            parts.append(e)
    else:
        parts.append("At-rest excursions: none detected.")
    return _assemble(parts, limit=MAX_CAPTION_LEN)


def steps_message(profile: ProfileConfig, snap: DaySnapshot, rows_7d: Sequence[dict[str, Any]] | None) -> str:
    """👟 Steps today versus the goal and the 7-day average."""
    s = snap.summary
    steps = _steps_today(snap)
    goal = s.step_goal
    day_str = snap.date_str
    parts: list[str | None] = [
        f"👟 <b>Steps — {_name(profile)}</b> — {_day_label(snap.day)}",
        "",
        f"{_bar(steps, goal)} {_n(steps, sep=True)} / {_n(goal, sep=True)}{_pct(steps, goal)}",
    ]
    avg = _row_avg(rows_7d, "total_steps", exclude_day=day_str)
    if avg is not None:
        parts.append(f"7-day avg {_n(avg, sep=True)} · today {_arrow(steps, avg, sep=True)}")
    else:
        parts.append(f"7-day avg {NA}")
    if goal and steps is not None and steps < goal:
        parts.append(f"{_n(goal - steps, sep=True)} more to reach the goal.")
    elif goal and steps is not None:
        parts.append("Goal reached 🎉")
    km = None if s.distance_m is None else s.distance_m / 1000.0
    active_s = None
    if s.active_seconds is not None or s.highly_active_seconds is not None:
        active_s = (s.active_seconds or 0) + (s.highly_active_seconds or 0)
    parts.append(f"Distance {_n(km, 1, ' km')} · Floors {_n(s.floors_up)} · Active time {fmt_duration(active_s)}")
    rows = [r for r in (rows_7d or []) if _row_get(r, "day")]
    if rows:
        rows = sorted(rows, key=lambda r: str(_row_get(r, "day")))[-7:]
        met = 0
        table = []
        for r in rows:
            st = _row_get(r, "total_steps")
            g = _row_get(r, "step_goal") or goal
            if st is not None and g and float(st) >= float(g):
                met += 1
            table.append(f"{_short_day(_row_get(r, 'day')):<6} {_n(st, sep=True):>7} {_bar(st, g, 8)}")
        parts.append("")
        parts.append("<pre>" + esc("\n".join(table)) + "</pre>")
        parts.append(f"Goal met on {met} of the last {len(rows)} days.")
    return _assemble(parts)


# ---------------------------------------------------------------------------
# Palpitation diary
# ---------------------------------------------------------------------------


def episode_alert(profile: ProfileConfig, ep: Episode, assessment: EpisodeAssessment | None) -> str:
    """❤️ Short alert for one possible palpitation: date, time, heart rate (buttons added by the bot)."""
    tz = _tz(profile)
    code = assessment.assessment if assessment is not None else ep.llm_assessment
    parts: list[str | None] = [
        f"❤️ <b>Possible palpitation — {_name(profile)}</b>",
        f"📅 {_day_label(ep.start, tz)} · 🕒 {_time_range(ep.start, ep.end, tz)} ({_minutes(ep.duration_min)})",
        f"💓 Peak <b>{_n(ep.peak_hr, unit=' bpm')}</b> · before {_n(ep.baseline_hr, unit=' bpm')} "
        f"(+{_n(ep.delta_hr)}) · avg {_n(ep.mean_hr, unit=' bpm')}",
        f"🧭 {'Asleep' if ep.asleep else 'At rest'}, {_n(ep.steps_in_window)} steps",
    ]
    if code:
        parts.append(f"🤖 AI view: {_assessment_label(code)}")
    parts.append("<b>Was it felt?</b> Tap below.")
    parts.append(NOT_DIAGNOSIS)
    return _assemble(parts)


_NOT_PALPITATION = {"likely_exertion", "likely_artifact"}


def _heart_split(episodes: Sequence[Episode] | None) -> tuple[list[Episode], list[Episode]]:
    """(shown, hidden): hide only what the AI ruled out, so unassessed episodes still show."""
    eps = sorted(episodes or [], key=lambda e: e.start)
    return [e for e in eps if e.llm_assessment not in _NOT_PALPITATION], [
        e for e in eps if e.llm_assessment in _NOT_PALPITATION
    ]


def heart_table(episodes: Sequence[Episode], tz: str, with_date: bool = True) -> str:
    """Aligned monospace table: date, start time, peak bpm, minutes (+ 'asleep')."""
    head = ("Date    " if with_date else "") + "Time   Peak  Min"
    rows = [head]
    for e in episodes:
        s = to_local(e.start, tz)
        row = (f"{s:%d %b}  " if with_date else "") + f"{s:%H:%M}  {e.peak_hr:>4}  {round(e.duration_min):>3}"
        rows.append(row + ("  asleep" if e.asleep else ""))
    return "<pre>" + esc("\n".join(rows)) + "</pre>"


def heart_month(profile: ProfileConfig, title: str, episodes: Sequence[Episode] | None) -> str:
    """❤️ Monthly list of possible palpitations as one clean table."""
    tz = _tz(profile)
    shown, hidden = _heart_split(episodes)
    days = len({to_local(e.start, tz).date() for e in shown})
    parts: list[str | None] = [f"❤️ <b>{_name(profile)} · {esc(title)}</b>"]
    if shown:
        parts.append(f"<b>{len(shown)}</b> possible palpitations on <b>{days}</b> days")
        parts.append(heart_table(shown, tz))
        parts.append("Peak = highest bpm · Min = minutes")
    else:
        parts.append("✅ No possible palpitations.")
    if hidden:
        parts.append(f"<i>{len(hidden)} more looked like exercise (in Obsidian).</i>")
    return _assemble(parts)


def heart_review(
    profile: ProfileConfig,
    day: date,
    episodes: Sequence[Episode] | None,
    snap: DaySnapshot | None,
) -> str:
    """❤️ 22:00 heart review: every possible palpitation today in one table."""
    tz = _tz(profile, snap)
    shown, hidden = _heart_split(episodes)
    parts: list[str | None] = [f"❤️ <b>{_name(profile)} · {_day_label(day)}</b>"]
    if shown:
        parts.append(f"<b>{len(shown)}</b> possible palpitation{'s' if len(shown) != 1 else ''} today")
        parts.append(heart_table(shown, tz, with_date=False))
    else:
        parts.append("✅ No possible palpitations today.")
    if hidden:
        parts.append(f"<i>{len(hidden)} more looked like exercise.</i>")
    if snap is not None and snap.summary.resting_hr:
        parts.append(f"Resting HR today: {_n(snap.summary.resting_hr, unit=' bpm')}")
    return _assemble(parts)


def _frequency_summary(episodes: Sequence[Episode], days: int, tz: str) -> str | None:
    if not episodes:
        return None
    n = len(episodes)
    per_week = n * 7.0 / max(days, 1)
    hours = Counter(to_local(e.start, tz).hour for e in episodes if e.start is not None)
    hour_txt = NA
    if hours:
        hour, count = hours.most_common(1)[0]
        hour_txt = f"{hour:02d}:00–{(hour + 1) % 24:02d}:00 ({count} of {n})"
    asleep = sum(1 for e in episodes if e.asleep)
    felt = sum(1 for e in episodes if e.felt is True)
    not_noticed = sum(1 for e in episodes if e.felt is False)
    unanswered = n - felt - not_noticed
    return (
        f"📈 <b>Frequency:</b> about {per_week:.1f} per week · most common hour {hour_txt} · "
        f"{asleep} during sleep · felt {felt}, not noticed {not_noticed}, unanswered {unanswered}"
    )


def episodes_list(
    profile: ProfileConfig,
    episodes: Sequence[Episode] | None,
    symptoms: Sequence[SymptomReport] | None,
    days: int,
) -> str:
    """🩺 Palpitation diary for ``/episodes``: detected episodes, reported symptoms, frequency."""
    tz = _tz(profile)
    try:
        days = max(int(days), 1)
    except (TypeError, ValueError):
        days = 7
    eps = sorted(episodes or [], key=lambda e: e.start or now_utc(), reverse=True)
    reps = sorted(symptoms or [], key=lambda r: r.event_time or now_utc(), reverse=True)
    parts: list[str | None] = [
        f"🩺 <b>Palpitation diary — {_name(profile)}</b> — last {days} days",
        "",
    ]
    if not eps and not reps:
        parts.append(f"Nothing recorded in the last {days} days.")
        parts.append("Log anything you feel with <code>/palp HH:MM note</code> so it reaches the doctor report.")
        return _assemble(parts)
    if eps:
        parts.append(f"<b>Detected episodes: {len(eps)}</b>")
        parts.extend(_limited([_episode_line(e, tz, True, True) for e in eps], 20))
    else:
        parts.append("Detected episodes: none.")
    parts.append("")
    if reps:
        parts.append(f"<b>📝 Reported symptoms: {len(reps)}</b>")
        parts.extend(_limited([_symptom_line(r, tz) for r in reps], 10))
    else:
        parts.append("📝 Reported symptoms: none.")
    freq = _frequency_summary(eps, days, tz)
    if freq:
        parts.append("")
        parts.append(freq)
    parts.append("")
    parts.append("Use /report 30 for the doctor PDF + CSV. " + NOT_DIAGNOSIS)
    return _assemble(parts)


RED_FLAG_TEXT = (
    "🚨 <b>Chest pain, fainting or severe breathlessness: call 995 now.</b> "
    "Do not wait for this bot or the doctor report."
)


def symptom_logged(profile: ProfileConfig, rep: SymptomReport) -> str:
    """📝 Confirmation after ``/palp`` / ``/note`` with the heart rate near that time."""
    tz = _tz(profile)
    parts: list[str | None] = [
        RED_FLAG_TEXT if rep.red_flag else None,
        f"📝 <b>Symptom logged — {_name(profile)}</b>",
        f"🕒 {_day_label(rep.event_time, tz)} {_hm(rep.event_time, tz)}",
    ]
    if rep.note:
        parts.append(f"Note: “{esc(rep.note)}”")
    if rep.hr_at_time is not None:
        hr_txt = f"❤️ Heart rate near that time: {_n(rep.hr_at_time, unit=' bpm')}"
        hr_txt += _delta(rep.hr_at_time, rep.baseline_hr, label="at-rest baseline")
        parts.append(hr_txt)
    else:
        parts.append("❤️ Heart rate near that time: n/a (no reading within a few minutes — it may appear after the next sync)")
    if rep.episode_id is not None:
        parts.append(f"🔗 Linked to detected episode #{_n(rep.episode_id)}")
    ex = rep.extracted or {}
    if ex.get("symptoms"):
        parts.append("Symptoms: " + esc(", ".join(s.replace("_", " ") for s in ex["symptoms"])))
    if ex.get("duration_minutes") is not None:
        parts.append(f"Duration: {_n(ex['duration_minutes'], unit=' min')}")
    if ex.get("possible_triggers"):
        parts.append("Before it: " + esc(", ".join(t.replace("_", " ") for t in ex["possible_triggers"])))
    parts.append("")
    parts.append("Saved to the doctor report. If it keeps happening or feels worse, contact your doctor.")
    return _assemble(parts)


def alert_message(alert: Alert) -> str:
    """Severity emoji + title + body for a rules-engine alert."""
    sev = str(getattr(alert, "severity", "") or "info").lower()
    emoji = SEVERITY_EMOJI.get(sev, "ℹ️")
    title = esc(getattr(alert, "title", None) or "Alert")
    who = getattr(alert, "profile", None)
    if who:
        title += f" ({esc(who)})"
    body = getattr(alert, "body", None)
    parts: list[str | None] = [f"{emoji} <b>{title}</b>"]
    if body:
        parts.append(esc(body))
    return _assemble(parts)


def doctor_report_caption(profile: ProfileConfig, days: int, n_episodes: int, n_symptoms: int) -> str:
    """🩺 Caption for the PDF/CSV doctor report (kept under the caption limit)."""
    parts: list[str | None] = [
        f"🩺 <b>Doctor report — {_name(profile)}</b> — last {_n(days)} days",
        f"{_n(n_episodes)} detected episode(s), {_n(n_symptoms)} reported symptom(s).",
        "Inside: how often, what time of day, asleep vs awake, felt vs detected, heart-rate charts.",
        NOT_DIAGNOSIS,
    ]
    return _assemble(parts, limit=MAX_CAPTION_LEN)


def help_text(is_admin: bool) -> str:
    """Command list with one-line descriptions (admin extras when ``is_admin``)."""
    parts: list[str | None] = [
        "🤖 <b>Garmin Health Monitor — commands</b>",
        "",
        "/today — live snapshot: steps, body battery, last heart rate, sync",
        "/yesterday — yesterday's full summary",
        "/sleep — last night's sleep in detail",
        "/hr [YYYY-MM-DD] — heart-rate chart with at-rest excursions",
        "/steps — steps today vs goal and 7-day average",
        "/episodes [days] — palpitation diary (default 7 days)",
        "/palp [HH:MM] [note] — log a palpitation you felt (now, or at HH:MM)",
        "/note &lt;text&gt; — add a note to the diary",
        "/report [days] — doctor report PDF + CSV (default 30 days)",
        "/analyze — ask the AI model for today's coaching now",
        "/profiles — who this chat can see",
        "/help — this list",
    ]
    if is_admin:
        parts.extend(
            [
                "",
                "<b>Admin</b>",
                "/status — service health: last poll, Ollama, sync per profile",
                "/mfa &lt;code&gt; — enter the Garmin verification code when asked",
            ]
        )
    parts.extend(
        [
            "",
            "When a chat can see several people, name them first: <code>/today Dad</code>.",
            "<i>Wrist-sensor data and local-model suggestions, not medical advice.</i>",
        ]
    )
    return _assemble(parts)


def mfa_request(profile: ProfileConfig) -> str:
    """🔐 Ask the admin for the Garmin MFA code."""
    parts: list[str | None] = [
        f"🔐 <b>Garmin Connect needs a verification code ({_name(profile)})</b>",
        "Garmin has sent a one-time code by email or SMS. Reply here with:",
        "<code>/mfa &lt;code&gt;</code>",
        "The login waits about 10 minutes; after that the next poll asks again.",
    ]
    return _assemble(parts)


__all__ = [
    "MAX_CAPTION_LEN",
    "MAX_LEN",
    "alert_message",
    "coaching_block",
    "confidence_label",
    "doctor_report_caption",
    "episode_alert",
    "episodes_list",
    "esc",
    "evening_summary",
    "help_text",
    "hr_message",
    "mfa_request",
    "morning_brief",
    "sleep_message",
    "steps_message",
    "symptom_logged",
    "today_status",
    "weekly_review",
]
