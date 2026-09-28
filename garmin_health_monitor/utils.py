"""Small shared helpers: time parsing/formatting, safe numeric access."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Garmin returns timestamps in several shapes:
#   * epoch milliseconds (ints) inside value arrays  -> UTC
#   * "2024-01-01T05:00:00.0" strings, either GMT or local depending on the key
_GARMIN_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?$")


def now_utc() -> datetime:
    return datetime.now(UTC)


def get_tz(name: str | None) -> ZoneInfo:
    return ZoneInfo(name or "UTC")


def to_local(dt: datetime, tz: str | ZoneInfo) -> datetime:
    """Convert an aware datetime to the given zone (naive datetimes are assumed UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    zone = tz if isinstance(tz, ZoneInfo) else get_tz(tz)
    return dt.astimezone(zone)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def ms_to_utc(ms: int | float | None) -> datetime | None:
    """Epoch milliseconds -> aware UTC datetime (None-safe)."""
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def parse_garmin_gmt(value: Any) -> datetime | None:
    """Parse a Garmin ``*GMT`` timestamp (string like ``2024-01-01T05:00:00.0`` or epoch ms)."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return ms_to_utc(value)
    if isinstance(value, str):
        m = _GARMIN_TS_RE.match(value.strip())
        if not m:
            return None
        d, t, _frac = m.groups()
        try:
            naive = datetime.strptime(f"{d}T{t}", "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
        return naive.replace(tzinfo=UTC)
    return None


def parse_garmin_local(value: Any, tz: str | ZoneInfo) -> datetime | None:
    """Parse a Garmin ``*Local`` timestamp string as wall-clock time in ``tz``."""
    if value is None:
        return None
    zone = tz if isinstance(tz, ZoneInfo) else get_tz(tz)
    if isinstance(value, int | float):
        # Garmin "local" epoch values are UTC epochs shifted by the offset; treat the
        # shifted value as wall-clock time in the profile zone.
        utc_like = ms_to_utc(value)
        if utc_like is None:
            return None
        return utc_like.replace(tzinfo=None).replace(tzinfo=zone)
    if isinstance(value, str):
        m = _GARMIN_TS_RE.match(value.strip())
        if not m:
            return None
        d, t, _frac = m.groups()
        try:
            naive = datetime.strptime(f"{d}T{t}", "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
        return naive.replace(tzinfo=zone)
    return None


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def parse_hhmm(value: str) -> time:
    """Parse ``"07:30"`` into a :class:`datetime.time`."""
    parts = str(value).strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"Expected HH:MM, got {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time {value!r}")
    return time(hour=hour, minute=minute)


def local_day_bounds(day: date, tz: str | ZoneInfo) -> tuple[datetime, datetime]:
    """Return (start, end) aware datetimes covering the local calendar day."""
    zone = tz if isinstance(tz, ZoneInfo) else get_tz(tz)
    start = datetime.combine(day, time.min, tzinfo=zone)
    end = start + timedelta(days=1)
    return start, end


def fmt_hm(dt: datetime | None, tz: str | ZoneInfo | None = None) -> str:
    if dt is None:
        return "--:--"
    if tz is not None:
        dt = to_local(dt, tz)
    return dt.strftime("%H:%M")


def fmt_duration(seconds: float | int | None) -> str:
    """Seconds -> ``7h 42m`` / ``42m`` / ``0m``."""
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def median(values: Iterable[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
