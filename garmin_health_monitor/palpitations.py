"""Heuristic detector for at-rest heart-rate excursions ("palpitation candidates").

What this can and cannot do
---------------------------
* Garmin Connect exposes wrist heart rate at **2-minute** resolution.  A
  palpitation that lasts seconds is invisible; runs of a few minutes or more
  show up as a sustained excursion.  The watch's own "abnormal heart rate"
  alert count (``DaySummary.abnormal_hr_alerts``) is a useful second signal.
* An optical sensor cannot see rhythm.  We only see *rate*.  The output is a
  symptom diary to discuss with a doctor, never a diagnosis.

Algorithm
---------
1. Annotate every HR sample with context: steps in its 15-minute bucket,
   Garmin's activity level, whether a recorded / auto-detected activity (plus a
   cool-down) covers it, whether the person was asleep, the stress reading.
2. A sample is "at rest" when nothing above says exertion.
3. Maintain a causal baseline: the median of the last N minutes of at-rest,
   non-flagged samples (falls back to the daily resting HR).
4. Flag at-rest samples that exceed an absolute threshold or the baseline plus
   a rise; asleep samples use lower thresholds.
5. Group flagged samples into episodes (small gaps tolerated), classify as
   ``sustained`` / ``spike`` / ``nocturnal`` and score a heuristic confidence.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import PalpitationConfig
from .models import (
    EPISODE_KIND_NOCTURNAL,
    EPISODE_KIND_SPIKE,
    EPISODE_KIND_SUSTAINED,
    EPISODE_SOURCE_DETECTOR,
    DaySnapshot,
    Episode,
    HRSample,
    StepBucket,
)
from .utils import clamp, mean, median, stable_hash

logger = logging.getLogger(__name__)

ACTIVE_LEVELS = {"active", "highlyactive", "highly_active"}
SLEEP_LEVELS = {"sleeping", "sleep"}
DEFAULT_SAMPLE_INTERVAL = timedelta(minutes=2)
STRESS_MOVEMENT = -2  # Garmin: too much motion to measure
STRESS_UNKNOWN = -1


@dataclass(slots=True)
class SampleContext:
    ts: datetime
    hr: int
    steps: int
    activity_level: str
    in_activity: bool
    asleep: bool
    stress: int | None
    at_rest: bool
    baseline: float | None = None
    flagged: bool = False


class _StepIndex:
    def __init__(self, buckets: list[StepBucket]):
        self.buckets = sorted(buckets, key=lambda b: b.start)
        self.starts = [b.start for b in self.buckets]

    def lookup(self, ts: datetime) -> StepBucket | None:
        if not self.buckets:
            return None
        i = bisect_right(self.starts, ts) - 1
        if i < 0:
            return None
        b = self.buckets[i]
        if b.start <= ts < b.end:
            return b
        return None


class _StressIndex:
    def __init__(self, samples: list) -> None:
        self.samples = sorted(samples, key=lambda s: s.ts)
        self.ts = [s.ts for s in self.samples]

    def lookup(self, ts: datetime, tolerance: timedelta = timedelta(minutes=3)) -> int | None:
        if not self.samples:
            return None
        i = bisect_right(self.ts, ts)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.samples):
                cand = self.samples[j]
                if abs((cand.ts - ts).total_seconds()) <= tolerance.total_seconds():
                    if best is None or abs((cand.ts - ts).total_seconds()) < abs((best.ts - ts).total_seconds()):
                        best = cand
        return best.level if best else None

    def window(self, start: datetime, end: datetime) -> list[int]:
        lo = bisect_right(self.ts, start - timedelta(minutes=3))
        hi = bisect_right(self.ts, end + timedelta(minutes=3))
        return [s.level for s in self.samples[lo:hi]]


def sleep_windows(snap: DaySnapshot, extra: list[tuple[datetime, datetime]] | None = None) -> list[tuple[datetime, datetime]]:
    wins: list[tuple[datetime, datetime]] = []
    if snap.sleep and snap.sleep.start and snap.sleep.end and snap.sleep.end > snap.sleep.start:
        wins.append((snap.sleep.start, snap.sleep.end))
    for w in extra or []:
        if w and w[0] and w[1] and w[1] > w[0]:
            wins.append((w[0], w[1]))
    return wins


def estimate_interval(samples: list[HRSample]) -> timedelta:
    if len(samples) < 2:
        return DEFAULT_SAMPLE_INTERVAL
    diffs = sorted((b.ts - a.ts).total_seconds() for a, b in zip(samples, samples[1:], strict=False) if b.ts > a.ts)
    if not diffs:
        return DEFAULT_SAMPLE_INTERVAL
    med = diffs[len(diffs) // 2]
    if med <= 0 or med > 15 * 60:
        return DEFAULT_SAMPLE_INTERVAL
    return timedelta(seconds=med)


def annotate(
    snap: DaySnapshot,
    cfg: PalpitationConfig,
    extra_sleep_windows: list[tuple[datetime, datetime]] | None = None,
) -> list[SampleContext]:
    """Attach movement / sleep / activity context to each HR sample."""
    samples = sorted(snap.hr, key=lambda s: s.ts)
    steps_idx = _StepIndex(snap.steps)
    stress_idx = _StressIndex(snap.stress)
    cooldown = timedelta(minutes=cfg.post_activity_cooldown_minutes)
    windows = [(a.start - timedelta(minutes=2), a.end + cooldown) for a in snap.activities]
    sleeps = sleep_windows(snap, extra_sleep_windows)

    out: list[SampleContext] = []
    for s in samples:
        bucket = steps_idx.lookup(s.ts)
        steps = bucket.steps if bucket else 0
        level = (bucket.activity_level if bucket else "unknown") or "unknown"
        level_l = level.lower()
        in_activity = any(ws <= s.ts <= we for ws, we in windows)
        asleep = any(ws <= s.ts <= we for ws, we in sleeps) or level_l in SLEEP_LEVELS
        stress = stress_idx.lookup(s.ts)
        at_rest = (
            not in_activity
            and steps <= cfg.max_steps_in_window
            and level_l not in ACTIVE_LEVELS
        )
        out.append(
            SampleContext(
                ts=s.ts,
                hr=s.hr,
                steps=steps,
                activity_level=level,
                in_activity=in_activity,
                asleep=asleep,
                stress=stress,
                at_rest=at_rest,
            )
        )
    return out


def _fallback_baseline(snap: DaySnapshot, asleep: bool) -> float | None:
    s = snap.summary
    if asleep and snap.sleep and snap.sleep.resting_hr:
        return float(snap.sleep.resting_hr)
    for cand in (s.resting_hr, s.resting_hr_7d_avg):
        if cand:
            return float(cand)
    return None


def _confidence(ep: Episode, cfg: PalpitationConfig, has_baseline: bool) -> float:
    c = 0.25
    if has_baseline:
        c += clamp(ep.delta_hr / 100.0, 0.0, 0.30)
    else:
        c += 0.10
    if ep.steps_in_window == 0:
        c += 0.15
    elif ep.steps_in_window <= 30:
        c += 0.07
    if ep.duration_min >= 6:
        c += 0.10
    elif ep.duration_min >= cfg.min_duration_minutes:
        c += 0.05
    if ep.max_jump_bpm >= 30:
        c += 0.10
    elif ep.max_jump_bpm >= 20:
        c += 0.05
    c -= 0.25 * ep.movement_fraction
    if ep.kind == EPISODE_KIND_SPIKE:
        c -= 0.10
    if ep.activity_level.lower() == "generic":
        c -= 0.05
    return round(clamp(c, 0.05, 0.95), 3)


def detect_episodes(
    snap: DaySnapshot,
    cfg: PalpitationConfig | None = None,
    extra_sleep_windows: list[tuple[datetime, datetime]] | None = None,
) -> list[Episode]:
    """Return candidate palpitation episodes for one day's snapshot."""
    cfg = cfg or PalpitationConfig()
    ctx = annotate(snap, cfg, extra_sleep_windows)
    if len(ctx) < 3:
        return []
    interval = estimate_interval(snap.hr)
    interval_min = interval.total_seconds() / 60.0
    window = timedelta(minutes=cfg.baseline_window_minutes)
    min_baseline_samples = 5

    # --- causal baseline + flagging --------------------------------------
    recent: deque[SampleContext] = deque()
    for c in ctx:
        while recent and (c.ts - recent[0].ts) > window:
            recent.popleft()
        base = median(x.hr for x in recent) if len(recent) >= min_baseline_samples else None
        if base is None:
            base = _fallback_baseline(snap, c.asleep)
        c.baseline = base
        if c.at_rest:
            if c.asleep:
                c.flagged = c.hr >= cfg.nocturnal_hr_threshold or (base is not None and c.hr >= base + cfg.nocturnal_rise)
            else:
                c.flagged = c.hr >= cfg.rest_hr_threshold or (base is not None and c.hr >= base + cfg.rise_over_baseline)
        else:
            c.flagged = False
        if c.at_rest and not c.flagged:
            recent.append(c)

    # --- group into runs ---------------------------------------------------
    runs: list[list[int]] = []
    current: list[int] = []
    gap = 0
    max_gap_seconds = (cfg.gap_tolerance_samples + 1) * interval.total_seconds() + 60
    for i, c in enumerate(ctx):
        if c.flagged:
            if current and (c.ts - ctx[current[-1]].ts).total_seconds() > max_gap_seconds:
                runs.append(current)
                current = []
            current.append(i)
            gap = 0
        elif current:
            gap += 1
            if gap > cfg.gap_tolerance_samples or not c.at_rest:
                runs.append(current)
                current = []
                gap = 0
    if current:
        runs.append(current)

    stress_idx = _StressIndex(snap.stress)
    steps_idx = _StepIndex(snap.steps)
    episodes: list[Episode] = []
    for run in runs:
        first, last = ctx[run[0]], ctx[run[-1]]
        members = [ctx[i] for i in range(run[0], run[-1] + 1)]
        flagged_members = [ctx[i] for i in run]
        peak = max(m.hr for m in flagged_members)
        mean_hr = mean(m.hr for m in members) or float(peak)
        base = first.baseline
        has_baseline = base is not None
        base_val = float(base) if base is not None else float(min(m.hr for m in members))
        duration_min = (last.ts - first.ts).total_seconds() / 60.0 + interval_min
        # onset jump: compare first flagged sample with the previous sample
        jumps = []
        if run[0] > 0:
            jumps.append(abs(first.hr - ctx[run[0] - 1].hr))
        jumps.extend(abs(b.hr - a.hr) for a, b in zip(members, members[1:], strict=False))
        max_jump = max(jumps) if jumps else 0
        asleep = sum(1 for m in flagged_members if m.asleep) * 2 >= len(flagged_members)
        # context windows
        bucket_steps = []
        levels = []
        for m in members:
            b = steps_idx.lookup(m.ts)
            if b:
                bucket_steps.append(b.steps)
                levels.append(b.activity_level)
        steps_in_window = max(bucket_steps) if bucket_steps else max(m.steps for m in members)
        level = "unknown"
        if levels:
            order = ["sleeping", "sedentary", "generic", "none", "active", "highlyActive"]
            level = max(levels, key=lambda lv: order.index(lv) if lv in order else 2)
        stress_vals = stress_idx.window(first.ts, last.ts)
        measured = [v for v in stress_vals if v is not None and v >= 0]
        movement = [v for v in stress_vals if v == STRESS_MOVEMENT]
        movement_fraction = (len(movement) / len(stress_vals)) if stress_vals else 0.0

        if asleep:
            kind = EPISODE_KIND_NOCTURNAL
        elif duration_min >= cfg.min_duration_minutes:
            kind = EPISODE_KIND_SUSTAINED
        elif peak >= cfg.spike_hr_threshold or max_jump >= cfg.spike_jump_bpm:
            kind = EPISODE_KIND_SPIKE
        else:
            continue  # one lonely sample just over the line: not worth a record
        if asleep and duration_min < cfg.min_duration_minutes and peak < cfg.spike_hr_threshold and max_jump < cfg.spike_jump_bpm:
            continue

        ep = Episode(
            profile=snap.profile,
            start=first.ts,
            end=last.ts + interval,
            duration_min=round(duration_min, 1),
            peak_hr=peak,
            mean_hr=round(mean_hr, 1),
            baseline_hr=round(base_val, 1),
            delta_hr=round(peak - base_val, 1),
            max_jump_bpm=int(max_jump),
            asleep=asleep,
            steps_in_window=int(steps_in_window),
            activity_level=level,
            stress_avg=round(mean(measured), 1) if measured else None,
            movement_fraction=round(movement_fraction, 2),
            kind=kind,
            source=EPISODE_SOURCE_DETECTOR,
            samples=[HRSample(ts=m.ts, hr=m.hr) for m in members],
        )
        ep.confidence = _confidence(ep, cfg, has_baseline)
        ep.fingerprint = stable_hash(snap.profile, first.ts.strftime("%Y-%m-%dT%H:%M"), kind)
        episodes.append(ep)

    logger.debug("%s %s: %d candidate episodes", snap.profile, snap.date_str, len(episodes))
    return episodes


def hr_context_at(snap: DaySnapshot, when: datetime, cfg: PalpitationConfig | None = None) -> tuple[int | None, float | None]:
    """HR closest to ``when`` (within 6 minutes) and the at-rest baseline before it.

    Used to enrich a person-reported symptom (``/palp``) with numbers.
    """
    cfg = cfg or PalpitationConfig()
    ctx = annotate(snap, cfg)
    if not ctx:
        return None, None
    nearest = min(ctx, key=lambda c: abs((c.ts - when).total_seconds()))
    if abs((nearest.ts - when).total_seconds()) > 6 * 60:
        return None, None
    window = timedelta(minutes=cfg.baseline_window_minutes)
    prior = [c.hr for c in ctx if c.at_rest and (when - window) <= c.ts < when - timedelta(minutes=1)]
    base = median(prior) if len(prior) >= 3 else _fallback_baseline(snap, nearest.asleep)
    return nearest.hr, base
