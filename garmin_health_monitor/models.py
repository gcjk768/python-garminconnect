"""Dataclasses shared by every module.

All datetimes stored on these objects are timezone-aware UTC unless the field
name says ``local``.  Conversion to the profile's zone happens at the edges
(message formatting, charts, reports).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Time series primitives
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HRSample:
    ts: datetime
    hr: int


@dataclass(slots=True)
class StepBucket:
    """15-minute step bucket from ``get_steps_data``."""

    start: datetime
    end: datetime
    steps: int
    activity_level: str  # sedentary | sleeping | active | highlyActive | generic | none


@dataclass(slots=True)
class StressSample:
    ts: datetime
    level: int  # 0..100; negative = Garmin could not measure (-1 unknown, -2 movement/activity)


@dataclass(slots=True)
class BodyBatterySample:
    ts: datetime
    level: int


@dataclass(slots=True)
class ValueSample:
    ts: datetime
    value: float


@dataclass(slots=True)
class ActivityWindow:
    """A recorded or auto-detected activity (used to exclude exertion from palpitation detection)."""

    start: datetime
    end: datetime
    name: str = ""
    type_key: str = ""
    activity_id: int | None = None
    duration_s: float | None = None
    avg_hr: int | None = None
    max_hr: int | None = None
    distance_m: float | None = None
    calories: float | None = None
    source: str = "recorded"  # recorded | auto_detected


# ---------------------------------------------------------------------------
# Daily summaries
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DaySummary:
    """Normalised subset of ``get_user_summary``."""

    total_steps: int | None = None
    step_goal: int | None = None
    distance_m: float | None = None
    total_kcal: float | None = None
    active_kcal: float | None = None
    bmr_kcal: float | None = None
    min_hr: int | None = None
    max_hr: int | None = None
    resting_hr: int | None = None
    resting_hr_7d_avg: int | None = None
    sleeping_seconds: int | None = None
    sedentary_seconds: int | None = None
    active_seconds: int | None = None
    highly_active_seconds: int | None = None
    moderate_intensity_min: int | None = None
    vigorous_intensity_min: int | None = None
    intensity_goal_min: int | None = None
    floors_up: float | None = None
    avg_stress: int | None = None
    max_stress: int | None = None
    stress_qualifier: str | None = None
    rest_stress_seconds: int | None = None
    low_stress_seconds: int | None = None
    medium_stress_seconds: int | None = None
    high_stress_seconds: int | None = None
    body_battery_charged: int | None = None
    body_battery_drained: int | None = None
    body_battery_high: int | None = None
    body_battery_low: int | None = None
    body_battery_latest: int | None = None
    abnormal_hr_alerts: int | None = None
    avg_spo2: float | None = None
    lowest_spo2: int | None = None
    avg_waking_respiration: float | None = None
    last_sync: datetime | None = None
    wellness_end: datetime | None = None


@dataclass(slots=True)
class SleepSummary:
    start: datetime | None = None
    end: datetime | None = None
    total_seconds: int | None = None
    deep_seconds: int | None = None
    light_seconds: int | None = None
    rem_seconds: int | None = None
    awake_seconds: int | None = None
    nap_seconds: int | None = None
    score: int | None = None
    score_qualifier: str | None = None
    avg_hrv: float | None = None
    hrv_status: str | None = None
    avg_spo2: float | None = None
    lowest_spo2: int | None = None
    avg_respiration: float | None = None
    resting_hr: int | None = None
    restless_moments: int | None = None
    awake_count: int | None = None
    body_battery_change: int | None = None
    sleep_hr: list[HRSample] = field(default_factory=list)


@dataclass(slots=True)
class HrvSummary:
    weekly_avg: float | None = None
    last_night_avg: float | None = None
    last_night_5min_high: float | None = None
    status: str | None = None  # BALANCED | UNBALANCED | LOW | POOR ...
    feedback: str | None = None
    baseline_low_upper: float | None = None
    baseline_balanced_low: float | None = None
    baseline_balanced_upper: float | None = None
    readings: list[ValueSample] = field(default_factory=list)


@dataclass(slots=True)
class ReadinessSummary:
    score: int | None = None
    level: str | None = None
    feedback_short: str | None = None
    feedback_long: str | None = None
    sleep_score: int | None = None
    recovery_time_min: int | None = None
    hrv_weekly_avg: float | None = None


@dataclass(slots=True)
class SpO2Summary:
    average: float | None = None
    lowest: int | None = None
    latest: int | None = None
    avg_sleep: float | None = None
    last_7d_avg: float | None = None
    samples: list[ValueSample] = field(default_factory=list)


@dataclass(slots=True)
class RespirationSummary:
    avg_waking: float | None = None
    avg_sleep: float | None = None
    lowest: float | None = None
    highest: float | None = None
    samples: list[ValueSample] = field(default_factory=list)


@dataclass(slots=True)
class DeviceInfo:
    name: str | None = None
    last_upload: datetime | None = None


@dataclass(slots=True)
class DaySnapshot:
    """Everything the monitor knows about one profile for one local calendar day."""

    profile: str
    day: date
    tz: str
    fetched_at: datetime
    summary: DaySummary = field(default_factory=DaySummary)
    hr: list[HRSample] = field(default_factory=list)
    steps: list[StepBucket] = field(default_factory=list)
    stress: list[StressSample] = field(default_factory=list)
    body_battery: list[BodyBatterySample] = field(default_factory=list)
    activities: list[ActivityWindow] = field(default_factory=list)
    sleep: SleepSummary | None = None
    hrv: HrvSummary | None = None
    readiness: ReadinessSummary | None = None
    spo2: SpO2Summary | None = None
    respiration: RespirationSummary | None = None
    device: DeviceInfo | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def date_str(self) -> str:
        return self.day.isoformat()


# ---------------------------------------------------------------------------
# Palpitation tracking
# ---------------------------------------------------------------------------

EPISODE_KIND_SUSTAINED = "sustained"
EPISODE_KIND_SPIKE = "spike"
EPISODE_KIND_NOCTURNAL = "nocturnal"

EPISODE_SOURCE_DETECTOR = "detector"
EPISODE_SOURCE_GARMIN_ALERT = "garmin_alert"
EPISODE_SOURCE_MANUAL = "manual"

ASSESSMENT_POSSIBLE = "possible_palpitation"
ASSESSMENT_EXERTION = "likely_exertion"
ASSESSMENT_ARTIFACT = "likely_artifact"
ASSESSMENT_UNCLEAR = "unclear"
ASSESSMENTS = (ASSESSMENT_POSSIBLE, ASSESSMENT_EXERTION, ASSESSMENT_ARTIFACT, ASSESSMENT_UNCLEAR)


@dataclass(slots=True)
class Episode:
    """A candidate palpitation episode: an at-rest heart-rate excursion."""

    profile: str
    start: datetime
    end: datetime
    duration_min: float
    peak_hr: int
    mean_hr: float
    baseline_hr: float
    delta_hr: float
    max_jump_bpm: int = 0
    asleep: bool = False
    steps_in_window: int = 0
    activity_level: str = "unknown"
    stress_avg: float | None = None
    movement_fraction: float = 0.0  # share of stress samples Garmin flagged as movement
    confidence: float = 0.0  # heuristic 0..1
    kind: str = EPISODE_KIND_SUSTAINED
    source: str = EPISODE_SOURCE_DETECTOR
    samples: list[HRSample] = field(default_factory=list)
    id: int | None = None
    fingerprint: str = ""
    llm_assessment: str | None = None
    llm_confidence: float | None = None
    llm_reasoning: str | None = None
    doctor_note: str | None = None
    llm_model: str | None = None
    felt: bool | None = None  # None = not answered, True = person felt it, False = did not notice
    notes: str | None = None
    notified_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_samples:
            d.pop("samples", None)
        else:
            d["samples"] = [{"ts": s.ts.isoformat(), "hr": s.hr} for s in self.samples]
        for key in ("start", "end", "notified_at", "created_at", "updated_at"):
            if d.get(key) is not None:
                d[key] = d[key].isoformat()
        return d


@dataclass(slots=True)
class SymptomReport:
    """A person-reported palpitation (``/palp`` command or an inline button answer)."""

    profile: str
    reported_at: datetime
    event_time: datetime
    note: str = ""
    hr_at_time: int | None = None
    baseline_hr: float | None = None
    episode_id: int | None = None
    chat_id: int | None = None
    source: str = "command"  # command | button
    id: int | None = None


@dataclass(slots=True)
class EpisodeAssessment:
    """Structured answer from the Ollama model about one episode."""

    assessment: str
    confidence: float
    reasoning: str
    doctor_note: str
    model: str


@dataclass(slots=True)
class CoachingAdvice:
    """Structured answer from the Ollama model about a day/week of activity."""

    summary: str
    do_more: list[str]
    do_less: list[str]
    watch_outs: list[str]
    heart_note: str | None
    model: str
    period: str = "daily"  # daily | weekly

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Alert:
    """An event-driven notification produced by the rules engine."""

    key: str  # dedupe key, e.g. "rhr_high:Dad:2026-09-28"
    severity: str  # info | warning | critical
    title: str
    body: str
    profile: str
