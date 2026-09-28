"""Turn raw Garmin Connect payloads into a :class:`DaySnapshot`.

Everything here is defensive: Garmin omits keys, returns ``null`` inside value
arrays, and changes descriptor order between accounts.  A missing endpoint
simply yields an empty list / ``None`` and is recorded in ``snapshot.errors``
by the caller.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from .models import (
    ActivityWindow,
    BodyBatterySample,
    DaySnapshot,
    DaySummary,
    DeviceInfo,
    HRSample,
    HrvSummary,
    ReadinessSummary,
    RespirationSummary,
    SleepSummary,
    SpO2Summary,
    StepBucket,
    StressSample,
    ValueSample,
)
from .utils import as_float, as_int, ms_to_utc, now_utc, parse_garmin_gmt

logger = logging.getLogger(__name__)

# Keys used in the raw payload dict produced by garmin_client.fetch_day
RAW_SUMMARY = "summary"
RAW_HEART_RATES = "heart_rates"
RAW_STEPS = "steps"
RAW_STRESS = "stress"
RAW_SLEEP = "sleep"
RAW_HRV = "hrv"
RAW_ACTIVITIES = "activities"
RAW_BB_EVENTS = "body_battery_events"
RAW_RESPIRATION = "respiration"
RAW_SPO2 = "spo2"
RAW_READINESS = "training_readiness"
RAW_DEVICE = "device_last_used"


def _descriptor_index(descriptors: Any, key: str, default: int, key_field: str = "key", index_field: str = "index") -> int:
    """Find the array index for ``key`` from a Garmin descriptor list."""
    if isinstance(descriptors, list):
        for d in descriptors:
            if isinstance(d, dict) and str(d.get(key_field, "")).lower() == key.lower():
                idx = as_int(d.get(index_field))
                if idx is not None:
                    return idx
    return default


def _pairs(values: Any, ts_idx: int, val_idx: int) -> list[tuple[datetime, Any]]:
    out: list[tuple[datetime, Any]] = []
    if not isinstance(values, list):
        return out
    for row in values:
        if not isinstance(row, list | tuple) or len(row) <= max(ts_idx, val_idx):
            continue
        ts = ms_to_utc(row[ts_idx]) if isinstance(row[ts_idx], int | float) else parse_garmin_gmt(row[ts_idx])
        if ts is None:
            continue
        out.append((ts, row[val_idx]))
    out.sort(key=lambda p: p[0])
    return out


# ---------------------------------------------------------------------------
# Individual endpoint normalisers
# ---------------------------------------------------------------------------


def normalize_summary(raw: dict[str, Any] | None) -> DaySummary:
    r = raw if isinstance(raw, dict) else {}
    return DaySummary(
        total_steps=as_int(r.get("totalSteps")),
        step_goal=as_int(r.get("dailyStepGoal")),
        distance_m=as_float(r.get("totalDistanceMeters")),
        total_kcal=as_float(r.get("totalKilocalories")),
        active_kcal=as_float(r.get("activeKilocalories")),
        bmr_kcal=as_float(r.get("bmrKilocalories")),
        min_hr=as_int(r.get("minHeartRate")),
        max_hr=as_int(r.get("maxHeartRate")),
        resting_hr=as_int(r.get("restingHeartRate")),
        resting_hr_7d_avg=as_int(r.get("lastSevenDaysAvgRestingHeartRate")),
        sleeping_seconds=as_int(r.get("sleepingSeconds")),
        sedentary_seconds=as_int(r.get("sedentarySeconds")),
        active_seconds=as_int(r.get("activeSeconds")),
        highly_active_seconds=as_int(r.get("highlyActiveSeconds")),
        moderate_intensity_min=as_int(r.get("moderateIntensityMinutes")),
        vigorous_intensity_min=as_int(r.get("vigorousIntensityMinutes")),
        intensity_goal_min=as_int(r.get("intensityMinutesGoal")),
        floors_up=as_float(r.get("floorsAscended")),
        avg_stress=as_int(r.get("averageStressLevel")),
        max_stress=as_int(r.get("maxStressLevel")),
        stress_qualifier=r.get("stressQualifier"),
        rest_stress_seconds=as_int(r.get("restStressDuration")),
        low_stress_seconds=as_int(r.get("lowStressDuration")),
        medium_stress_seconds=as_int(r.get("mediumStressDuration")),
        high_stress_seconds=as_int(r.get("highStressDuration")),
        body_battery_charged=as_int(r.get("bodyBatteryChargedValue")),
        body_battery_drained=as_int(r.get("bodyBatteryDrainedValue")),
        body_battery_high=as_int(r.get("bodyBatteryHighestValue")),
        body_battery_low=as_int(r.get("bodyBatteryLowestValue")),
        body_battery_latest=as_int(r.get("bodyBatteryMostRecentValue")),
        abnormal_hr_alerts=as_int(r.get("abnormalHeartRateAlertsCount")),
        avg_spo2=as_float(r.get("averageSpo2")),
        lowest_spo2=as_int(r.get("lowestSpo2")),
        avg_waking_respiration=as_float(r.get("avgWakingRespirationValue")),
        last_sync=parse_garmin_gmt(r.get("lastSyncTimestampGMT")),
        wellness_end=parse_garmin_gmt(r.get("wellnessEndTimeGmt")),
    )


def normalize_heart_rates(raw: dict[str, Any] | None) -> list[HRSample]:
    r = raw if isinstance(raw, dict) else {}
    ts_idx = _descriptor_index(r.get("heartRateValueDescriptors"), "timestamp", 0)
    hr_idx = _descriptor_index(r.get("heartRateValueDescriptors"), "heartrate", 1)
    out: list[HRSample] = []
    for ts, val in _pairs(r.get("heartRateValues"), ts_idx, hr_idx):
        hr = as_int(val)
        if hr is None or hr <= 0:
            continue
        out.append(HRSample(ts=ts, hr=hr))
    return out


def normalize_steps(raw: list[dict[str, Any]] | None) -> list[StepBucket]:
    out: list[StepBucket] = []
    if not isinstance(raw, list):
        return out
    for b in raw:
        if not isinstance(b, dict):
            continue
        start = parse_garmin_gmt(b.get("startGMT"))
        end = parse_garmin_gmt(b.get("endGMT"))
        if start is None:
            continue
        if end is None:
            end = start + timedelta(minutes=15)
        out.append(
            StepBucket(
                start=start,
                end=end,
                steps=as_int(b.get("steps"), 0) or 0,
                activity_level=str(b.get("primaryActivityLevel") or "none"),
            )
        )
    out.sort(key=lambda s: s.start)
    return out


def normalize_stress(raw: dict[str, Any] | None) -> tuple[list[StressSample], list[BodyBatterySample]]:
    r = raw if isinstance(raw, dict) else {}
    ts_idx = _descriptor_index(r.get("stressValueDescriptorsDTOList"), "timestamp", 0)
    lvl_idx = _descriptor_index(r.get("stressValueDescriptorsDTOList"), "stressLevel", 1)
    stress = []
    for ts, val in _pairs(r.get("stressValuesArray"), ts_idx, lvl_idx):
        lvl = as_int(val)
        if lvl is None:
            continue
        stress.append(StressSample(ts=ts, level=lvl))

    bb_desc = r.get("bodyBatteryValueDescriptorsDTOList")
    bb_ts_idx = _descriptor_index(
        bb_desc, "timestamp", 0, key_field="bodyBatteryValueDescriptorKey", index_field="bodyBatteryValueDescriptorIndex"
    )
    bb_lvl_idx = _descriptor_index(
        bb_desc, "bodyBatteryLevel", 2, key_field="bodyBatteryValueDescriptorKey", index_field="bodyBatteryValueDescriptorIndex"
    )
    body_battery = []
    for ts, val in _pairs(r.get("bodyBatteryValuesArray"), bb_ts_idx, bb_lvl_idx):
        lvl = as_int(val)
        if lvl is None:
            continue
        body_battery.append(BodyBatterySample(ts=ts, level=lvl))
    return stress, body_battery


def normalize_sleep(raw: dict[str, Any] | None) -> SleepSummary | None:
    r = raw if isinstance(raw, dict) else {}
    dto = r.get("dailySleepDTO")
    if not isinstance(dto, dict):
        return None
    start = ms_to_utc(dto.get("sleepStartTimestampGMT")) or parse_garmin_gmt(dto.get("sleepStartTimestampGMT"))
    end = ms_to_utc(dto.get("sleepEndTimestampGMT")) or parse_garmin_gmt(dto.get("sleepEndTimestampGMT"))
    total = as_int(dto.get("sleepTimeSeconds"))
    if start is None and total is None:
        return None
    scores = dto.get("sleepScores") if isinstance(dto.get("sleepScores"), dict) else {}
    overall = scores.get("overall") if isinstance(scores.get("overall"), dict) else {}
    awake_count = scores.get("awakeCount") if isinstance(scores.get("awakeCount"), dict) else {}
    sleep_hr: list[HRSample] = []
    for item in r.get("sleepHeartRate") or []:
        if not isinstance(item, dict):
            continue
        ts = ms_to_utc(item.get("startGMT")) or parse_garmin_gmt(item.get("startGMT"))
        hr = as_int(item.get("value"))
        if ts is not None and hr and hr > 0:
            sleep_hr.append(HRSample(ts=ts, hr=hr))
    sleep_hr.sort(key=lambda s: s.ts)
    return SleepSummary(
        start=start,
        end=end,
        total_seconds=total,
        deep_seconds=as_int(dto.get("deepSleepSeconds")),
        light_seconds=as_int(dto.get("lightSleepSeconds")),
        rem_seconds=as_int(dto.get("remSleepSeconds")),
        awake_seconds=as_int(dto.get("awakeSleepSeconds")),
        nap_seconds=as_int(dto.get("napTimeSeconds")),
        score=as_int(overall.get("value")),
        score_qualifier=overall.get("qualifierKey"),
        avg_hrv=as_float(dto.get("avgSleepHRV")) or as_float(r.get("avgOvernightHrv")),
        hrv_status=r.get("hrvStatus"),
        avg_spo2=as_float(dto.get("averageSpO2Value")) or as_float(dto.get("avgSpO2")),
        lowest_spo2=as_int(dto.get("lowestSpO2Value")),
        avg_respiration=as_float(dto.get("averageRespirationValue")),
        resting_hr=as_int(r.get("restingHeartRate")),
        restless_moments=as_int(r.get("restlessMomentsCount")),
        awake_count=as_int(awake_count.get("value")),
        body_battery_change=as_int(r.get("bodyBatteryChange")),
        sleep_hr=sleep_hr,
    )


def normalize_hrv(raw: dict[str, Any] | None) -> HrvSummary | None:
    r = raw if isinstance(raw, dict) else {}
    summary = r.get("hrvSummary")
    readings: list[ValueSample] = []
    for item in r.get("hrvReadings") or []:
        if not isinstance(item, dict):
            continue
        ts = parse_garmin_gmt(item.get("readingTimeGMT"))
        val = as_float(item.get("hrvValue"))
        if ts is not None and val is not None:
            readings.append(ValueSample(ts=ts, value=val))
    if not isinstance(summary, dict) and not readings:
        return None
    s = summary if isinstance(summary, dict) else {}
    baseline = s.get("baseline") if isinstance(s.get("baseline"), dict) else {}
    return HrvSummary(
        weekly_avg=as_float(s.get("weeklyAvg")),
        last_night_avg=as_float(s.get("lastNightAvg")),
        last_night_5min_high=as_float(s.get("lastNight5MinHigh")),
        status=s.get("status"),
        feedback=s.get("feedbackPhrase"),
        baseline_low_upper=as_float(baseline.get("lowUpper")),
        baseline_balanced_low=as_float(baseline.get("balancedLow")),
        baseline_balanced_upper=as_float(baseline.get("balancedUpper")),
        readings=readings,
    )


def normalize_activities(raw: list[dict[str, Any]] | None) -> list[ActivityWindow]:
    out: list[ActivityWindow] = []
    if not isinstance(raw, list):
        return out
    for a in raw:
        if not isinstance(a, dict):
            continue
        start = parse_garmin_gmt(a.get("startTimeGMT"))
        if start is None:
            continue
        duration = as_float(a.get("elapsedDuration")) or as_float(a.get("duration")) or 0.0
        end = start + timedelta(seconds=max(duration, 60.0))
        atype = a.get("activityType") if isinstance(a.get("activityType"), dict) else {}
        out.append(
            ActivityWindow(
                start=start,
                end=end,
                name=str(a.get("activityName") or atype.get("typeKey") or "activity"),
                type_key=str(atype.get("typeKey") or ""),
                activity_id=as_int(a.get("activityId")),
                duration_s=as_float(a.get("duration")),
                avg_hr=as_int(a.get("averageHR")),
                max_hr=as_int(a.get("maxHR")),
                distance_m=as_float(a.get("distance")),
                calories=as_float(a.get("calories")),
                source="recorded",
            )
        )
    out.sort(key=lambda w: w.start)
    return out


def normalize_bb_events(raw: list[dict[str, Any]] | None) -> list[ActivityWindow]:
    """Auto-detected activities from body-battery events (walks the watch noticed)."""
    out: list[ActivityWindow] = []
    if not isinstance(raw, list):
        return out
    for e in raw:
        if not isinstance(e, dict):
            continue
        etype = str(e.get("eventType") or "").upper()
        if "ACTIVITY" not in etype:
            continue
        start = parse_garmin_gmt(e.get("eventStartTimeGmt"))
        dur_ms = as_float(e.get("durationInMilliseconds"))
        if start is None or not dur_ms:
            continue
        out.append(
            ActivityWindow(
                start=start,
                end=start + timedelta(milliseconds=dur_ms),
                name=str(e.get("shortFeedback") or etype.replace("_", " ").title()),
                type_key=etype.lower(),
                duration_s=dur_ms / 1000.0,
                source="auto_detected" if "AUTO" in etype else "recorded",
            )
        )
    out.sort(key=lambda w: w.start)
    return out


def normalize_respiration(raw: dict[str, Any] | None) -> RespirationSummary | None:
    r = raw if isinstance(raw, dict) else {}
    if not r:
        return None
    ts_idx = _descriptor_index(r.get("respirationValueDescriptorsDTOList"), "timestamp", 0)
    val_idx = _descriptor_index(r.get("respirationValueDescriptorsDTOList"), "respiration", 1)
    samples = []
    for ts, val in _pairs(r.get("respirationValuesArray"), ts_idx, val_idx):
        v = as_float(val)
        if v is None or v < 0:
            continue
        samples.append(ValueSample(ts=ts, value=v))
    return RespirationSummary(
        avg_waking=as_float(r.get("avgWakingRespirationValue")),
        avg_sleep=as_float(r.get("avgSleepRespirationValue")),
        lowest=as_float(r.get("lowestRespirationValue")),
        highest=as_float(r.get("highestRespirationValue")),
        samples=samples,
    )


def normalize_spo2(raw: dict[str, Any] | None) -> SpO2Summary | None:
    r = raw if isinstance(raw, dict) else {}
    if not r:
        return None
    samples = []
    for key in ("spO2HourlyAverages", "spO2ValuesArray"):
        for ts, val in _pairs(r.get(key), 0, 1):
            v = as_float(val)
            if v is None or v <= 0:
                continue
            samples.append(ValueSample(ts=ts, value=v))
        if samples:
            break
    return SpO2Summary(
        average=as_float(r.get("averageSpO2")),
        lowest=as_int(r.get("lowestSpO2")),
        latest=as_int(r.get("latestSpO2")),
        avg_sleep=as_float(r.get("avgSleepSpO2")),
        last_7d_avg=as_float(r.get("lastSevenDaysAvgSpO2")),
        samples=samples,
    )


def normalize_readiness(raw: list[dict[str, Any]] | dict[str, Any] | None) -> ReadinessSummary | None:
    items: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, list):
        items = [i for i in raw if isinstance(i, dict)]
    if not items:
        return None
    latest = max(items, key=lambda i: str(i.get("timestamp") or ""))
    return ReadinessSummary(
        score=as_int(latest.get("score")),
        level=latest.get("level"),
        feedback_short=latest.get("feedbackShort"),
        feedback_long=latest.get("feedbackLong"),
        sleep_score=as_int(latest.get("sleepScore")),
        recovery_time_min=as_int(latest.get("recoveryTime")),
        hrv_weekly_avg=as_float(latest.get("hrvWeeklyAverage")),
    )


def normalize_device(raw: dict[str, Any] | None) -> DeviceInfo | None:
    r = raw if isinstance(raw, dict) else {}
    if not r:
        return None
    return DeviceInfo(
        name=r.get("lastUsedDeviceName"),
        last_upload=ms_to_utc(r.get("lastUsedDeviceUploadTime")) or parse_garmin_gmt(r.get("lastUsedDeviceUploadTime")),
    )


# ---------------------------------------------------------------------------
# Whole-day
# ---------------------------------------------------------------------------


def normalize_day(
    profile: str,
    day: date,
    tz: str,
    raw: dict[str, Any],
    errors: dict[str, str] | None = None,
    fetched_at: datetime | None = None,
) -> DaySnapshot:
    stress, body_battery = normalize_stress(raw.get(RAW_STRESS))
    activities = normalize_activities(raw.get(RAW_ACTIVITIES))
    auto = normalize_bb_events(raw.get(RAW_BB_EVENTS))
    # Auto-detected windows that overlap a recorded activity are redundant.
    for w in auto:
        if not any(a.start <= w.end and w.start <= a.end for a in activities):
            activities.append(w)
    activities.sort(key=lambda w: w.start)

    snap = DaySnapshot(
        profile=profile,
        day=day,
        tz=tz,
        fetched_at=fetched_at or now_utc(),
        summary=normalize_summary(raw.get(RAW_SUMMARY)),
        hr=normalize_heart_rates(raw.get(RAW_HEART_RATES)),
        steps=normalize_steps(raw.get(RAW_STEPS)),
        stress=stress,
        body_battery=body_battery,
        activities=activities,
        sleep=normalize_sleep(raw.get(RAW_SLEEP)),
        hrv=normalize_hrv(raw.get(RAW_HRV)),
        readiness=normalize_readiness(raw.get(RAW_READINESS)),
        spo2=normalize_spo2(raw.get(RAW_SPO2)),
        respiration=normalize_respiration(raw.get(RAW_RESPIRATION)),
        device=normalize_device(raw.get(RAW_DEVICE)),
        raw=raw,
        errors=dict(errors or {}),
    )
    # Fill summary gaps from other endpoints when the daily summary lacked them
    s = snap.summary
    if s.resting_hr is None and isinstance(raw.get(RAW_HEART_RATES), dict):
        s.resting_hr = as_int(raw[RAW_HEART_RATES].get("restingHeartRate"))
    if s.resting_hr_7d_avg is None and isinstance(raw.get(RAW_HEART_RATES), dict):
        s.resting_hr_7d_avg = as_int(raw[RAW_HEART_RATES].get("lastSevenDaysAvgRestingHeartRate"))
    if s.max_hr is None and snap.hr:
        s.max_hr = max(x.hr for x in snap.hr)
    if s.min_hr is None and snap.hr:
        s.min_hr = min(x.hr for x in snap.hr)
    if snap.sleep and snap.sleep.resting_hr is None and s.resting_hr is not None:
        snap.sleep.resting_hr = s.resting_hr
    return snap
