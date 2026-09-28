"""Shared test fixtures: a synthetic Garmin day generator that mimics the raw payload shapes."""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from garmin_health_monitor.config import (
    AppConfig,
    FeatureFlags,
    GarminConfig,
    OllamaConfig,
    PalpitationConfig,
    ProfileConfig,
    ScheduleConfig,
    TelegramConfig,
)
from garmin_health_monitor.normalize import (
    RAW_ACTIVITIES,
    RAW_BB_EVENTS,
    RAW_DEVICE,
    RAW_HEART_RATES,
    RAW_HRV,
    RAW_READINESS,
    RAW_RESPIRATION,
    RAW_SLEEP,
    RAW_SPO2,
    RAW_STEPS,
    RAW_STRESS,
    RAW_SUMMARY,
    normalize_day,
)
from garmin_health_monitor.storage import Storage

TZ = "Asia/Singapore"
DAY = date(2026, 9, 27)


def _gmt_str(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.0")


def _ms(dt: datetime) -> int:
    return int(dt.astimezone(UTC).timestamp() * 1000)


def make_raw_day(
    day: date = DAY,
    tz: str = TZ,
    rhr: int = 58,
    episodes: list[tuple[int, int, int, int]] | None = None,
    activities: list[tuple[int, int, int]] | None = None,
    sleep: tuple[float, float] | None = (22.5, 6.5),
    steps_per_hour: int = 300,
    seed: int = 1,
    with_nulls: bool = True,
) -> dict:
    """Build a raw Garmin-like payload dict for one local day.

    episodes:   list of (hour, minute, duration_minutes, peak_hr) at-rest HR excursions
    activities: list of (hour, duration_minutes, avg_hr) recorded walks/runs
    sleep:      (bedtime_hour_prev_day, wake_hour) or None
    """
    rng = random.Random(seed)
    zone = ZoneInfo(tz)
    episodes = episodes or []
    activities = activities or []
    day_start = datetime.combine(day, time.min, tzinfo=zone)
    day_end = day_start + timedelta(days=1)

    sleep_start = sleep_end = None
    if sleep:
        bed_h, wake_h = sleep
        sleep_start = day_start - timedelta(hours=24 - bed_h)
        sleep_end = day_start + timedelta(hours=wake_h)

    act_windows = []
    for h, dur, avg in activities:
        s = day_start + timedelta(hours=h)
        act_windows.append((s, s + timedelta(minutes=dur), avg))

    ep_windows = []
    for h, m, dur, peak in episodes:
        s = day_start + timedelta(hours=h, minutes=m)
        ep_windows.append((s, s + timedelta(minutes=dur), peak))

    # -- heart rate, 2-minute samples --------------------------------------
    hr_values = []
    t = day_start
    while t < day_end:
        asleep = sleep_start is not None and sleep_start <= t < sleep_end
        hr = rhr - 6 if asleep else rhr + 8
        hr += rng.randint(-3, 3)
        for s, e, avg in act_windows:
            if s <= t < e:
                hr = avg + rng.randint(-5, 5)
            elif e <= t < e + timedelta(minutes=8):
                hr = avg - 20 + rng.randint(-3, 3)
        for s, e, peak in ep_windows:
            if s <= t < e:
                hr = peak + rng.randint(-2, 2)
        if with_nulls and rng.random() < 0.02:
            hr_values.append([_ms(t), None])
        else:
            hr_values.append([_ms(t), int(hr)])
        t += timedelta(minutes=2)

    # -- steps, 15-minute buckets -------------------------------------------
    steps = []
    t = day_start
    while t < day_end:
        end = t + timedelta(minutes=15)
        asleep = sleep_start is not None and sleep_start <= t < sleep_end
        in_act = any(s <= t < e for s, e, _ in act_windows)
        if asleep:
            level, n = "sleeping", 0
        elif in_act:
            level, n = "active", 400 + rng.randint(0, 200)
        else:
            level, n = "sedentary", rng.randint(0, max(1, steps_per_hour // 4))
        steps.append(
            {
                "startGMT": _gmt_str(t),
                "endGMT": _gmt_str(end),
                "steps": n,
                "pushes": 0,
                "primaryActivityLevel": level,
                "activityLevelConstant": True,
            }
        )
        t = end

    # -- stress / body battery, 3-minute samples ----------------------------
    stress_values = []
    bb_values = []
    t = day_start
    level_bb = 80
    while t < day_end:
        asleep = sleep_start is not None and sleep_start <= t < sleep_end
        in_act = any(s <= t < e for s, e, _ in act_windows)
        if in_act:
            val = -2
        elif asleep:
            val = rng.randint(5, 20)
            level_bb = min(100, level_bb + 1)
        else:
            val = rng.randint(15, 45)
            level_bb = max(5, level_bb - (1 if rng.random() < 0.5 else 0))
        stress_values.append([_ms(t), val])
        bb_values.append([_ms(t), "MEASURED", level_bb, 1.0])
        t += timedelta(minutes=3)

    total_steps = sum(b["steps"] for b in steps)
    raw = {
        RAW_SUMMARY: {
            "calendarDate": day.isoformat(),
            "totalSteps": total_steps,
            "dailyStepGoal": 6000,
            "totalDistanceMeters": total_steps * 0.7,
            "totalKilocalories": 1900.0,
            "activeKilocalories": 350.0,
            "bmrKilocalories": 1550.0,
            "minHeartRate": rhr - 8,
            "maxHeartRate": max(v for _, v in hr_values if v),
            "restingHeartRate": rhr,
            "lastSevenDaysAvgRestingHeartRate": rhr - 1,
            "sleepingSeconds": int((sleep_end - sleep_start).total_seconds()) if sleep else 0,
            "sedentarySeconds": 40000,
            "activeSeconds": 4000,
            "highlyActiveSeconds": 1200,
            "moderateIntensityMinutes": 25,
            "vigorousIntensityMinutes": 5,
            "intensityMinutesGoal": 150,
            "floorsAscended": 6.0,
            "averageStressLevel": 28,
            "maxStressLevel": 80,
            "stressQualifier": "BALANCED",
            "restStressDuration": 30000,
            "lowStressDuration": 20000,
            "mediumStressDuration": 6000,
            "highStressDuration": 1200,
            "bodyBatteryChargedValue": 60,
            "bodyBatteryDrainedValue": 55,
            "bodyBatteryHighestValue": 85,
            "bodyBatteryLowestValue": 25,
            "bodyBatteryMostRecentValue": 40,
            "abnormalHeartRateAlertsCount": len(episodes),
            "averageSpo2": 96.0,
            "lowestSpo2": 92,
            "avgWakingRespirationValue": 15.0,
            "lastSyncTimestampGMT": _gmt_str(day_end - timedelta(minutes=30)),
            "wellnessEndTimeGmt": _gmt_str(day_end),
        },
        RAW_HEART_RATES: {
            "calendarDate": day.isoformat(),
            "restingHeartRate": rhr,
            "maxHeartRate": max(v for _, v in hr_values if v),
            "minHeartRate": rhr - 8,
            "lastSevenDaysAvgRestingHeartRate": rhr - 1,
            "heartRateValueDescriptors": [
                {"key": "timestamp", "index": 0},
                {"key": "heartrate", "index": 1},
            ],
            "heartRateValues": hr_values,
        },
        RAW_STEPS: steps,
        RAW_STRESS: {
            "calendarDate": day.isoformat(),
            "maxStressLevel": 80,
            "avgStressLevel": 28,
            "stressValueDescriptorsDTOList": [
                {"key": "timestamp", "index": 0},
                {"key": "stressLevel", "index": 1},
            ],
            "stressValuesArray": stress_values,
            "bodyBatteryValueDescriptorsDTOList": [
                {"bodyBatteryValueDescriptorIndex": 0, "bodyBatteryValueDescriptorKey": "timestamp"},
                {"bodyBatteryValueDescriptorIndex": 1, "bodyBatteryValueDescriptorKey": "bodyBatteryStatus"},
                {"bodyBatteryValueDescriptorIndex": 2, "bodyBatteryValueDescriptorKey": "bodyBatteryLevel"},
                {"bodyBatteryValueDescriptorIndex": 3, "bodyBatteryValueDescriptorKey": "version"},
            ],
            "bodyBatteryValuesArray": bb_values,
        },
        RAW_ACTIVITIES: [
            {
                "activityId": 1000 + i,
                "activityName": "Morning Walk" if avg < 120 else "Run",
                "startTimeGMT": (day_start + timedelta(hours=h)).astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "startTimeLocal": (day_start + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"),
                "activityType": {"typeKey": "walking" if avg < 120 else "running"},
                "duration": dur * 60.0,
                "elapsedDuration": dur * 60.0,
                "distance": dur * 80.0,
                "averageHR": avg,
                "maxHR": avg + 15,
                "calories": dur * 5.0,
            }
            for i, (h, dur, avg) in enumerate(activities)
        ],
        RAW_BB_EVENTS: [],
        RAW_RESPIRATION: {
            "calendarDate": day.isoformat(),
            "avgWakingRespirationValue": 15.0,
            "avgSleepRespirationValue": 13.0,
            "lowestRespirationValue": 10.0,
            "highestRespirationValue": 22.0,
            "respirationValueDescriptorsDTOList": [
                {"key": "timestamp", "index": 0},
                {"key": "respiration", "index": 1},
            ],
            "respirationValuesArray": [[_ms(day_start + timedelta(hours=h)), 14.0 + (h % 3)] for h in range(24)],
        },
        RAW_SPO2: {
            "calendarDate": day.isoformat(),
            "averageSpO2": 96.0,
            "lowestSpO2": 92,
            "latestSpO2": 97,
            "avgSleepSpO2": 95.0,
            "lastSevenDaysAvgSpO2": 96.2,
            "spO2HourlyAverages": [[_ms(day_start + timedelta(hours=h)), 95 + (h % 3)] for h in range(0, 7)],
        },
        RAW_READINESS: [
            {
                "calendarDate": day.isoformat(),
                "timestamp": _gmt_str(day_start + timedelta(hours=7)),
                "level": "MODERATE",
                "score": 62,
                "feedbackShort": "MODERATE",
                "feedbackLong": "You are moderately ready.",
                "sleepScore": 78,
                "recoveryTime": 600,
                "hrvWeeklyAverage": 44.0,
                "inputContext": "AFTER_WAKEUP_RESET",
            }
        ],
        RAW_DEVICE: {
            "lastUsedDeviceName": "Venu 3",
            "lastUsedDeviceUploadTime": _ms(day_end - timedelta(minutes=30)),
        },
    }
    if sleep:
        sleep_hr = []
        t = sleep_start
        while t < sleep_end:
            sleep_hr.append({"value": rhr - 6 + rng.randint(-3, 3), "startGMT": _ms(t)})
            t += timedelta(minutes=5)
        total = int((sleep_end - sleep_start).total_seconds())
        raw[RAW_SLEEP] = {
            "dailySleepDTO": {
                "calendarDate": day.isoformat(),
                "sleepTimeSeconds": total - 1200,
                "napTimeSeconds": 0,
                "deepSleepSeconds": int(total * 0.2),
                "lightSleepSeconds": int(total * 0.5),
                "remSleepSeconds": int(total * 0.22),
                "awakeSleepSeconds": 1200,
                "sleepStartTimestampGMT": _ms(sleep_start),
                "sleepEndTimestampGMT": _ms(sleep_end),
                "sleepStartTimestampLocal": _ms(sleep_start) + 8 * 3600 * 1000,
                "sleepEndTimestampLocal": _ms(sleep_end) + 8 * 3600 * 1000,
                "averageSpO2Value": 95.0,
                "lowestSpO2Value": 91,
                "averageRespirationValue": 13.0,
                "avgSleepHRV": 42.0,
                "sleepScores": {
                    "overall": {"value": 78, "qualifierKey": "GOOD"},
                    "awakeCount": {"value": 2, "qualifierKey": "GOOD"},
                    "restlessness": {"value": 70, "qualifierKey": "FAIR"},
                },
            },
            "sleepHeartRate": sleep_hr,
            "restingHeartRate": rhr - 2,
            "restlessMomentsCount": 18,
            "avgOvernightHrv": 42.0,
            "hrvStatus": "BALANCED",
            "bodyBatteryChange": 55,
        }
        raw[RAW_HRV] = {
            "hrvSummary": {
                "calendarDate": day.isoformat(),
                "weeklyAvg": 44.0,
                "lastNightAvg": 42.0,
                "lastNight5MinHigh": 61.0,
                "status": "BALANCED",
                "feedbackPhrase": "HRV_BALANCED_5",
                "baseline": {"lowUpper": 35.0, "balancedLow": 38.0, "balancedUpper": 52.0, "markerValue": 0.4},
            },
            "hrvReadings": [
                {"hrvValue": 40 + (i % 7), "readingTimeGMT": _gmt_str(sleep_start + timedelta(minutes=5 * i))}
                for i in range(0, int((sleep_end - sleep_start).total_seconds() // 300))
            ],
            "startTimestampGMT": _gmt_str(sleep_start),
            "endTimestampGMT": _gmt_str(sleep_end),
        }
    return raw


def make_snapshot(profile: str = "Dad", **kwargs):
    day = kwargs.pop("day", DAY)
    tz = kwargs.pop("tz", TZ)
    raw = make_raw_day(day=day, tz=tz, **kwargs)
    return normalize_day(profile, day, tz, raw)


@pytest.fixture
def quiet_day():
    return make_snapshot(profile="Dad")


@pytest.fixture
def episode_day():
    # two at-rest episodes: 10:00 for 8 min peaking 125, 15:30 for 6 min peaking 112; a walk at 8:00
    return make_snapshot(
        profile="Dad",
        episodes=[(10, 0, 8, 125), (15, 30, 6, 112)],
        activities=[(8, 30, 105)],
    )


@pytest.fixture
def storage(tmp_path):
    st = Storage(tmp_path / "test.db")
    yield st
    st.close()


def make_profile(name: str = "Dad", palpitations: bool = True, chat_ids: list[int] | None = None) -> ProfileConfig:
    return ProfileConfig(
        name=name,
        garmin=GarminConfig(email="x@example.com", password="pw", tokenstore=f"/tmp/tokens/{name.lower()}"),
        telegram_chat_ids=chat_ids if chat_ids is not None else [111],
        timezone=TZ,
        persona="68-year-old, occasional palpitations",
        goals="6000 steps, 7h sleep",
        features=FeatureFlags(palpitations=palpitations, doctor_report=palpitations),
        palpitations=PalpitationConfig(),
    )


def make_app_config(tmp_path, profiles: list[ProfileConfig] | None = None) -> AppConfig:
    return AppConfig(
        timezone=TZ,
        database=str(tmp_path / "monitor.db"),
        data_dir=str(tmp_path),
        telegram=TelegramConfig(bot_token="123:abc", admin_chat_ids=[999]),
        ollama=OllamaConfig(enabled=True, base_url="http://ollama.test:11434", model="test-model"),
        schedule=ScheduleConfig(),
        profiles=profiles or [make_profile()],
    )


@pytest.fixture
def profile():
    return make_profile()


@pytest.fixture
def app_config(tmp_path):
    return make_app_config(tmp_path)
