from datetime import timedelta

from garmin_health_monitor.alerts import evaluate_alerts
from garmin_health_monitor.models import BodyBatterySample, HrvSummary
from garmin_health_monitor.utils import get_tz, now_utc
from tests.conftest import make_profile, make_snapshot


def _now_local(snap, hour):
    from datetime import datetime, time

    return datetime.combine(snap.day, time(hour=hour), tzinfo=get_tz(snap.tz)).astimezone(get_tz("UTC"))


def test_quiet_day_no_alerts():
    p = make_profile()
    snap = make_snapshot()
    now = snap.summary.last_sync + timedelta(minutes=10)
    alerts = evaluate_alerts(p, snap, now=now)
    assert [a.key for a in alerts] == []


def test_rhr_and_spo2_and_hrv_alerts():
    p = make_profile()
    snap = make_snapshot()
    snap.summary.resting_hr = 72
    snap.summary.resting_hr_7d_avg = 58
    snap.spo2.lowest = 86
    snap.hrv = HrvSummary(status="LOW", last_night_avg=28.0, weekly_avg=44.0)
    now = snap.summary.last_sync + timedelta(minutes=10)
    keys = {a.key.split(":")[0]: a for a in evaluate_alerts(p, snap, now=now)}
    assert "rhr_high" in keys and "14" in keys["rhr_high"].body
    assert "spo2_low" in keys
    assert "hrv_low" in keys
    assert all(a.profile == "Dad" for a in keys.values())


def test_absolute_rhr_threshold_without_average():
    p = make_profile()
    snap = make_snapshot()
    snap.summary.resting_hr = 95
    snap.summary.resting_hr_7d_avg = None
    now = snap.summary.last_sync + timedelta(minutes=10)
    assert any(a.key.startswith("rhr_high") for a in evaluate_alerts(p, snap, now=now))


def test_no_sync_alert():
    p = make_profile()
    snap = make_snapshot()
    snap.device = None
    now = snap.summary.last_sync + timedelta(hours=p.alerts.no_sync_hours + 1)
    alerts = evaluate_alerts(p, snap, now=now)
    assert any(a.key.startswith("no_sync") for a in alerts)
    # a stale watch must not also trigger the sedentary nudge (no fresh data)
    assert not any(a.key.startswith("sedentary") for a in alerts)


def test_sedentary_nudge_and_body_battery():
    p = make_profile()
    snap = make_snapshot(steps_per_hour=0)
    snap.summary.total_steps = 300
    now = _now_local(snap, 16)
    snap.summary.last_sync = now - timedelta(minutes=5)
    snap.device.last_upload = now - timedelta(minutes=5)
    snap.body_battery = [BodyBatterySample(ts=now - timedelta(minutes=3), level=8)]
    keys = {a.key.split(":")[0] for a in evaluate_alerts(p, snap, now=now)}
    assert "sedentary" in keys
    assert "body_battery_low" in keys


def test_abnormal_hr_alert_only_when_count_rises():
    p = make_profile()
    snap = make_snapshot(episodes=[(10, 0, 8, 125)])
    now = snap.summary.last_sync + timedelta(minutes=10)
    assert snap.summary.abnormal_hr_alerts == 1
    a = [x for x in evaluate_alerts(p, snap, now=now, previous_abnormal_count=None) if x.key.startswith("garmin_abnormal_hr")]
    assert len(a) == 1 and a[0].key.endswith(":1")
    assert not [x for x in evaluate_alerts(p, snap, now=now, previous_abnormal_count=1) if x.key.startswith("garmin_abnormal_hr")]


def test_keys_are_per_day_for_dedupe():
    p = make_profile()
    snap = make_snapshot()
    snap.summary.resting_hr = 95
    now = now_utc()
    snap.summary.last_sync = now
    keys = [a.key for a in evaluate_alerts(p, snap, now=now)]
    assert f"rhr_high:Dad:{snap.date_str}" in keys


def _with_low_hr(snap, hour, minutes, hr=36):
    from datetime import datetime, time

    from garmin_health_monitor.models import HRSample

    start = datetime.combine(snap.day, time(hour=hour), tzinfo=get_tz(snap.tz))
    end = start + timedelta(minutes=minutes)
    snap.hr = [HRSample(ts=s.ts, hr=hr) if start <= s.ts < end else s for s in snap.hr]
    return snap


def test_low_hr_while_awake_alerts_once_with_time_and_lowest():
    p = make_profile()
    p.alerts.low_hr_below, p.alerts.low_hr_minutes = 40, 10
    snap = _with_low_hr(make_snapshot(), 14, 14)
    alerts = [a for a in evaluate_alerts(p, snap, now=snap.summary.last_sync) if a.key.startswith("low_hr")]
    assert len(alerts) == 1
    assert "Under 40 bpm for" in alerts[0].body and "from 14:00" in alerts[0].body and "lowest 36" in alerts[0].body


def test_low_hr_short_dip_or_off_by_default_is_silent():
    p = make_profile()
    snap = _with_low_hr(make_snapshot(), 14, 14)
    assert not [a for a in evaluate_alerts(p, snap, now=snap.summary.last_sync) if a.key.startswith("low_hr")]
    p.alerts.low_hr_below = 40
    snap = _with_low_hr(make_snapshot(), 14, 4)  # 4 minutes is too short
    assert not [a for a in evaluate_alerts(p, snap, now=snap.summary.last_sync) if a.key.startswith("low_hr")]


def test_only_keeps_listed_rules_and_no_sync_after_6h():
    p = make_profile()
    p.alerts.only, p.alerts.no_sync_hours = ("no_sync", "low_hr"), 6
    snap = make_snapshot()
    snap.summary.resting_hr, snap.summary.resting_hr_7d_avg = 72, 58  # would be rhr_high, filtered out
    keys = [a.key.split(":")[0] for a in evaluate_alerts(p, snap, now=snap.summary.last_sync + timedelta(hours=7))]
    assert keys == ["no_sync"]
