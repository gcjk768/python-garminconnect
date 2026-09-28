"""Rule-based, event-driven alerts (no model involved).

Each rule yields an :class:`Alert` with a dedupe ``key`` so the scheduler
sends it at most once (``storage.notification_sent(key)``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .config import ProfileConfig
from .models import Alert, DaySnapshot
from .utils import fmt_hm, now_utc, to_local

logger = logging.getLogger(__name__)


def _key(kind: str, profile: str, day: str, extra: str = "") -> str:
    return f"{kind}:{profile}:{day}" + (f":{extra}" if extra else "")


def evaluate_alerts(
    profile: ProfileConfig,
    snap: DaySnapshot,
    now: datetime | None = None,
    previous_abnormal_count: int | None = None,
) -> list[Alert]:
    """Evaluate the alert rules for one snapshot of *today*."""
    cfg = profile.alerts
    now = now or now_utc()
    local_now = to_local(now, profile.timezone)
    day = snap.date_str
    s = snap.summary
    out: list[Alert] = []

    # Resting heart rate above the person's own baseline
    if s.resting_hr is not None:
        if s.resting_hr_7d_avg and s.resting_hr >= s.resting_hr_7d_avg + cfg.rhr_above_7d_avg:
            out.append(
                Alert(
                    key=_key("rhr_high", profile.name, day),
                    severity="warning",
                    title="Resting heart rate is higher than usual",
                    body=(
                        f"Resting HR today is {s.resting_hr} bpm, {s.resting_hr - s.resting_hr_7d_avg} above the "
                        f"7-day average of {s.resting_hr_7d_avg}. Poor sleep, illness, dehydration or stress are common causes."
                    ),
                    profile=profile.name,
                )
            )
        elif s.resting_hr >= cfg.rhr_absolute_high:
            out.append(
                Alert(
                    key=_key("rhr_high", profile.name, day),
                    severity="warning",
                    title="Resting heart rate is high",
                    body=f"Resting HR today is {s.resting_hr} bpm (alert threshold {cfg.rhr_absolute_high}).",
                    profile=profile.name,
                )
            )

    # Low blood oxygen overnight
    lowest_spo2 = snap.spo2.lowest if snap.spo2 and snap.spo2.lowest is not None else s.lowest_spo2
    if lowest_spo2 is not None and 0 < lowest_spo2 < cfg.spo2_below:
        out.append(
            Alert(
                key=_key("spo2_low", profile.name, day),
                severity="warning",
                title="Low blood oxygen reading",
                body=f"Lowest SpO2 was {lowest_spo2}% (alert below {cfg.spo2_below}%). Single dips can be sensor noise; repeated nights are worth mentioning to a doctor.",
                profile=profile.name,
            )
        )

    # HRV status
    if snap.hrv and snap.hrv.status and snap.hrv.status.upper() in {x.upper() for x in cfg.hrv_alert_statuses}:
        val = f"{snap.hrv.last_night_avg:.0f} ms" if snap.hrv.last_night_avg else "n/a"
        out.append(
            Alert(
                key=_key("hrv_low", profile.name, day),
                severity="info",
                title=f"Overnight HRV status: {snap.hrv.status.title()}",
                body=f"Last night's HRV averaged {val} (weekly avg {snap.hrv.weekly_avg or 'n/a'}). Take it easy today and prioritise sleep.",
                profile=profile.name,
            )
        )

    # Body battery critically low while awake
    latest_bb = s.body_battery_latest
    if snap.body_battery:
        latest_bb = snap.body_battery[-1].level
    if latest_bb is not None and latest_bb <= cfg.body_battery_below and 9 <= local_now.hour <= 22:
        out.append(
            Alert(
                key=_key("body_battery_low", profile.name, day),
                severity="info",
                title="Body battery is very low",
                body=f"Body battery is at {latest_bb}. A rest, a nap or an early night would help.",
                profile=profile.name,
            )
        )

    # Watch not syncing (forgot to wear / charge / phone off)
    last_sync = s.last_sync
    if snap.device and snap.device.last_upload and (last_sync is None or snap.device.last_upload > last_sync):
        last_sync = snap.device.last_upload
    if last_sync is not None:
        age = now - last_sync
        if age > timedelta(hours=cfg.no_sync_hours):
            hours = int(age.total_seconds() // 3600)
            out.append(
                Alert(
                    key=_key("no_sync", profile.name, day),
                    severity="warning",
                    title="Watch has not synced",
                    body=(
                        f"Last sync was {hours}h ago ({fmt_hm(last_sync, profile.timezone)}). "
                        "Check the watch is worn, charged and the Garmin Connect app is open on the phone."
                    ),
                    profile=profile.name,
                )
            )

    # Sedentary nudge in the afternoon
    if (
        s.total_steps is not None
        and local_now.hour >= cfg.sedentary_nudge_hour
        and s.total_steps < cfg.sedentary_nudge_steps
        and last_sync is not None
        and (now - last_sync) <= timedelta(hours=3)
    ):
        out.append(
            Alert(
                key=_key("sedentary", profile.name, day),
                severity="info",
                title="Quiet day so far",
                body=f"Only {s.total_steps} steps by {local_now.strftime('%H:%M')}. A short walk now would do a lot of good.",
                profile=profile.name,
            )
        )

    # Garmin's own abnormal heart-rate alerts (set on the watch) increased
    if (
        cfg.abnormal_hr_alerts
        and s.abnormal_hr_alerts is not None
        and s.abnormal_hr_alerts > 0
        and s.abnormal_hr_alerts > (previous_abnormal_count or 0)
    ):
        out.append(
            Alert(
                key=_key("garmin_abnormal_hr", profile.name, day, str(s.abnormal_hr_alerts)),
                severity="warning",
                title="Garmin abnormal heart-rate alert",
                body=(
                    f"The watch has raised {s.abnormal_hr_alerts} abnormal heart-rate alert(s) today "
                    "(heart rate above the limit set on the watch while inactive). It has been added to the diary."
                ),
                profile=profile.name,
            )
        )

    return out
