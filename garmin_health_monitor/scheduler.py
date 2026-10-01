"""Scheduled jobs on top of python-telegram-bot's JobQueue.

The bot's event loop owns everything; blocking service calls are pushed to a
worker thread with ``asyncio.to_thread`` so polling Telegram never stalls.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING

from telegram.ext import ContextTypes

from . import messages
from .config import AppConfig, ProfileConfig
from .models import Alert, Episode
from .service import MonitorService, PollResult
from .utils import get_tz, parse_hhmm, to_local

if TYPE_CHECKING:
    from .telegram_bot import HealthBot

logger = logging.getLogger(__name__)

OUTAGE_ALERT_AFTER = timedelta(hours=2)  # Garmin blips (e.g. HTTP 521) usually clear well within this

_WEEKDAYS = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}  # PTB: 0 = Sunday


def _at(hhmm: str, tz: str | None) -> time:
    t = parse_hhmm(hhmm)
    return time(hour=t.hour, minute=t.minute, tzinfo=get_tz(tz))


class Scheduler:
    def __init__(self, config: AppConfig, service: MonitorService, bot: HealthBot):
        self.config = config
        self.service = service
        self.bot = bot
        self._busy: set[str] = set()
        self._down_since: dict[str, datetime] = {}  # profile -> first failed poll of the current outage
        self._down_alerted: set[str] = set()

    # ------------------------------------------------------------ registration

    def register(self) -> None:
        jq = self.bot.app.job_queue
        if jq is None:  # pragma: no cover - job-queue extra missing
            raise RuntimeError("python-telegram-bot[job-queue] is required for scheduling")
        sched = self.config.schedule
        jq.run_once(self.job_startup, when=timedelta(seconds=5), name="startup")
        jq.run_repeating(self.job_poll, interval=timedelta(minutes=sched.poll_minutes), first=timedelta(seconds=60), name="poll")
        if self.config.backup_dir:
            jq.run_daily(self.job_backup, time=_at("03:30", self.config.timezone), name="backup")
        for p in self.config.profiles:
            tz = p.timezone or self.config.timezone
            if p.features.morning_brief:
                jq.run_daily(self.job_morning, time=_at(sched.morning_brief, tz), data=p, name=f"morning:{p.slug}")
            if p.features.evening_summary:
                jq.run_daily(self.job_evening, time=_at(sched.evening_summary, tz), data=p, name=f"evening:{p.slug}")
            if p.features.weekly_review:
                day = _WEEKDAYS.get(str(sched.weekly_review_day).lower()[:3], 0)
                jq.run_daily(self.job_weekly, time=_at(sched.weekly_review_time, tz), days=(day,), data=p, name=f"weekly:{p.slug}")
            if p.features.monthly_summary:
                jq.run_monthly(self.job_monthly, when=_at(sched.monthly_summary_time, tz), day=int(sched.monthly_summary_day), data=p, name=f"monthly:{p.slug}")
            if p.features.workout_nudge:
                nudge_day = _WEEKDAYS.get(str(sched.workout_nudge_day).lower()[:3], 4)
                jq.run_daily(self.job_workout_nudge, time=_at(sched.workout_nudge_time, tz), days=(nudge_day,), data=p, name=f"nudge:{p.slug}")
            for hhmm in p.medication.times:
                jq.run_daily(self.job_medication, time=_at(hhmm, tz), data=(p, hhmm), name=f"med:{p.slug}:{hhmm}")
            if p.features.doctor_report and p.features.palpitations:
                jq.run_monthly(self.job_doctor_report, when=_at(sched.doctor_report_time, tz), day=int(sched.doctor_report_day_of_month), data=p, name=f"report:{p.slug}")
        logger.info("Scheduled jobs: %s", ", ".join(j.name or "?" for j in jq.jobs()))

    # ------------------------------------------------------------------- jobs

    async def job_startup(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.bot.loop = asyncio.get_running_loop()
        for p in self.config.profiles:
            key = f"backfilled:{p.slug}"
            if self.service.storage.kv_get(key):
                continue
            days = int(self.config.schedule.backfill_days)
            try:
                n = await asyncio.to_thread(self.service.backfill, p, days)
                self.service.storage.kv_set(key, datetime.now().isoformat())
                logger.info("%s: backfilled %d day(s)", p.name, n)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: backfill failed: %s", p.name, exc)
                await self._notify_admins(messages.problem("Startup backfill failed", p.name, exc))
        await self.job_poll(context)

    async def job_poll(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        for p in self.config.profiles:
            if p.name in self._busy:
                logger.info("%s: previous poll still running, skipping", p.name)
                continue
            self._busy.add(p.name)
            try:
                result: PollResult = await asyncio.to_thread(self.service.poll, p)
                await self.deliver(p, result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s: poll job failed", p.name)
                await self._poll_failed(p, str(exc))
            finally:
                self._busy.discard(p.name)

    async def job_morning(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        p: ProfileConfig = context.job.data  # type: ignore[union-attr]
        await self._send_report_job(p, self.service.morning_brief_text, "morning brief")

    async def job_evening(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        p: ProfileConfig = context.job.data  # type: ignore[union-attr]
        await self._send_report_job(p, self.service.evening_summary_text, "evening summary")

    async def job_weekly(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        p: ProfileConfig = context.job.data  # type: ignore[union-attr]
        await self._send_report_job(p, self.service.weekly_review_text, "weekly review")

    async def job_monthly(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        p: ProfileConfig = context.job.data  # type: ignore[union-attr]
        try:
            png, caption, extra = await asyncio.to_thread(self.service.monthly_summary, p)
            if png:
                await self.bot.send_photo(p.telegram_chat_ids, png, caption, None, p)
            else:
                await self.bot.send_text(p.telegram_chat_ids, caption, None, p)
            if extra:
                await self.bot.send_text(p.telegram_chat_ids, extra, None, p)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: monthly summary failed", p.name)
            await self._notify_admins(messages.problem("Monthly summary failed", p.name, exc))

    async def job_workout_nudge(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        p: ProfileConfig = context.job.data  # type: ignore[union-attr]
        text = await asyncio.to_thread(self.service.workout_nudge_text, p)
        if text:
            await self.bot.send_text(p.telegram_chat_ids, text, None, p)
            self.service.log_event(p, "🏃", "Workout nudge sent")

    async def job_backup(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            path = await asyncio.to_thread(self.service.backup)
            logger.info("Database backed up to %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("backup failed")
            await self._notify_admins(messages.problem("Nightly database backup failed", None, exc))

    async def job_medication(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        p, hhmm = context.job.data  # type: ignore[union-attr]
        await self.bot.send_med_reminder(p, hhmm, self.service.today(p))

    async def job_doctor_report(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        p: ProfileConfig = context.job.data  # type: ignore[union-attr]
        try:
            files = await asyncio.to_thread(self.service.doctor_report, p, self.config.schedule.doctor_report_days)
            caption = messages.doctor_report_caption(p, self.config.schedule.doctor_report_days, files.n_episodes, files.n_symptoms)
            await self.bot.send_document(p.telegram_chat_ids, files.pdf_path, caption)
            await self.bot.send_document(p.telegram_chat_ids, files.csv_path, "📄 CSV export of the same diary")
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: doctor report job failed", p.name)
            await self._notify_admins(messages.problem("Doctor report failed", p.name, exc))

    # ---------------------------------------------------------------- helpers

    async def _send_report_job(self, p: ProfileConfig, fn, label: str) -> None:
        try:
            text = await asyncio.to_thread(fn, p)
            await self.bot.send_text(p.telegram_chat_ids, text)
            self.service.log_event(p, "📨", f"{label.capitalize()} sent")
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: %s failed", p.name, label)
            await self._notify_admins(messages.problem(f"{label.capitalize()} failed", p.name, exc))

    async def deliver(self, p: ProfileConfig, result: PollResult) -> None:
        """Send episode alerts and rule alerts from a poll result (respecting quiet hours)."""
        if result.error:
            await self._poll_failed(p, result.error)
            return
        await self._poll_ok(p)
        quiet = self.service.in_quiet_hours(p)
        for ep in result.to_notify:
            if quiet and not self._is_critical_episode(p, ep):
                logger.info("%s: holding episode %s until quiet hours end", p.name, ep.id)
                continue
            await self.notify_episode(p, ep)
        for alert in result.alerts:
            if quiet and alert.severity != "critical":
                continue
            await self.notify_alert(p, alert)

    async def notify_episode(self, p: ProfileConfig, ep: Episode) -> None:
        text, png = await asyncio.to_thread(self.service.episode_alert_payload, p, ep)
        await self.bot.notify_episode(p, ep, text, png)
        if ep.id is not None:
            self.service.storage.mark_episode_notified(ep.id)
        start = to_local(ep.start, p.timezone or self.config.timezone)
        self.service.log_event(p, "🚨", "Episode alert sent", f"{start:%H:%M}, peak {ep.peak_hr} bpm",
                               f"Episodes/{start.date().isoformat()}")

    async def notify_alert(self, p: ProfileConfig, alert: Alert) -> None:
        if self.service.storage.notification_sent(alert.key):
            return
        await self.bot.send_text(p.telegram_chat_ids, messages.alert_message(alert))
        self.service.storage.mark_notification(alert.key, p.name)
        self.service.log_event(p, "⚠️", "Alert sent", f"{alert.severity}: {alert.title}")

    @staticmethod
    def _is_critical_episode(p: ProfileConfig, ep: Episode) -> bool:
        return ep.peak_hr >= max(140, p.palpitations.spike_hr_threshold + 20) and ep.duration_min >= 10

    async def _poll_failed(self, p: ProfileConfig, error: str) -> None:
        """Alert once, only if Garmin has been failing for OUTAGE_ALERT_AFTER (short blips stay silent)."""
        now = self.service.clock()
        since = self._down_since.setdefault(p.name, now)
        logger.warning("%s: Garmin poll failed (down since %s): %s", p.name, since, error)
        if p.name in self._down_alerted or now - since < OUTAGE_ALERT_AFTER:
            return
        self._down_alerted.add(p.name)
        hours = (now - since).total_seconds() / 3600
        self.service.log_event(p, "🔌", "Garmin down", f"no data for {hours:.0f}h, alerts paused: {error[:120]}")
        tz = p.timezone or self.config.timezone
        if "login" in error.lower():
            why = "🔑 Garmin login is failing. If this lasts, run <code>garmin-monitor login</code> for this profile."
        else:
            why = "🌐 Garmin's servers seem to be down."
        await self._notify_admins(
            f"⚠️ <b>No Garmin data since {since.astimezone(get_tz(tz)):%H:%M}</b> · {messages.esc(p.name)} "
            f"({hours:.0f}h)\n\n{why}\n⏸ <b>Alerts are paused until it is back.</b>"
        )

    async def _poll_ok(self, p: ProfileConfig) -> None:
        """Garmin is reachable again: catch up on earlier days the outage covered, then all-clear."""
        since = self._down_since.pop(p.name, None)
        if since is None:
            return
        # Today's poll re-reads the whole day; an outage past midnight would skip the end of the
        # previous day(s), so re-check those too and alert on anything found.
        day = since.astimezone(get_tz(p.timezone or self.config.timezone)).date()
        while day < self.service.today(p):
            res = await asyncio.to_thread(self.service.poll, p, day, False)
            for ep in [] if res.error else res.to_notify:
                await self.notify_episode(p, ep)
            day += timedelta(days=1)
        if p.name in self._down_alerted:
            self._down_alerted.discard(p.name)
            self.service.log_event(p, "✅", "Garmin back", "missed days re-checked")
            await self._notify_admins(f"✅ <b>Garmin data is back</b> · {messages.esc(p.name)}\n\n▶️ Nothing lost; missed episodes are checked now.")

    async def _notify_admins(self, text: str) -> None:
        ids = self.config.telegram.admin_chat_ids
        if ids:
            await self.bot.send_text(ids, text)
