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
from .utils import get_tz, parse_hhmm

if TYPE_CHECKING:
    from .telegram_bot import HealthBot

logger = logging.getLogger(__name__)

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

    # ------------------------------------------------------------ registration

    def register(self) -> None:
        jq = self.bot.app.job_queue
        if jq is None:  # pragma: no cover - job-queue extra missing
            raise RuntimeError("python-telegram-bot[job-queue] is required for scheduling")
        sched = self.config.schedule
        jq.run_once(self.job_startup, when=timedelta(seconds=5), name="startup")
        jq.run_repeating(self.job_poll, interval=timedelta(minutes=sched.poll_minutes), first=timedelta(seconds=60), name="poll")
        for p in self.config.profiles:
            tz = p.timezone or self.config.timezone
            if p.features.morning_brief:
                jq.run_daily(self.job_morning, time=_at(sched.morning_brief, tz), data=p, name=f"morning:{p.slug}")
            if p.features.evening_summary:
                jq.run_daily(self.job_evening, time=_at(sched.evening_summary, tz), data=p, name=f"evening:{p.slug}")
            if p.features.weekly_review:
                day = _WEEKDAYS.get(str(sched.weekly_review_day).lower()[:3], 0)
                jq.run_daily(self.job_weekly, time=_at(sched.weekly_review_time, tz), days=(day,), data=p, name=f"weekly:{p.slug}")
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
                await self._notify_admins(f"⚠️ {messages.esc(p.name)}: startup backfill failed: {messages.esc(str(exc))}")
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
                await self._notify_admins_once(f"poll_error:{p.slug}:{datetime.now():%Y-%m-%d}", f"⚠️ {messages.esc(p.name)}: poll failed: {messages.esc(str(exc))}")
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

    async def job_doctor_report(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        p: ProfileConfig = context.job.data  # type: ignore[union-attr]
        try:
            files = await asyncio.to_thread(self.service.doctor_report, p, self.config.schedule.doctor_report_days)
            caption = messages.doctor_report_caption(p, self.config.schedule.doctor_report_days, files.n_episodes, files.n_symptoms)
            await self.bot.send_document(p.telegram_chat_ids, files.pdf_path, caption)
            await self.bot.send_document(p.telegram_chat_ids, files.csv_path, "CSV export of the same diary")
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: doctor report job failed", p.name)
            await self._notify_admins(f"⚠️ {messages.esc(p.name)}: doctor report failed: {messages.esc(str(exc))}")

    # ---------------------------------------------------------------- helpers

    async def _send_report_job(self, p: ProfileConfig, fn, label: str) -> None:
        try:
            text = await asyncio.to_thread(fn, p)
            await self.bot.send_text(p.telegram_chat_ids, text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: %s failed", p.name, label)
            await self._notify_admins(f"⚠️ {messages.esc(p.name)}: {label} failed: {messages.esc(str(exc))}")

    async def deliver(self, p: ProfileConfig, result: PollResult) -> None:
        """Send episode alerts and rule alerts from a poll result (respecting quiet hours)."""
        if result.error:
            await self._notify_admins_once(f"poll_error:{p.slug}:{datetime.now():%Y-%m-%d}", f"⚠️ {messages.esc(p.name)}: {messages.esc(result.error)}")
            return
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

    async def notify_alert(self, p: ProfileConfig, alert: Alert) -> None:
        if self.service.storage.notification_sent(alert.key):
            return
        await self.bot.send_text(p.telegram_chat_ids, messages.alert_message(alert))
        self.service.storage.mark_notification(alert.key, p.name)

    @staticmethod
    def _is_critical_episode(p: ProfileConfig, ep: Episode) -> bool:
        return ep.peak_hr >= max(140, p.palpitations.spike_hr_threshold + 20) and ep.duration_min >= 10

    async def _notify_admins(self, text: str) -> None:
        ids = self.config.telegram.admin_chat_ids
        if ids:
            await self.bot.send_text(ids, text)

    async def _notify_admins_once(self, key: str, text: str) -> None:
        if self.service.storage.notification_sent(key):
            return
        await self._notify_admins(text)
        self.service.storage.mark_notification(key)
