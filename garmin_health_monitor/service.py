"""MonitorService: the orchestration layer used by the scheduler, the bot and the CLI.

Everything here is synchronous; the Telegram layer wraps calls in
``asyncio.to_thread``.  One lock per profile serialises Garmin access.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from . import analysis, charts, messages, report, vault
from .alerts import evaluate_alerts
from .config import AppConfig, ProfileConfig
from .garmin_client import GarminAuthRequired, GarminSession, GarminUnavailable
from .llm import LLMClient, LLMError, describe_backend
from .models import (
    EPISODE_SOURCE_MANUAL,
    Alert,
    CoachingAdvice,
    DaySnapshot,
    Episode,
    EpisodeAssessment,
    SymptomReport,
)
from .normalize import normalize_day
from .palpitations import detect_episodes, hr_context_at
from .storage import Storage
from .utils import get_tz, local_day_bounds, now_utc, to_local, to_utc

logger = logging.getLogger(__name__)

MfaPromptFactory = Callable[[ProfileConfig], Callable[[], str] | None]


@dataclass(slots=True)
class PollResult:
    profile: str
    snapshot: DaySnapshot | None
    new_episodes: list[Episode] = field(default_factory=list)
    to_notify: list[Episode] = field(default_factory=list)  # episodes needing a Telegram alert
    alerts: list[Alert] = field(default_factory=list)
    error: str | None = None


class MonitorService:
    def __init__(
        self,
        config: AppConfig,
        storage: Storage | None = None,
        llm: LLMClient | None = None,
        mfa_prompt_factory: MfaPromptFactory | None = None,
        session_factory: Callable[[ProfileConfig, Callable[[], str] | None], GarminSession] | None = None,
        clock: Callable[[], datetime] = now_utc,
    ):
        self.config = config
        self.storage = storage or Storage(config.database)
        self.llm = llm
        self._mfa_factory = mfa_prompt_factory
        self._session_factory = session_factory or (lambda p, m: GarminSession(p, m))
        self._sessions: dict[str, GarminSession] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._cache: dict[tuple[str, str, bool], tuple[datetime, DaySnapshot]] = {}
        self.clock = clock
        self.started_at = clock()
        self.reports_dir = Path(config.data_dir) / "reports"

    # ------------------------------------------------------------------ helpers

    def lock(self, profile: ProfileConfig) -> threading.RLock:
        return self._locks.setdefault(profile.name, threading.RLock())

    def session(self, profile: ProfileConfig) -> GarminSession:
        sess = self._sessions.get(profile.name)
        if sess is None:
            prompt = self._mfa_factory(profile) if self._mfa_factory else None
            sess = self._session_factory(profile, prompt)
            self._sessions[profile.name] = sess
        return sess

    def local_now(self, profile: ProfileConfig) -> datetime:
        return to_local(self.clock(), profile.timezone)

    def today(self, profile: ProfileConfig) -> date:
        return self.local_now(profile).date()

    def profile_for_chat(self, chat_id: int, name: str | None = None) -> ProfileConfig:
        visible = self.config.profiles_for_chat(chat_id)
        if name:
            for p in visible:
                if p.name.lower() == name.lower() or p.slug == name.lower():
                    return p
            raise LookupError(f"No profile named {name!r} for this chat")
        if len(visible) == 1:
            return visible[0]
        if not visible:
            raise LookupError("This chat is not authorised for any profile")
        raise LookupError("Several profiles: " + ", ".join(p.name for p in visible))

    def in_quiet_hours(self, profile: ProfileConfig, when: datetime | None = None) -> bool:
        qh = profile.palpitations.quiet_hours
        if not qh:
            return False
        start, end = int(qh[0]), int(qh[1])
        hour = to_local(when or self.clock(), profile.timezone).hour
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    # --------------------------------------------------------------- data access

    def fetch_and_store(self, profile: ProfileConfig, day: date, light: bool = False, max_age_s: int = 0) -> DaySnapshot:
        """Fetch ``day`` from Garmin, normalise and persist it.

        ``max_age_s`` returns a cached snapshot when one was fetched recently
        (keeps ``/today`` snappy and avoids hammering Garmin).
        """
        key = (profile.name, day.isoformat(), light)
        if max_age_s:
            cached = self._cache.get(key) or self._cache.get((profile.name, day.isoformat(), False))
            if cached and (self.clock() - cached[0]).total_seconds() <= max_age_s:
                return cached[1]
        with self.lock(profile):
            raw, errors = self.session(profile).fetch_day(day, light=light)
            if light:
                # keep the slow-changing endpoints from the last full fetch of this day
                previous = self.storage.get_snapshot_raw(profile.name, day) or {}
                for k, v in previous.items():
                    raw.setdefault(k, v)
            snap = normalize_day(profile.name, day, profile.timezone or self.config.timezone, raw, errors, fetched_at=self.clock())
            self.storage.save_snapshot(snap)
            self._cache[key] = (self.clock(), snap)
            return snap

    def load_snapshot(self, profile: ProfileConfig, day: date) -> DaySnapshot | None:
        raw = self.storage.get_snapshot_raw(profile.name, day)
        if not raw:
            return None
        row = self.storage.get_snapshot_row(profile.name, day) or {}
        return normalize_day(profile.name, day, profile.timezone or self.config.timezone, raw, row.get("errors") or {}, fetched_at=row.get("fetched_at"))

    def snapshot_for(self, profile: ProfileConfig, day: date | None = None, light: bool = True, max_age_s: int = 120) -> DaySnapshot | None:
        """Today: fetch (cached). Past days: stored snapshot, fetched once if missing."""
        day = day or self.today(profile)
        if day >= self.today(profile):
            try:
                return self.fetch_and_store(profile, day, light=light, max_age_s=max_age_s)
            except (GarminUnavailable, GarminAuthRequired) as exc:
                logger.warning("%s: live fetch failed (%s); using stored data", profile.name, exc)
                return self.load_snapshot(profile, day)
        snap = self.load_snapshot(profile, day)
        if snap is None:
            try:
                snap = self.fetch_and_store(profile, day, light=False)
            except (GarminUnavailable, GarminAuthRequired) as exc:
                logger.warning("%s: fetch of %s failed: %s", profile.name, day, exc)
        return snap

    def rows(self, profile: ProfileConfig, days: int, end: date | None = None) -> list[dict[str, Any]]:
        end = end or self.today(profile)
        start = end - timedelta(days=days - 1)
        return self.storage.get_snapshot_rows(profile.name, start, end)

    def day_range_utc(self, profile: ProfileConfig, start_day: date, end_day: date) -> tuple[datetime, datetime]:
        s, _ = local_day_bounds(start_day, profile.timezone)
        _, e = local_day_bounds(end_day, profile.timezone)
        return s, e

    # ------------------------------------------------------------- palpitations

    def _next_day_sleep_window(self, profile: ProfileConfig, day: date) -> list[tuple[datetime, datetime]]:
        row = self.storage.get_snapshot_row(profile.name, day + timedelta(days=1))
        if row and row.get("sleep_start") and row.get("sleep_end"):
            return [(row["sleep_start"], row["sleep_end"])]
        return []

    def run_detection(self, profile: ProfileConfig, snap: DaySnapshot, notify: bool = True) -> tuple[list[Episode], list[Episode]]:
        """Detect + persist episodes. Returns ``(new_episodes, episodes_to_notify)``."""
        if not profile.features.palpitations:
            return [], []
        cfg = profile.palpitations
        found = detect_episodes(snap, cfg, extra_sleep_windows=self._next_day_sleep_window(profile, snap.day))
        new: list[Episode] = []
        to_notify: list[Episode] = []
        for ep in found:
            stored, is_new = self.storage.upsert_episode(ep)
            if is_new:
                new.append(stored)
                logger.info("%s: new %s episode %s peak %d (conf %.2f)", profile.name, stored.kind, stored.start, stored.peak_hr, stored.confidence)
            if not notify and stored.notified_at is None:
                self.storage.mark_episode_notified(stored.id)  # historical: no alert
                continue
            if stored.notified_at is None and stored.confidence >= cfg.alert_min_confidence and cfg.notify:
                to_notify.append(stored)
        # attach numbers to symptom reports logged before the data arrived
        self._enrich_symptoms(profile, snap)
        if found:
            self._log_to_vault(profile, snap.day)
        return new, to_notify

    def _log_to_vault(self, profile: ProfileConfig, day: date) -> None:
        """Rewrite the Obsidian note for ``day`` (no-op without ``vault_dir``); never breaks monitoring."""
        if not self.config.vault_dir:
            return
        start, end = self.day_range_utc(profile, day, day)
        try:
            vault.write_day(self.config.vault_dir, profile.name, day, self.storage.get_episodes(profile.name, start, end),
                            profile.timezone or self.config.timezone, self.today(profile))
        except OSError as exc:
            logger.warning("%s: vault write failed: %s", profile.name, exc)

    def _enrich_symptoms(self, profile: ProfileConfig, snap: DaySnapshot) -> None:
        start, end = local_day_bounds(snap.day, profile.timezone)
        for rep in self.storage.symptoms_without_hr(profile.name):
            if not (start <= rep.event_time < end) or rep.id is None:
                continue
            hr, base = hr_context_at(snap, rep.event_time, profile.palpitations)
            ep = self.storage.find_overlapping_episode(profile.name, rep.event_time, rep.event_time, margin_minutes=15)
            if ep and ep.id is not None:
                self.storage.set_episode_felt(ep.id, True)
            if hr is not None or ep is not None:
                self.storage.update_symptom_hr(rep.id, hr, base, ep.id if ep else None)

    def assess_episode(self, profile: ProfileConfig, ep: Episode, snap: DaySnapshot | None = None) -> EpisodeAssessment | None:
        if self.llm is None or ep.id is None:
            return None
        if ep.llm_assessment:
            return EpisodeAssessment(
                assessment=ep.llm_assessment,
                confidence=ep.llm_confidence or 0.0,
                reasoning=ep.llm_reasoning or "",
                doctor_note=ep.doctor_note or "",
                model=ep.llm_model or "",
            )
        snap = snap or self.load_snapshot(profile, to_local(ep.start, profile.timezone).date())
        window = (ep.start - timedelta(days=3), ep.end + timedelta(hours=6))
        symptoms = self.storage.get_symptoms(profile.name, *window)
        try:
            result = analysis.assess_episode(self.llm, profile, ep, snap, symptoms)
        except LLMError as exc:
            logger.warning("%s: episode assessment failed: %s", profile.name, exc)
            return None
        self.storage.update_episode_assessment(ep.id, result.assessment, result.confidence, result.reasoning, result.doctor_note, result.model)
        ep.llm_assessment, ep.llm_confidence = result.assessment, result.confidence
        ep.llm_reasoning, ep.doctor_note, ep.llm_model = result.reasoning, result.doctor_note, result.model
        self._log_to_vault(profile, to_local(ep.start, profile.timezone).date())
        return result

    def assess_pending(self, profile: ProfileConfig, limit: int = 10) -> int:
        """Give unassessed (e.g. backfilled) episodes a model assessment. Returns the count done."""
        if self.llm is None:
            return 0
        done = 0
        for ep in self.storage.unassessed_episodes(profile.name, limit=limit):
            if self.assess_episode(profile, ep) is not None:
                done += 1
        return done

    # ------------------------------------------------------------------- poll

    def poll(self, profile: ProfileConfig, day: date | None = None, light: bool = True) -> PollResult:
        """One monitoring cycle: fetch today, detect, assess, evaluate alert rules."""
        day = day or self.today(profile)
        try:
            snap = self.fetch_and_store(profile, day, light=light)
        except GarminAuthRequired as exc:
            logger.error("%s: %s", profile.name, exc)
            return PollResult(profile=profile.name, snapshot=None, error=f"Garmin login needed: {exc}")
        except GarminUnavailable as exc:
            logger.warning("%s: %s", profile.name, exc)
            return PollResult(profile=profile.name, snapshot=None, error=str(exc))

        new, to_notify = self.run_detection(profile, snap, notify=(day == self.today(profile)))
        for ep in to_notify:
            self.assess_episode(profile, ep, snap)

        alerts: list[Alert] = []
        if profile.features.alerts and day == self.today(profile):
            prev_key = f"abnormal_hr:{profile.name}:{snap.date_str}"
            prev = self.storage.kv_get(prev_key)
            alerts = evaluate_alerts(profile, snap, now=self.clock(), previous_abnormal_count=int(prev) if prev else None)
            if snap.summary.abnormal_hr_alerts is not None:
                self.storage.kv_set(prev_key, str(snap.summary.abnormal_hr_alerts))
            alerts = [a for a in alerts if not self.storage.notification_sent(a.key)]
        return PollResult(profile=profile.name, snapshot=snap, new_episodes=new, to_notify=to_notify, alerts=alerts)

    def backfill(self, profile: ProfileConfig, days: int) -> int:
        """Fetch the last ``days`` days (full) without sending alerts. Returns days fetched."""
        done = 0
        today = self.today(profile)
        for i in range(days, 0, -1):
            d = today - timedelta(days=i)
            if self.storage.get_snapshot_raw(profile.name, d):
                continue
            try:
                snap = self.fetch_and_store(profile, d, light=False)
            except GarminAuthRequired:
                raise
            except GarminUnavailable as exc:
                logger.warning("%s: backfill %s failed: %s", profile.name, d, exc)
                break
            self.run_detection(profile, snap, notify=False)
            done += 1
        return done

    # ------------------------------------------------------------- coaching

    def _coaching(self, profile: ProfileConfig, snap: DaySnapshot, kind: str = "daily") -> CoachingAdvice | None:
        if not profile.features.daily_coaching:
            return None
        rows = self.rows(profile, 7, end=snap.day)
        start, end = self.day_range_utc(profile, snap.day, snap.day)
        episodes = self.storage.get_episodes(profile.name, start, end)
        symptoms = self.storage.get_symptoms(profile.name, start, end)
        advice: CoachingAdvice | None = None
        if self.llm is not None:
            try:
                advice = analysis.daily_coaching(self.llm, profile, rows, snap, episodes, symptoms)
            except LLMError as exc:
                logger.warning("%s: coaching via %s failed: %s", profile.name, describe_backend(self.llm), exc)
        if advice is None:
            advice = analysis.rule_based_coaching(profile, rows, snap, episodes_today=episodes, symptoms_today=symptoms)
        self.storage.save_analysis(profile.name, kind, snap.day, advice.model, advice.to_dict())
        return advice

    def stored_coaching(self, profile: ProfileConfig, day: date) -> CoachingAdvice | None:
        rec = self.storage.get_latest_analysis(profile.name, "daily", day)
        if not rec:
            return None
        p = rec["payload"]
        try:
            return CoachingAdvice(
                summary=p.get("summary", ""),
                do_more=list(p.get("do_more") or []),
                do_less=list(p.get("do_less") or []),
                watch_outs=list(p.get("watch_outs") or []),
                heart_note=p.get("heart_note"),
                model=p.get("model") or rec.get("model") or "",
                period=p.get("period") or "daily",
            )
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------ messages

    def fitness(self, profile: ProfileConfig, day: date) -> dict[str, Any] | None:
        """VO2 max, training status, intensity minutes, race times; ``None`` if Garmin refuses."""
        try:
            return self.session(profile).fetch_fitness(day)
        except (GarminUnavailable, GarminAuthRequired) as exc:
            logger.warning("%s: fitness data unavailable: %s", profile.name, exc)
            return None

    def morning_brief_text(self, profile: ProfileConfig) -> str:
        today = self.today(profile)
        yesterday = today - timedelta(days=1)
        snap = self.snapshot_for(profile, today, light=False, max_age_s=300)
        # Re-run yesterday's detection now that last night's sleep window is known
        if profile.features.palpitations:
            ysnap = self.load_snapshot(profile, yesterday)
            if ysnap is not None:
                self.run_detection(profile, ysnap, notify=True)
        yrow = self.storage.get_snapshot_row(profile.name, yesterday)
        start = to_utc(datetime.combine(yesterday, time(hour=20), tzinfo=get_tz(profile.timezone)))
        overnight = self.storage.get_episodes(profile.name, start, self.clock()) if profile.features.palpitations else []
        coaching = self.stored_coaching(profile, yesterday)
        return messages.morning_brief(profile, snap, yrow, overnight, coaching)

    def evening_summary_text(self, profile: ProfileConfig, day: date | None = None) -> str:
        day = day or self.today(profile)
        snap = self.snapshot_for(profile, day, light=False, max_age_s=120)
        if snap is None:
            return messages.esc(f"No Garmin data available for {profile.name} on {day.isoformat()}.")
        if profile.features.palpitations:
            self.run_detection(profile, snap, notify=(day == self.today(profile)))
            self.assess_pending(profile, limit=5)
        rows = self.rows(profile, 7, end=day)
        start, end = self.day_range_utc(profile, day, day)
        episodes = self.storage.get_episodes(profile.name, start, end)
        if profile.features.heart_review:  # heart-only profile: no coaching, no fitness
            return messages.heart_review(profile, day, episodes, snap)
        symptoms = self.storage.get_symptoms(profile.name, start, end)
        coaching = self._coaching(profile, snap)
        return messages.evening_summary(profile, snap, rows, episodes, symptoms, coaching, self.fitness(profile, snap.day))

    def weekly_review_text(self, profile: ProfileConfig) -> str:
        today = self.today(profile)
        this_week = self.rows(profile, 7, end=today)
        prev_week = self.rows(profile, 7, end=today - timedelta(days=7))
        start, end = self.day_range_utc(profile, today - timedelta(days=6), today)
        episodes = self.storage.get_episodes(profile.name, start, end) if profile.features.palpitations else []
        coaching: CoachingAdvice | None = None
        if self.llm is not None and profile.features.daily_coaching:
            try:
                coaching = analysis.weekly_review(self.llm, profile, this_week, prev_week, episodes)
                self.storage.save_analysis(profile.name, "weekly", today, coaching.model, coaching.to_dict())
            except LLMError as exc:
                logger.warning("%s: weekly review via %s failed: %s", profile.name, describe_backend(self.llm), exc)
        return messages.weekly_review(profile, this_week, prev_week, episodes, coaching)

    def today_text(self, profile: ProfileConfig) -> str:
        today = self.today(profile)
        snap = self.snapshot_for(profile, today, light=True, max_age_s=120)
        if snap is None:
            return messages.esc(f"No Garmin data yet for {profile.name} today.")
        if profile.features.palpitations:
            self.run_detection(profile, snap, notify=True)
        start, end = self.day_range_utc(profile, today, today)
        episodes = self.storage.get_episodes(profile.name, start, end)
        return messages.today_status(profile, snap, episodes, now=self.clock())

    def yesterday_text(self, profile: ProfileConfig) -> str:
        return self.evening_summary_text(profile, self.today(profile) - timedelta(days=1))

    def sleep_text(self, profile: ProfileConfig, day: date | None = None) -> str:
        day = day or self.today(profile)
        snap = self.snapshot_for(profile, day, light=False, max_age_s=600)
        if snap is None:
            return messages.esc(f"No sleep data for {profile.name} on {day.isoformat()}.")
        return messages.sleep_message(profile, snap)

    def steps_text(self, profile: ProfileConfig) -> str:
        today = self.today(profile)
        snap = self.snapshot_for(profile, today, light=True, max_age_s=120)
        if snap is None:
            return messages.esc(f"No step data yet for {profile.name} today.")
        return messages.steps_message(profile, snap, self.rows(profile, 7, end=today))

    def hr_chart(self, profile: ProfileConfig, day: date | None = None) -> tuple[str, bytes | None]:
        day = day or self.today(profile)
        snap = self.snapshot_for(profile, day, light=True, max_age_s=120)
        if snap is None:
            return messages.esc(f"No heart-rate data for {profile.name} on {day.isoformat()}."), None
        start, end = self.day_range_utc(profile, day, day)
        episodes = self.storage.get_episodes(profile.name, start, end)
        symptoms = self.storage.get_symptoms(profile.name, start, end)
        text = messages.hr_message(profile, snap, episodes)
        try:
            png = charts.hr_day_chart(snap, episodes, symptoms, profile.timezone or self.config.timezone)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chart failed: %s", exc)
            png = None
        return text, png

    def episodes_text(self, profile: ProfileConfig, days: int = 7) -> str:
        days = max(1, min(int(days), 365))
        today = self.today(profile)
        start, end = self.day_range_utc(profile, today - timedelta(days=days - 1), today)
        episodes = self.storage.get_episodes(profile.name, start, end)
        symptoms = self.storage.get_symptoms(profile.name, start, end)
        return messages.episodes_list(profile, episodes, symptoms, days)

    def episode_alert_payload(self, profile: ProfileConfig, ep: Episode) -> tuple[str, bytes | None]:
        assessment = None
        if ep.llm_assessment:
            assessment = EpisodeAssessment(
                assessment=ep.llm_assessment,
                confidence=ep.llm_confidence or 0.0,
                reasoning=ep.llm_reasoning or "",
                doctor_note=ep.doctor_note or "",
                model=ep.llm_model or "",
            )
        text = messages.episode_alert(profile, ep, assessment)
        png: bytes | None = None
        try:
            samples = self.storage.get_hr_samples(profile.name, ep.start - timedelta(minutes=45), ep.end + timedelta(minutes=45))
            if samples:
                png = charts.episode_chart(samples, ep, profile.timezone or self.config.timezone)
        except Exception as exc:  # noqa: BLE001
            logger.warning("episode chart failed: %s", exc)
        return text, png

    # ------------------------------------------------------------ symptoms

    def log_symptom(self, profile: ProfileConfig, event_time: datetime | None, note: str, chat_id: int | None) -> str:
        now = self.clock()
        event_time = to_utc(event_time) if event_time else now
        if event_time > now + timedelta(minutes=5):
            event_time = event_time - timedelta(days=1)
        rep = SymptomReport(profile=profile.name, reported_at=now, event_time=event_time, note=note.strip(), chat_id=chat_id, source="command")
        day = to_local(event_time, profile.timezone).date()
        snap: DaySnapshot | None = None
        if day == self.today(profile):
            try:
                snap = self.fetch_and_store(profile, day, light=True, max_age_s=60)
            except (GarminUnavailable, GarminAuthRequired) as exc:
                logger.warning("%s: could not refresh HR for symptom: %s", profile.name, exc)
                snap = self.load_snapshot(profile, day)
        else:
            snap = self.snapshot_for(profile, day)
        if snap is not None:
            rep.hr_at_time, rep.baseline_hr = hr_context_at(snap, event_time, profile.palpitations)
        if rep.hr_at_time is None:
            near = self.storage.hr_near(profile.name, event_time)
            rep.hr_at_time = near.hr if near else None
        ep = self.storage.find_overlapping_episode(profile.name, event_time, event_time, margin_minutes=15)
        if ep and ep.id is not None:
            rep.episode_id = ep.id
            self.storage.set_episode_felt(ep.id, True, note=note.strip() or None)
        if self.llm is not None and rep.note:
            try:
                rep.extracted = analysis.extract_symptom(self.llm, rep.note, event_time, profile.timezone or self.config.timezone)
            except LLMError as exc:
                logger.warning("%s: symptom extraction via %s failed: %s", profile.name, describe_backend(self.llm), exc)
        rep.red_flag = analysis.has_red_flag(rep.note, rep.extracted)
        self.storage.add_symptom(rep)
        return messages.symptom_logged(profile, rep)

    def set_felt(self, profile: ProfileConfig | None, episode_id: int, felt: bool) -> str:
        ep = self.storage.get_episode(episode_id)
        if ep is None:
            return "That episode no longer exists."
        self.storage.set_episode_felt(episode_id, felt)
        if felt:
            existing = [s for s in self.storage.get_symptoms(ep.profile, ep.start - timedelta(minutes=15), ep.end + timedelta(minutes=15)) if s.episode_id == episode_id]
            if not existing:
                self.storage.add_symptom(
                    SymptomReport(
                        profile=ep.profile,
                        reported_at=self.clock(),
                        event_time=ep.start,
                        note="Confirmed via alert button",
                        hr_at_time=ep.peak_hr,
                        baseline_hr=ep.baseline_hr,
                        episode_id=episode_id,
                        source="button",
                    )
                )
            return "Recorded: felt. Thank you, this goes into the doctor report."
        return "Recorded: not noticed. Thank you."

    def add_manual_episode(self, profile: ProfileConfig, start: datetime, minutes: int, note: str) -> Episode:
        """Record an episode the person describes (e.g. from before monitoring started)."""
        start = to_utc(start)
        end = start + timedelta(minutes=max(1, minutes))
        near = self.storage.hr_near(profile.name, start)
        hr = near.hr if near else 0
        ep = Episode(
            profile=profile.name,
            start=start,
            end=end,
            duration_min=float(max(1, minutes)),
            peak_hr=hr,
            mean_hr=float(hr),
            baseline_hr=float(hr),
            delta_hr=0.0,
            confidence=0.5,
            kind="sustained",
            source=EPISODE_SOURCE_MANUAL,
            felt=True,
            notes=note,
            notified_at=self.clock(),
        )
        from .utils import stable_hash

        ep.fingerprint = stable_hash(profile.name, start.isoformat(), "manual")
        stored, _ = self.storage.upsert_episode(ep)
        return stored

    # -------------------------------------------------------------- reports

    def doctor_report(self, profile: ProfileConfig, days: int = 30) -> report.ReportFiles:
        days = max(1, min(int(days), 365))
        end = self.today(profile)
        start = end - timedelta(days=days - 1)
        self.assess_pending(profile, limit=20)
        narrative: str | None = None
        if self.llm is not None:
            s, e = self.day_range_utc(profile, start, end)
            episodes = self.storage.get_episodes(profile.name, s, e)
            symptoms = self.storage.get_symptoms(profile.name, s, e)
            rows = self.storage.get_snapshot_rows(profile.name, start, end)
            if episodes or symptoms:
                try:
                    narrative = analysis.doctor_narrative(self.llm, profile, episodes, symptoms, rows)
                except LLMError as exc:
                    logger.warning("%s: doctor narrative failed: %s", profile.name, exc)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        return report.generate_doctor_report(profile, self.storage, start, end, self.reports_dir, narrative=narrative)

    def analyze_now(self, profile: ProfileConfig) -> str:
        today = self.today(profile)
        snap = self.snapshot_for(profile, today, light=False, max_age_s=120)
        if snap is None:
            return messages.esc(f"No data to analyse for {profile.name} yet.")
        advice = self._coaching(profile, snap)
        if advice is None:
            return messages.esc("Coaching is disabled for this profile.")
        header = f"🧠 <b>Analysis for {messages.esc(profile.name)}</b> ({messages.esc(today.isoformat())}, {messages.esc(advice.model)})\n"
        return header + messages.coaching_block(advice)

    def status_text(self) -> str:
        lines = [f"<b>Garmin Health Monitor</b> up since {self.started_at.strftime('%Y-%m-%d %H:%M')} UTC"]
        llm_state = "disabled"
        if self.llm is not None:
            try:
                llm_state = f"{describe_backend(self.llm)} {'OK' if self.llm.is_available() else 'UNAVAILABLE'}"
            except Exception as exc:  # noqa: BLE001
                llm_state = f"{describe_backend(self.llm)} error: {exc}"
        lines.append(f"LLM: {messages.esc(llm_state)}")
        for p in self.config.profiles:
            days = self.storage.days_with_data(p.name)
            last = days[-1] if days else "none"
            tokens = "tokens ✅" if self.session(p).has_tokens() else "tokens ❌ (run login)"
            today = self.today(p)
            s, e = self.day_range_utc(p, today - timedelta(days=6), today)
            n_eps = len(self.storage.get_episodes(p.name, s, e)) if p.features.palpitations else 0
            row = self.storage.get_snapshot_row(p.name, today)
            fetched = row["fetched_at"].strftime("%H:%M UTC") if row and row.get("fetched_at") else "never today"
            lines.append(
                f"• <b>{messages.esc(p.name)}</b>: {len(days)} days stored (latest {messages.esc(last)}), "
                f"last fetch {messages.esc(fetched)}, {tokens}"
                + (f", {n_eps} episode(s) in 7d" if p.features.palpitations else "")
            )
        return "\n".join(lines)
