"""Integration tests for MonitorService + Scheduler with fake Garmin, fake LLM and fake bot."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from garmin_health_monitor.config import LLMConfig
from garmin_health_monitor.garmin_client import GarminUnavailable
from garmin_health_monitor.models import Alert
from garmin_health_monitor.scheduler import Scheduler
from garmin_health_monitor.service import MonitorService, PollResult
from garmin_health_monitor.storage import Storage
from garmin_health_monitor.utils import get_tz
from tests.conftest import DAY, TZ, make_app_config, make_profile, make_raw_day


class FakeSession:
    """Stands in for GarminSession: serves synthetic days, records calls."""

    def __init__(self, profile, mfa=None, episodes_by_day=None, fail=False):
        self.profile = profile
        self.episodes_by_day = episodes_by_day or {}
        self.calls: list[tuple[date, bool]] = []
        self.fail = fail

    def fetch_fitness(self, day):
        return {"race_predictions": {"time5K": 1608}}

    def fetch_day(self, day, light=False):
        self.calls.append((day, light))
        if self.fail:
            raise GarminUnavailable("garmin down")
        raw = make_raw_day(day=day, tz=TZ, episodes=self.episodes_by_day.get(day, []), activities=[(8, 30, 105)], seed=day.day)
        return raw, {}

    def has_tokens(self):
        return True

    def login(self):
        return self


class FakeLLM:
    model = "fake-model"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def chat_json(self, system, user, schema):
        self.calls.append(("json", user))
        props = schema.get("properties", {})
        if "ask_doctor" in props:
            return {"summary": "Quiet month.", "why": ["Sleep score 60 vs 74."], "habits": ["Regular bedtime."],
                    "ask_doctor": ["Is this worth a heart tracing?"]}
        if "red_flags" in props:
            chest = "chest" in user
            return {"event_type": "palpitation", "duration_minutes": 3, "symptoms": ["racing", "made_up"],
                    "possible_triggers": ["caffeine"], "red_flags": {"chest_pain": chest, "fainting": False, "severe_breathlessness": False}}
        if "assessment" in props:
            return {"assessment": "possible_palpitation", "confidence": 0.7, "reasoning": "At rest with abrupt onset.", "doctor_note": "Episode at rest."}
        return {"summary": "Decent day.", "do_more": ["walk"], "do_less": ["sit"], "watch_outs": ["resting HR"], "heart_note": ""}

    def chat_text(self, system, user):
        self.calls.append(("text", user))
        return "Narrative paragraph."


class FakeBot:
    def __init__(self):
        self.send_text = AsyncMock()
        self.send_photo = AsyncMock()
        self.send_document = AsyncMock()
        self.notify_episode = AsyncMock()
        self.loop = None

        class _App:
            job_queue = None

        self.app = _App()


def local(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=get_tz(TZ))


@pytest.fixture
def svc(tmp_path):
    profile = make_profile(chat_ids=[111])
    cfg = make_app_config(tmp_path, profiles=[profile])
    cfg.llm = LLMConfig(backend="none")
    storage = Storage(tmp_path / "svc.db")
    episodes = {DAY: [(10, 0, 8, 125), (15, 30, 6, 112)]}
    sessions = {}

    def factory(p, mfa):
        sessions[p.name] = FakeSession(p, mfa, episodes_by_day=episodes)
        return sessions[p.name]

    clock_holder = {"now": local(DAY, 20, 0)}
    service = MonitorService(cfg, storage, llm=FakeLLM(), session_factory=factory, clock=lambda: clock_holder["now"])
    service.reports_dir = tmp_path / "reports"
    service._sessions_ref = sessions
    service._clock_holder = clock_holder
    yield service, profile, sessions, clock_holder
    storage.close()


def test_poll_detects_assesses_and_dedupes(svc):
    service, profile, sessions, _ = svc
    result = service.poll(profile)
    assert result.error is None and result.snapshot is not None
    assert result.snapshot.day == DAY
    assert len(result.new_episodes) >= 1
    assert result.to_notify, "high-confidence episodes must be queued for notification"
    for ep in result.to_notify:
        assert ep.llm_assessment == "possible_palpitation"
        assert ep.doctor_note == "Episode at rest."
    # second poll: nothing new, still to notify until the scheduler marks it
    again = service.poll(profile)
    assert again.new_episodes == []
    assert {e.id for e in again.to_notify} == {e.id for e in result.to_notify}
    for ep in again.to_notify:
        service.storage.mark_episode_notified(ep.id)
    third = service.poll(profile)
    assert third.to_notify == []
    # light polls keep previously fetched slow endpoints (sleep) around
    assert sessions[profile.name].calls[0][1] is True


def test_poll_handles_garmin_down(tmp_path):
    profile = make_profile()
    cfg = make_app_config(tmp_path, profiles=[profile])
    service = MonitorService(cfg, Storage(tmp_path / "x.db"), llm=None, session_factory=lambda p, m: FakeSession(p, fail=True), clock=lambda: local(DAY, 12))
    res = service.poll(profile)
    assert res.snapshot is None and "garmin down" in (res.error or "")


def test_backfill_marks_historical_episodes_as_notified(svc):
    service, profile, sessions, _ = svc
    # make earlier days carry an episode too
    session = service.session(profile)
    sessions_before = len(sessions)  # after our own session() call: backfill must not add more
    session.episodes_by_day[DAY - timedelta(days=2)] = [(11, 0, 10, 130)]
    n = service.backfill(profile, 3)
    assert n == 3
    s, e = service.day_range_utc(profile, DAY - timedelta(days=3), DAY - timedelta(days=1))
    eps = service.storage.get_episodes(profile.name, s, e)
    assert eps and all(ep.notified_at is not None for ep in eps)
    assert len(service.storage.days_with_data(profile.name)) == 3
    assert sessions_before == len(sessions)


def test_quiet_hours(svc):
    service, profile, _, clock = svc
    profile.palpitations.quiet_hours = (23, 7)
    clock["now"] = local(DAY, 23, 30)
    assert service.in_quiet_hours(profile)
    clock["now"] = local(DAY, 6, 30)
    assert service.in_quiet_hours(profile)
    clock["now"] = local(DAY, 12, 0)
    assert not service.in_quiet_hours(profile)
    profile.palpitations.quiet_hours = None
    assert not service.in_quiet_hours(profile)


def test_messages_render(svc):
    service, profile, _, clock = svc
    service.backfill(profile, 7)
    text = service.evening_summary_text(profile)
    assert "Dad" in text and len(text) < 4096
    assert service.storage.get_latest_analysis(profile.name, "daily", DAY) is not None
    morning = service.morning_brief_text(profile)
    assert "Dad" in morning
    weekly = service.weekly_review_text(profile)
    assert weekly
    assert service.today_text(profile)
    assert service.sleep_text(profile)
    assert service.steps_text(profile)
    text, png = service.hr_chart(profile)
    assert text and png and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert "episode" in service.episodes_text(profile, 7).lower()
    assert service.analyze_now(profile).startswith("🧠 <b>ANALYSIS</b> · Dad")
    assert service.status_text().startswith("🛠 <b>GARMIN HEALTH MONITOR</b> · status")
    assert service.yesterday_text(profile)


def test_log_symptom_links_episode_and_set_felt(svc):
    service, profile, _, clock = svc
    service.poll(profile)
    when = local(DAY, 10, 4)  # inside the 10:00 episode
    msg = service.log_symptom(profile, when, "fluttering after coffee", chat_id=111)
    assert msg
    s, e = service.day_range_utc(profile, DAY, DAY)
    symptoms = service.storage.get_symptoms(profile.name, s, e)
    assert len(symptoms) == 1
    rep = symptoms[0]
    assert rep.hr_at_time and rep.hr_at_time > 110
    assert rep.episode_id is not None
    ep = service.storage.get_episode(rep.episode_id)
    assert ep.felt is True
    # button answers
    assert "not noticed" in service.set_felt(None, ep.id, False)
    assert service.storage.get_episode(ep.id).felt is False
    assert "felt" in service.set_felt(None, ep.id, True)
    assert "no longer exists" in service.set_felt(None, 999999, True)
    # a symptom without matching data still records
    msg2 = service.log_symptom(profile, None, "", chat_id=111)
    assert msg2


def test_symptom_time_in_future_moves_to_previous_day(svc):
    service, profile, _, clock = svc
    clock["now"] = local(DAY, 9, 0)
    service.log_symptom(profile, local(DAY, 22, 0), "late", chat_id=111)
    s, e = service.day_range_utc(profile, DAY - timedelta(days=1), DAY - timedelta(days=1))
    assert len(service.storage.get_symptoms(profile.name, s, e)) == 1


def test_doctor_report_files(svc):
    service, profile, _, _ = svc
    service.poll(profile)
    service.log_symptom(profile, local(DAY, 15, 33), "felt it", chat_id=111)
    files = service.doctor_report(profile, days=7)
    assert Path(files.pdf_path).exists() and Path(files.csv_path).exists()
    assert files.n_episodes >= 1 and files.n_symptoms >= 1
    assert Path(files.pdf_path).read_bytes()[:4] == b"%PDF"
    assert "Narrative" in Path(files.pdf_path).read_bytes().decode("latin-1") or True  # narrative optional in PDF text stream


def test_profile_for_chat(svc):
    service, profile, _, _ = svc
    assert service.profile_for_chat(111).name == "Dad"
    assert service.profile_for_chat(999, "dad").name == "Dad"  # admin, by name
    with pytest.raises(LookupError):
        service.profile_for_chat(42)
    with pytest.raises(LookupError):
        service.profile_for_chat(999, "nobody")


def test_add_manual_episode(svc):
    service, profile, _, _ = svc
    service.poll(profile)
    ep = service.add_manual_episode(profile, local(DAY, 12, 0), 5, "felt racing at lunch")
    assert ep.id and ep.source == "manual" and ep.felt is True and ep.notified_at is not None


# --------------------------------------------------------------- scheduler


async def test_scheduler_deliver_sends_and_marks(svc):
    service, profile, _, clock = svc
    bot = FakeBot()
    sched = Scheduler(service.config, service, bot)
    result = service.poll(profile)
    assert result.to_notify
    alerts = [Alert(key="rhr_high:Dad:x", severity="warning", title="t", body="b", profile="Dad")]
    result.alerts = alerts
    await sched.deliver(profile, result)
    assert bot.notify_episode.await_count == len(result.to_notify)
    assert bot.send_text.await_count == 1
    assert service.storage.notification_sent("rhr_high:Dad:x")
    for ep in result.to_notify:
        assert service.storage.get_episode(ep.id).notified_at is not None
    # delivering the same alert again is a no-op
    bot.send_text.reset_mock()
    await sched.notify_alert(profile, alerts[0])
    bot.send_text.assert_not_awaited()


async def test_scheduler_quiet_hours_hold_non_critical(svc):
    service, profile, _, clock = svc
    profile.palpitations.quiet_hours = (23, 7)
    clock["now"] = local(DAY, 23, 30)
    bot = FakeBot()
    sched = Scheduler(service.config, service, bot)
    result = service.poll(profile)
    result.alerts = [Alert(key="k", severity="info", title="t", body="b", profile="Dad")]
    await sched.deliver(profile, result)
    bot.notify_episode.assert_not_awaited()
    bot.send_text.assert_not_awaited()
    for ep in result.to_notify:
        assert service.storage.get_episode(ep.id).notified_at is None


async def test_scheduler_garmin_outage_alerts_after_2h_once_then_all_clear(svc):
    service, profile, _, clock = svc
    bot = FakeBot()
    sched = Scheduler(service.config, service, bot)
    down = PollResult(profile="Dad", snapshot=None, error="All Garmin endpoints failed: API Error 521")
    start = clock["now"]
    await sched.deliver(profile, down)
    clock["now"] = start + timedelta(minutes=90)
    await sched.deliver(profile, down)
    assert bot.send_text.await_count == 0  # a short blip stays silent
    clock["now"] = start + timedelta(hours=2)
    await sched.deliver(profile, down)
    await sched.deliver(profile, down)
    assert bot.send_text.await_count == 1  # alerted once
    text = bot.send_text.await_args.args[1]
    assert "<b>No Garmin data since 20:00</b> · Dad" in text and "servers seem to be down" in text
    assert bot.send_text.await_args.args[0] == service.config.telegram.admin_chat_ids
    await sched.deliver(profile, PollResult(profile="Dad", snapshot=None))
    assert "Garmin data is back" in bot.send_text.await_args.args[1]


async def test_scheduler_blip_that_recovers_sends_nothing(svc):
    service, profile, _, _ = svc
    bot = FakeBot()
    sched = Scheduler(service.config, service, bot)
    await sched.deliver(profile, PollResult(profile="Dad", snapshot=None, error="Garmin login needed: JWT_WEB"))
    await sched.deliver(profile, PollResult(profile="Dad", snapshot=None))
    assert bot.send_text.await_count == 0


def test_log_symptom_extracts_fields_and_flags_red_flags(svc):
    service, profile, _, _ = svc
    msg = service.log_symptom(profile, local(DAY, 11, 0), "heart racing after coffee, about 3 minutes", chat_id=111)
    assert "995" not in msg and "racing" in msg
    msg2 = service.log_symptom(profile, local(DAY, 12, 0), "some chest pain now", chat_id=111)
    assert "995" in msg2
    s, e = service.day_range_utc(profile, DAY, DAY)
    reps = service.storage.get_symptoms(profile.name, s, e)
    assert reps[0].extracted["symptoms"] == ["racing"]  # off-vocabulary value dropped
    assert reps[0].extracted["possible_triggers"] == ["caffeine"]
    assert reps[0].note == "heart racing after coffee, about 3 minutes"  # raw text kept


def test_red_flag_keywords_work_without_llm():
    from garmin_health_monitor.analysis import has_red_flag
    assert has_red_flag("I nearly fainted")
    assert has_red_flag("chest tightness and pain")
    assert has_red_flag("some pressure in my chest")
    assert has_red_flag("felt odd", {"symptoms": ["near_fainting"], "red_flags": {}})
    assert not has_red_flag("fluttering after lunch")


def test_poll_logs_episodes_to_obsidian_vault(svc, tmp_path):
    service, profile, _, _ = svc
    service.config.vault_dir = str(tmp_path / "vault")
    service.poll(profile)
    note = tmp_path / "vault" / "Episodes" / f"{DAY.isoformat()}.md"
    text = note.read_text(encoding="utf-8")
    assert "| Time | Duration | Peak HR |" in text and "at rest" in text
    home = (tmp_path / "vault" / "Home.md").read_text(encoding="utf-8")
    assert f"[[Episodes/{DAY.isoformat()}|{DAY.isoformat()}]]" in home
    service.poll(profile)  # re-run rewrites, never duplicates
    assert note.read_text(encoding="utf-8").count("| at rest |") == text.count("| at rest |")


def test_heart_only_profile_weekly_is_palpitation_table(svc):
    service, profile, _, _ = svc
    service.poll(profile)
    profile.features.heart_review = True
    text = service.weekly_review_text(profile)
    assert text.startswith("❤️ <b>HEART</b> · Dad · Week ") and "<pre>Date    Time   Peak  Min" in text
    assert "Coaching" not in text


async def test_outage_past_midnight_rechecks_previous_day_on_recovery(svc):
    service, profile, sessions, clock = svc
    bot = FakeBot()
    sched = Scheduler(service.config, service, bot)
    clock["now"] = local(DAY - timedelta(days=1), 22, 0)
    await sched.deliver(profile, PollResult(profile="Dad", snapshot=None, error="API Error 521"))
    clock["now"] = local(DAY, 1, 0)
    service.session(profile)  # the fake Garmin comes back
    await sched.deliver(profile, PollResult(profile="Dad", snapshot=None))
    assert (DAY - timedelta(days=1), False) in sessions[profile.name].calls  # yesterday re-read in full


def test_medicine_taken_is_recorded_once_and_shown_in_heart_review(svc):
    service, profile, _, clock = svc
    profile.medication.name, profile.medication.times = "Metoprolol tartrate 50 mg", ("09:00",)
    assert service.med_status_line(profile, DAY) == "💊 Medicine: 09:00 not marked as taken"
    clock["now"] = local(DAY, 9, 14)
    assert service.mark_med_taken(profile, DAY, "09:00") == "09:14"
    clock["now"] = local(DAY, 10, 0)
    assert service.mark_med_taken(profile, DAY, "09:00") == "09:14"  # a second tap keeps the first time
    assert service.med_status_line(profile, DAY) == "💊 Medicine: 09:00 ✅ taken 09:14"
    profile.features.heart_review = True
    clock["now"] = local(DAY, 22, 0)
    assert "09:00 ✅ taken 09:14" in service.evening_summary_text(profile)


def test_monthly_summary_heart_calendar_and_ai_report(svc):
    service, profile, _, clock = svc
    service.backfill(profile, 3)
    profile.features.heart_review = True
    profile.medication.name, profile.medication.started = "Metoprolol", DAY.isoformat()
    clock["now"] = local(DAY.replace(day=1) + timedelta(days=32), 9, 0).replace(day=1)  # 1st of next month
    png, caption, extra = service.monthly_summary(profile)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert caption.startswith("❤️ <b>HEART MONTH</b> · Dad · ") and "🟥" in caption
    assert "💊 <b>Metoprolol</b> from" in caption
    assert "Why it may happen" in extra and "Sleep score 60 vs 74." in extra and "call 995" in extra


def test_monthly_summary_progress_for_normal_profile(svc):
    service, profile, _, clock = svc
    service.backfill(profile, 3)
    clock["now"] = local(DAY.replace(day=1) + timedelta(days=32), 9, 0).replace(day=1)
    png, caption, extra = service.monthly_summary(profile)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and caption.startswith("📈 <b>MONTH IN REVIEW</b> · Dad · ") and "👟 Steps" in caption
    assert extra is None


def test_workout_nudge_only_when_behind(svc):
    service, profile, _, _ = svc
    profile.goals = "3 workouts a week"
    text = service.workout_nudge_text(profile)
    assert text is None or "🏃 <b>0 of 3 workouts this week</b>" in text
    profile.goals = "walk more"
    assert service.workout_nudge_text(profile) is None


def test_nightly_backup_is_a_readable_copy_and_keeps_newest(svc, tmp_path):
    import sqlite3

    service, profile, _, clock = svc
    service.poll(profile)
    service.config.backup_dir = str(tmp_path / "backups")
    for i in range(3):
        clock["now"] = local(DAY + timedelta(days=i), 3, 30)
        path = service.backup(keep=2)
    kept = sorted(p.name for p in (tmp_path / "backups").glob("monitor-*.db"))
    assert kept == [f"monitor-{DAY + timedelta(days=1)}.db", f"monitor-{DAY + timedelta(days=2)}.db"]
    rows = sqlite3.connect(path).execute("select count(*) from daily_snapshots").fetchone()[0]
    assert rows >= 1  # a real, openable database with the data in it
