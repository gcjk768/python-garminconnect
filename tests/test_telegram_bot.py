"""Tests for ``garmin_health_monitor.telegram_bot`` (no network: handlers are called directly)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import InlineKeyboardMarkup
from telegram.error import BadRequest

from garmin_health_monitor.telegram_bot import (
    FELT_CALLBACK_RE,
    HealthBot,
    MfaBroker,
    build_felt_keyboard,
    chunk_text,
    parse_symptom_args,
)
from tests.conftest import TZ, make_app_config, make_profile

USER_CHAT = 111  # Dad's chat (see make_profile)
ADMIN_CHAT = 999  # admin (see make_app_config)
STRANGER_CHAT = 555


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeReportFiles:
    pdf_path: Path
    csv_path: Path
    summary: str = "ok"
    n_episodes: int = 3
    n_symptoms: int = 2


class FakeService:
    """Stands in for MonitorService: records calls and returns canned values."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail: set[str] = set()
        self.hr_png: bytes | None = b"\x89PNG fake"

    def _rec(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        if name in self.fail:
            raise RuntimeError(f"{name} boom")

    def today_text(self, profile):
        self._rec("today_text", profile)
        return f"<b>Today for {profile.name}</b>: steps 1,234"

    def yesterday_text(self, profile):
        self._rec("yesterday_text", profile)
        return f"Yesterday for {profile.name}"

    def sleep_text(self, profile, day=None):
        self._rec("sleep_text", profile, day)
        return f"Sleep for {profile.name}"

    def steps_text(self, profile):
        self._rec("steps_text", profile)
        return f"Steps for {profile.name}"

    def hr_chart(self, profile, day=None):
        self._rec("hr_chart", profile, day)
        return (f"HR for {profile.name} {day}", self.hr_png)

    def episodes_text(self, profile, days=7):
        self._rec("episodes_text", profile, days)
        return f"Episodes {profile.name} {days}d"

    def log_symptom(self, profile, event_time, note, chat_id):
        self._rec("log_symptom", profile, event_time, note, chat_id)
        return f"Logged: {note or '(no note)'}"

    def set_felt(self, profile, episode_id, felt):
        self._rec("set_felt", profile, episode_id, felt)
        return "Recorded: felt." if felt else "Recorded: not noticed."

    def doctor_report(self, profile, days=30):
        self._rec("doctor_report", profile, days)
        pdf = self.tmp_path / f"{profile.slug}-report.pdf"
        csv = self.tmp_path / f"{profile.slug}-report.csv"
        pdf.write_bytes(b"%PDF-1.4 fake")
        csv.write_text("record_type,start\nepisode,2026-09-27T10:00\n")
        return FakeReportFiles(pdf_path=pdf, csv_path=csv)

    def analyze_now(self, profile):
        self._rec("analyze_now", profile)
        return f"Analysis for {profile.name}"

    def status_text(self):
        self._rec("status_text")
        return "<b>Garmin Health Monitor</b> up"

    def called(self, name: str) -> list[tuple[Any, ...]]:
        return [args for n, args in self.calls if n == name]


def make_update(chat_id: int, text: str = "/today", user_id: int | None = None) -> MagicMock:
    """A minimal ``telegram.Update`` stand-in with AsyncMock reply methods."""
    message = MagicMock(name="message")
    message.text = text
    message.reply_text = AsyncMock(name="reply_text")
    message.reply_photo = AsyncMock(name="reply_photo")
    message.reply_document = AsyncMock(name="reply_document")
    update = MagicMock(name="update")
    update.effective_chat = MagicMock(id=chat_id)
    update.effective_user = MagicMock(id=user_id or chat_id)
    update.effective_message = message
    update.message = message
    update.callback_query = None
    return update


def make_context(args: list[str] | None = None) -> MagicMock:
    ctx = MagicMock(name="context")
    ctx.args = list(args or [])
    return ctx


def make_callback_update(
    chat_id: int, data: str, text: str | None = "Alert text", caption: str | None = None
) -> MagicMock:
    query = MagicMock(name="callback_query")
    query.data = data
    query.answer = AsyncMock(name="answer")
    query.edit_message_text = AsyncMock(name="edit_message_text")
    query.edit_message_caption = AsyncMock(name="edit_message_caption")
    query.edit_message_reply_markup = AsyncMock(name="edit_message_reply_markup")
    msg = MagicMock(name="alert_message")
    msg.text = text
    msg.text_html = text
    msg.caption = caption
    msg.caption_html = caption
    query.message = msg
    update = MagicMock(name="update")
    update.effective_chat = MagicMock(id=chat_id)
    update.effective_user = MagicMock(id=chat_id)
    update.callback_query = query
    update.effective_message = msg
    update.message = None
    return update


def replies(update: MagicMock) -> list[str]:
    return [
        c.kwargs.get("text") or (c.args[0] if c.args else "")
        for c in update.message.reply_text.await_args_list
    ]


@pytest.fixture
def service(tmp_path):
    return FakeService(tmp_path)


@pytest.fixture
def bot(tmp_path, service):
    cfg = make_app_config(tmp_path)
    return HealthBot(cfg, service)


@pytest.fixture
def two_profile_bot(tmp_path, service):
    # chat 111 sees both Dad and Mum; chat 222 sees only Mum
    cfg = make_app_config(
        tmp_path,
        profiles=[make_profile("Dad", chat_ids=[111]), make_profile("Mum", chat_ids=[111, 222])],
    )
    return HealthBot(cfg, service)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_bot_builds_without_network(bot):
    assert bot.app is not None
    assert bot.app.job_queue is not None  # scheduler needs it
    assert isinstance(bot.mfa, MfaBroker)
    assert bot.loop is None
    handlers = [h for group in bot.app.handlers.values() for h in group]
    names = {n for h in handlers for n in getattr(h, "commands", [])}
    for cmd in (
        "start",
        "help",
        "profiles",
        "today",
        "yesterday",
        "sleep",
        "hr",
        "steps",
        "episodes",
        "palp",
        "note",
        "report",
        "analyze",
        "status",
        "mfa",
    ):
        assert cmd in names, cmd


def test_bot_accepts_missing_service(tmp_path):
    hb = HealthBot(make_app_config(tmp_path), None)
    assert hb.service is None


# ---------------------------------------------------------------------------
# Authorisation and profile resolution
# ---------------------------------------------------------------------------


async def test_unauthorised_chat_is_ignored(bot, service, caplog):
    upd = make_update(STRANGER_CHAT)
    with caplog.at_level("INFO"):
        await bot.cmd_today(upd, make_context())
        await bot.cmd_start(upd, make_context())
        await bot.cmd_palp(upd, make_context(["10:00"]))
        await bot.cmd_report(upd, make_context())
    upd.message.reply_text.assert_not_awaited()
    upd.message.reply_document.assert_not_awaited()
    assert service.calls == []
    assert any("unauthorised" in r.message.lower() for r in caplog.records)


async def test_unauthorised_callback_is_ignored(bot, service):
    upd = make_callback_update(STRANGER_CHAT, "felt:5:yes")
    await bot.cb_felt(upd, make_context())
    assert service.called("set_felt") == []
    upd.callback_query.edit_message_text.assert_not_awaited()


async def test_today_with_single_profile(bot, service):
    upd = make_update(USER_CHAT, "/today")
    await bot.cmd_today(upd, make_context())
    assert len(service.called("today_text")) == 1
    (profile,) = service.called("today_text")[0]
    assert profile.name == "Dad"
    upd.message.reply_text.assert_awaited_once()
    kwargs = upd.message.reply_text.await_args.kwargs
    assert "Today for Dad" in kwargs["text"]
    assert kwargs["parse_mode"] == "HTML"


async def test_admin_sees_profile_without_naming_it(bot, service):
    upd = make_update(ADMIN_CHAT, "/steps")
    await bot.cmd_steps(upd, make_context())
    assert service.called("steps_text")[0][0].name == "Dad"


async def test_profile_selection_by_name(two_profile_bot, service):
    upd = make_update(111, "/today mum")
    await two_profile_bot.cmd_today(upd, make_context(["mum"]))
    assert service.called("today_text")[0][0].name == "Mum"
    assert "Today for Mum" in replies(upd)[0]


async def test_profile_selection_by_slug_and_args_are_stripped(two_profile_bot, service):
    upd = make_update(111, "/episodes dad 14")
    await two_profile_bot.cmd_episodes(upd, make_context(["dad", "14"]))
    profile, days = service.called("episodes_text")[0]
    assert profile.name == "Dad"
    assert days == 14


async def test_ambiguous_profile_asks_to_specify(two_profile_bot, service):
    upd = make_update(111, "/today")
    await two_profile_bot.cmd_today(upd, make_context())
    assert service.calls == []
    text = replies(upd)[0]
    assert "Dad" in text and "Mum" in text
    assert "Which person" in text


async def test_chat_cannot_pick_profile_it_may_not_see(two_profile_bot, service):
    upd = make_update(222, "/today dad")
    await two_profile_bot.cmd_today(upd, make_context(["dad"]))
    # 222 only sees Mum: "dad" is not a profile match, so Mum is used and "dad" stays an arg
    assert service.called("today_text")[0][0].name == "Mum"


async def test_service_missing_replies_gently(tmp_path):
    hb = HealthBot(make_app_config(tmp_path), None)
    upd = make_update(USER_CHAT)
    await hb.cmd_today(upd, make_context())
    assert "not running" in replies(upd)[0]


# ---------------------------------------------------------------------------
# Simple commands
# ---------------------------------------------------------------------------


async def test_start_lists_profiles_and_help(bot):
    upd = make_update(USER_CHAT, "/start")
    await bot.cmd_start(upd, make_context())
    text = replies(upd)[0]
    assert "Dad" in text
    assert "/today" in text and "/palp" in text
    assert "/status" not in text  # admin-only entries hidden from a normal chat


async def test_help_for_admin_includes_admin_commands(bot):
    upd = make_update(ADMIN_CHAT, "/help")
    await bot.cmd_help(upd, make_context())
    text = replies(upd)[0]
    assert "/status" in text and "/mfa" in text


async def test_profiles_command(two_profile_bot):
    upd = make_update(111, "/profiles")
    await two_profile_bot.cmd_profiles(upd, make_context())
    text = replies(upd)[0]
    assert "Dad" in text and "Mum" in text and TZ in text


async def test_yesterday_sleep_steps_analyze(bot, service):
    for handler, name in (
        (bot.cmd_yesterday, "yesterday_text"),
        (bot.cmd_sleep, "sleep_text"),
        (bot.cmd_steps, "steps_text"),
        (bot.cmd_analyze, "analyze_now"),
    ):
        upd = make_update(USER_CHAT)
        await handler(upd, make_context())
        assert service.called(name), name
        assert any("Dad" in t for t in replies(upd)), name


async def test_service_error_becomes_short_message(bot, service):
    service.fail.add("today_text")
    upd = make_update(USER_CHAT)
    await bot.cmd_today(upd, make_context())
    text = replies(upd)[0]
    assert "Sorry" in text
    assert "Traceback" not in text and "boom" not in text


async def test_episodes_default_and_cap(bot, service):
    upd = make_update(USER_CHAT)
    await bot.cmd_episodes(upd, make_context())
    assert service.called("episodes_text")[0][1] == 7
    await bot.cmd_episodes(upd, make_context(["500"]))
    assert service.called("episodes_text")[1][1] == 90
    await bot.cmd_episodes(upd, make_context(["abc"]))
    assert service.called("episodes_text")[2][1] == 7


# ---------------------------------------------------------------------------
# /hr
# ---------------------------------------------------------------------------


async def test_hr_sends_photo_with_caption(bot, service):
    upd = make_update(USER_CHAT, "/hr 2026-09-27")
    await bot.cmd_hr(upd, make_context(["2026-09-27"]))
    profile, day = service.called("hr_chart")[0]
    assert day.isoformat() == "2026-09-27"
    upd.message.reply_photo.assert_awaited_once()
    kwargs = upd.message.reply_photo.await_args.kwargs
    assert "HR for Dad" in kwargs["caption"]
    upd.message.reply_text.assert_not_awaited()


async def test_hr_without_png_sends_text(bot, service):
    service.hr_png = None
    upd = make_update(USER_CHAT, "/hr")
    await bot.cmd_hr(upd, make_context())
    assert service.called("hr_chart")[0][1] is None
    upd.message.reply_photo.assert_not_awaited()
    assert "HR for Dad" in replies(upd)[0]


async def test_hr_bad_date(bot, service):
    upd = make_update(USER_CHAT, "/hr nonsense")
    await bot.cmd_hr(upd, make_context(["nonsense"]))
    assert service.calls == []
    assert "YYYY-MM-DD" in replies(upd)[0]


# ---------------------------------------------------------------------------
# /palp and /note
# ---------------------------------------------------------------------------


def test_parse_symptom_args_time_and_note():
    now = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)  # 18:00 in Singapore
    when, note = parse_symptom_args(["14:30", "felt", "fluttering", "after", "lunch"], TZ, now)
    assert note == "felt fluttering after lunch"
    assert when == datetime(2026, 9, 28, 6, 30, tzinfo=UTC)
    assert when.tzinfo is UTC


def test_parse_symptom_args_future_time_means_yesterday():
    now = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)  # 18:00 local
    when, note = parse_symptom_args(["21:15"], TZ, now)
    assert when == datetime(2026, 9, 27, 13, 15, tzinfo=UTC)
    assert note == ""


def test_parse_symptom_args_variants():
    now = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)
    assert parse_symptom_args(["1030", "x"], TZ, now)[0] == datetime(2026, 9, 28, 2, 30, tzinfo=UTC)
    assert parse_symptom_args(["9.05"], TZ, now)[0] == datetime(2026, 9, 28, 1, 5, tzinfo=UTC)
    assert parse_symptom_args(["felt", "it"], TZ, now) == (None, "felt it")
    assert parse_symptom_args([], TZ, now) == (None, "")
    with pytest.raises(ValueError):
        parse_symptom_args(["25:70", "bad"], TZ, now)


async def test_palp_with_time_and_note(bot, service):
    upd = make_update(USER_CHAT, "/palp 14:30 felt fluttering")
    await bot.cmd_palp(upd, make_context(["14:30", "felt", "fluttering"]))
    profile, when, note, chat_id = service.called("log_symptom")[0]
    assert profile.name == "Dad"
    assert note == "felt fluttering"
    assert chat_id == USER_CHAT
    assert when is not None and when.tzinfo is not None
    local = when.astimezone(profile_tz())
    assert (local.hour, local.minute) == (14, 30)
    # never in the future
    assert when <= datetime.now(UTC) + timedelta(minutes=2)
    assert "Logged" in replies(upd)[0]


async def test_palp_without_time_uses_now(bot, service):
    upd = make_update(USER_CHAT, "/palp dizzy")
    await bot.cmd_palp(upd, make_context(["dizzy"]))
    profile, when, note, chat_id = service.called("log_symptom")[0]
    assert when is None
    assert note == "dizzy"


async def test_palp_with_profile_name_then_time(two_profile_bot, service):
    upd = make_update(111, "/palp Dad 08:00 on waking")
    await two_profile_bot.cmd_palp(upd, make_context(["Dad", "08:00", "on", "waking"]))
    profile, when, note, chat_id = service.called("log_symptom")[0]
    assert profile.name == "Dad"
    assert note == "on waking"
    assert when.astimezone(profile_tz()).hour == 8


async def test_palp_invalid_time_is_rejected(bot, service):
    upd = make_update(USER_CHAT, "/palp 99:99 x")
    await bot.cmd_palp(upd, make_context(["99:99", "x"]))
    assert service.calls == []
    assert "HH:MM" in replies(upd)[0]


async def test_note_requires_text(bot, service):
    upd = make_update(USER_CHAT, "/note")
    await bot.cmd_note(upd, make_context())
    assert service.calls == []
    assert "note" in replies(upd)[0].lower()


async def test_note_logs_symptom_with_no_time(bot, service):
    upd = make_update(USER_CHAT, "/note short of breath")
    await bot.cmd_note(upd, make_context(["short", "of", "breath"]))
    profile, when, note, chat_id = service.called("log_symptom")[0]
    assert when is None and note == "short of breath"


def profile_tz():
    from zoneinfo import ZoneInfo

    return ZoneInfo(TZ)


# ---------------------------------------------------------------------------
# /report
# ---------------------------------------------------------------------------


async def test_report_sends_two_documents(bot, service):
    upd = make_update(USER_CHAT, "/report")
    await bot.cmd_report(upd, make_context())
    profile, days = service.called("doctor_report")[0]
    assert days == 30
    assert "Generating" in replies(upd)[0]
    assert upd.message.reply_document.await_count == 2
    first, second = upd.message.reply_document.await_args_list
    assert first.kwargs["document"].filename.endswith(".pdf")
    assert second.kwargs["document"].filename.endswith(".csv")
    assert "Doctor report" in first.kwargs["caption"]
    assert "3 detected episode" in first.kwargs["caption"]
    assert (
        "not a diagnosis" in first.kwargs["caption"].lower()
        or "diagnos" in first.kwargs["caption"].lower()
    )


async def test_report_days_capped(bot, service):
    upd = make_update(USER_CHAT, "/report 9999")
    await bot.cmd_report(upd, make_context(["9999"]))
    assert service.called("doctor_report")[0][1] == 365


async def test_report_failure_replies_once(bot, service):
    service.fail.add("doctor_report")
    upd = make_update(USER_CHAT, "/report")
    await bot.cmd_report(upd, make_context())
    upd.message.reply_document.assert_not_awaited()
    assert any("Sorry" in t for t in replies(upd))


# ---------------------------------------------------------------------------
# /status and /mfa
# ---------------------------------------------------------------------------


async def test_status_admin_uses_service(bot, service):
    upd = make_update(ADMIN_CHAT, "/status")
    await bot.cmd_status(upd, make_context())
    assert service.called("status_text")
    assert "Garmin Health Monitor" in replies(upd)[0]


async def test_status_for_normal_chat_is_short(bot, service):
    upd = make_update(USER_CHAT, "/status")
    await bot.cmd_status(upd, make_context())
    assert not service.called("status_text")
    assert "Dad" in replies(upd)[0]


async def test_mfa_command_delivers_code_to_waiting_thread(bot):
    result: dict[str, str] = {}

    def worker():
        result["code"] = bot.mfa.wait_for_code("Dad", timeout=5)

    t = threading.Thread(target=worker)
    t.start()
    for _ in range(100):
        if bot.mfa.pending():
            break
        threading.Event().wait(0.01)
    upd = make_update(ADMIN_CHAT, "/mfa 123456")
    await bot.cmd_mfa(upd, make_context(["123456"]))
    t.join(timeout=5)
    assert result["code"] == "123456"
    assert "received" in replies(upd)[0].lower()


async def test_mfa_command_when_nothing_pending(bot):
    upd = make_update(ADMIN_CHAT, "/mfa 123456")
    await bot.cmd_mfa(upd, make_context(["123456"]))
    assert "no garmin login" in replies(upd)[0].lower()


async def test_mfa_command_without_code(bot):
    upd = make_update(ADMIN_CHAT, "/mfa")
    await bot.cmd_mfa(upd, make_context())
    assert "/mfa" in replies(upd)[0]


# ---------------------------------------------------------------------------
# MfaBroker
# ---------------------------------------------------------------------------


def test_mfa_broker_roundtrip_with_thread():
    broker = MfaBroker()
    got: dict[str, str] = {}

    def worker():
        got["code"] = broker.wait_for_code("Dad", timeout=5)

    t = threading.Thread(target=worker)
    t.start()
    for _ in range(200):
        if broker.pending() == ["Dad"]:
            break
        threading.Event().wait(0.01)
    assert broker.pending() == ["Dad"]
    assert broker.submit("Dad", " 654321 ") is True
    t.join(timeout=5)
    assert got["code"] == "654321"
    assert broker.pending() == []


def test_mfa_broker_submit_without_pending_is_false():
    broker = MfaBroker()
    assert broker.submit("Dad", "123") is False
    assert broker.submit(None, "123") is False
    assert broker.submit("Dad", "") is False


def test_mfa_broker_single_pending_accepts_any_name():
    broker = MfaBroker()
    got: dict[str, str] = {}

    def worker():
        got["code"] = broker.wait_for_code("Dad", timeout=5)

    t = threading.Thread(target=worker)
    t.start()
    for _ in range(200):
        if broker.pending():
            break
        threading.Event().wait(0.01)
    assert broker.submit(None, "111111") is True
    t.join(timeout=5)
    assert got["code"] == "111111"


def test_mfa_broker_timeout_raises():
    broker = MfaBroker()
    with pytest.raises(TimeoutError):
        broker.wait_for_code("Dad", timeout=0.05)
    assert broker.pending() == []


# ---------------------------------------------------------------------------
# Felt callback
# ---------------------------------------------------------------------------


def test_build_felt_keyboard():
    kb = build_felt_keyboard(42)
    assert isinstance(kb, InlineKeyboardMarkup)
    datas = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert datas == ["felt:42:yes", "felt:42:no"]
    assert all(FELT_CALLBACK_RE.match(d) for d in datas)


async def test_felt_callback_records_and_edits_markup(bot, service):
    upd = make_callback_update(USER_CHAT, "felt:42:yes", text="<b>Alert</b> 10:00")
    await bot.cb_felt(upd, make_context())
    assert service.called("set_felt") == [(None, 42, True)]
    q = upd.callback_query
    q.answer.assert_awaited_once()
    q.edit_message_text.assert_awaited_once()
    kwargs = q.edit_message_text.await_args.kwargs
    assert kwargs["reply_markup"] is None
    assert kwargs["text"].startswith("<b>Alert</b> 10:00")
    assert "recorded: felt" in kwargs["text"]


async def test_felt_callback_no_on_photo_edits_caption(bot, service):
    upd = make_callback_update(USER_CHAT, "felt:7:no", text=None, caption="Chart caption")
    await bot.cb_felt(upd, make_context())
    assert service.called("set_felt") == [(None, 7, False)]
    q = upd.callback_query
    q.edit_message_caption.assert_awaited_once()
    kwargs = q.edit_message_caption.await_args.kwargs
    assert kwargs["reply_markup"] is None
    assert "recorded: not noticed" in kwargs["caption"]
    q.edit_message_text.assert_not_awaited()


async def test_felt_callback_falls_back_to_removing_buttons(bot, service):
    upd = make_callback_update(USER_CHAT, "felt:7:yes")
    upd.callback_query.edit_message_text.side_effect = BadRequest("Message can't be edited")
    await bot.cb_felt(upd, make_context())
    upd.callback_query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)


async def test_felt_callback_service_error(bot, service):
    service.fail.add("set_felt")
    upd = make_callback_update(USER_CHAT, "felt:7:yes")
    await bot.cb_felt(upd, make_context())
    upd.callback_query.answer.assert_awaited_once()
    assert "Sorry" in upd.callback_query.answer.await_args.args[0]
    upd.callback_query.edit_message_text.assert_not_awaited()


# ---------------------------------------------------------------------------
# Outgoing helpers
# ---------------------------------------------------------------------------


def test_chunk_text_short_and_empty():
    assert chunk_text("") == []
    assert chunk_text("hello") == ["hello"]


def test_chunk_text_splits_on_lines_and_balances_tags():
    lines = [f"<b>Line {i}</b> " + "x" * 60 for i in range(150)]
    text = "\n".join(lines)
    chunks = chunk_text(text, 4000)
    assert len(chunks) > 1
    assert all(len(c) <= 4000 for c in chunks)
    assert all(c.count("<b>") == c.count("</b>") for c in chunks)
    # nothing lost: every line is present somewhere
    joined = "".join(chunks)
    assert all(f"Line {i}" in joined for i in range(150))
    # boundaries fall on line starts
    assert all(c.startswith("<b>Line") for c in chunks)


def test_chunk_text_reopens_tag_across_boundary():
    text = "<i>" + "word " * 100 + "</i>"
    chunks = chunk_text(text, 120)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 120
        assert c.startswith("<i>") and c.endswith("</i>")


async def test_send_text_chunks_and_uses_thread(bot):
    fake = AsyncMock(name="tgbot")
    bot.app.bot = fake
    text = "\n".join(f"line {i:03d} {'y' * 40}" for i in range(300))
    assert len(text) > 8000
    await bot.send_text([111, 222], text)
    calls = fake.send_message.await_args_list
    per_chat = {}
    for c in calls:
        per_chat.setdefault(c.kwargs["chat_id"], []).append(c.kwargs["text"])
    assert set(per_chat) == {111, 222}
    for parts in per_chat.values():
        assert len(parts) >= 3
        assert all(len(p) <= 4000 for p in parts)
        assert "".join(parts).count("line ") == 300
    assert all(c.kwargs["parse_mode"] == "HTML" for c in calls)


async def test_send_text_keyboard_only_on_last_chunk(bot):
    fake = AsyncMock(name="tgbot")
    bot.app.bot = fake
    kb = build_felt_keyboard(1)
    text = "\n".join(f"row {i}" for i in range(2000))
    await bot.send_text(111, text, reply_markup=kb)
    markups = [c.kwargs["reply_markup"] for c in fake.send_message.await_args_list]
    assert markups[-1] is kb
    assert all(m is None for m in markups[:-1])


async def test_send_text_html_parse_failure_retries_plain(bot):
    fake = AsyncMock(name="tgbot")
    fake.send_message.side_effect = [BadRequest("Can't parse entities: unclosed tag"), None]
    bot.app.bot = fake
    await bot.send_text([111], "<b>broken")
    assert fake.send_message.await_count == 2
    second = fake.send_message.await_args_list[1].kwargs
    assert second["parse_mode"] is None
    assert "<b>" not in second["text"]


async def test_send_text_one_failing_chat_does_not_stop_others(bot):
    fake = AsyncMock(name="tgbot")
    fake.send_message.side_effect = [BadRequest("chat not found"), None]
    bot.app.bot = fake
    await bot.send_text([111, 222], "hello")
    chats = [c.kwargs["chat_id"] for c in fake.send_message.await_args_list]
    assert chats == [111, 222]


async def test_send_text_uses_forum_thread(tmp_path, service):
    cfg = make_app_config(tmp_path)
    cfg.profiles[0].telegram_threads = {111: 77}
    hb = HealthBot(cfg, service)
    fake = AsyncMock(name="tgbot")
    hb.app.bot = fake
    await hb.send_text([111], "hi")
    assert fake.send_message.await_args.kwargs["message_thread_id"] == 77


async def test_send_photo_and_document(bot, tmp_path):
    fake = AsyncMock(name="tgbot")
    bot.app.bot = fake
    await bot.send_photo([111], b"png-bytes", "<b>cap</b>")
    fake.send_photo.assert_awaited_once()
    assert fake.send_photo.await_args.kwargs["caption"] == "<b>cap</b>"
    path = tmp_path / "r.pdf"
    path.write_bytes(b"%PDF")
    await bot.send_document([111, 222], path, "doc")
    assert fake.send_document.await_count == 2
    assert fake.send_document.await_args.kwargs["document"].filename == "r.pdf"


async def test_send_photo_long_caption_becomes_text(bot):
    fake = AsyncMock(name="tgbot")
    bot.app.bot = fake
    kb = build_felt_keyboard(3)
    await bot.send_photo([111], b"png", "c" * 2000, reply_markup=kb)
    fake.send_photo.assert_awaited_once()
    assert "caption" not in fake.send_photo.await_args.kwargs
    fake.send_message.assert_awaited_once()
    assert fake.send_message.await_args.kwargs["reply_markup"] is kb


async def test_send_photo_without_png_falls_back_to_text(bot):
    fake = AsyncMock(name="tgbot")
    bot.app.bot = fake
    await bot.send_photo([111], b"", "text only")
    fake.send_photo.assert_not_awaited()
    fake.send_message.assert_awaited_once()


async def test_notify_episode_photo_with_buttons(bot, episode_day):
    from garmin_health_monitor.palpitations import detect_episodes

    eps = detect_episodes(episode_day, bot.config.profiles[0].palpitations)
    assert eps
    ep = eps[0]
    ep.id = 12
    fake = AsyncMock(name="tgbot")
    bot.app.bot = fake
    await bot.notify_episode(bot.config.profiles[0], ep, "<b>Episode</b>", b"png")
    fake.send_photo.assert_awaited_once()
    kw = fake.send_photo.await_args.kwargs
    assert kw["chat_id"] == 111
    datas = [b.callback_data for row in kw["reply_markup"].inline_keyboard for b in row]
    assert datas == ["felt:12:yes", "felt:12:no"]
    # without a chart -> text with buttons
    await bot.notify_episode(bot.config.profiles[0], ep, "<b>Episode</b>", None)
    assert fake.send_message.await_args.kwargs["reply_markup"] is not None


async def test_unknown_command(bot):
    upd = make_update(USER_CHAT, "/whatever")
    await bot.cmd_unknown(upd, make_context())
    assert "/help" in replies(upd)[0]
    other = make_update(USER_CHAT, "/whatever@OtherBot")
    bot.app.bot = MagicMock(username="HealthBot")
    await bot.cmd_unknown(other, make_context())
    other.message.reply_text.assert_not_awaited()


def test_help_text_mentions_every_registered_command(bot):
    from garmin_health_monitor import messages

    text = messages.help_text(True)
    handlers = [h for group in bot.app.handlers.values() for h in group]
    names = {n for h in handlers for n in getattr(h, "commands", [])}
    for cmd in names - {"start", "analyse"}:
        assert f"/{cmd}" in text, cmd
