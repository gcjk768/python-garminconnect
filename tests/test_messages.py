"""Tests for the Telegram message renderers in ``garmin_health_monitor.messages``."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from garmin_health_monitor import messages as M
from garmin_health_monitor.models import (
    ASSESSMENT_EXERTION,
    ASSESSMENT_POSSIBLE,
    Alert,
    CoachingAdvice,
    DaySnapshot,
    DaySummary,
    Episode,
    EpisodeAssessment,
    SymptomReport,
)
from garmin_health_monitor.palpitations import detect_episodes
from tests.conftest import DAY, TZ, make_profile, make_snapshot

ALLOWED_TAG_RE = re.compile(r"</?(b|i|code|pre)>|<a href=\"[^\"]*\">|</a>")


def _strip_allowed_tags(text: str) -> str:
    return ALLOWED_TAG_RE.sub("", text)


def _assert_valid_html(text: str) -> None:
    """Only the allowed Telegram tags remain; no raw ``<``/``>`` from dynamic values."""
    stripped = _strip_allowed_tags(text)
    assert "<" not in stripped, f"unescaped '<' in message: {stripped!r}"
    assert ">" not in stripped, f"unescaped '>' in message: {stripped!r}"
    for tag in ("b", "i", "code", "pre"):
        assert text.count(f"<{tag}>") == text.count(f"</{tag}>"), f"unbalanced <{tag}> tags"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(day, **kw):
    base = {
        "profile": "Dad",
        "day": day.isoformat(),
        "total_steps": 5000,
        "step_goal": 6000,
        "resting_hr": 58,
        "resting_hr_7d_avg": 57,
        "sleep_seconds": 7 * 3600,
        "sleep_score": 75,
        "hrv_last_night": 42.0,
        "avg_stress": 30,
        "body_battery_high": 80,
        "activities": [],
    }
    base.update(kw)
    return base


@pytest.fixture
def rows_7d():
    return [_row(DAY - timedelta(days=i), total_steps=4000 + 200 * i, resting_hr=56 + (i % 3)) for i in range(7, 0, -1)]


@pytest.fixture
def rows_prev_week():
    return [_row(DAY - timedelta(days=i), total_steps=3500, resting_hr=59) for i in range(14, 7, -1)]


@pytest.fixture
def episodes(episode_day):
    eps = detect_episodes(episode_day)
    assert len(eps) == 2
    eps[0].id = 1
    eps[0].felt = True
    eps[0].llm_assessment = ASSESSMENT_POSSIBLE
    eps[0].doctor_note = "8 min at 127 bpm while seated."
    eps[1].id = 2
    eps[1].felt = False
    return eps


@pytest.fixture
def coaching():
    return CoachingAdvice(
        summary="A steady day with a short walk.",
        do_more=["Take a second short walk after lunch", "Keep the 22:30 bedtime"],
        do_less=["Long sitting stretches"],
        watch_outs=["Resting heart rate slightly above the weekly average"],
        heart_note="Two at-rest heart-rate rises today; both are in the diary.",
        model="llama3.1:8b",
    )


@pytest.fixture
def assessment():
    return EpisodeAssessment(
        assessment=ASSESSMENT_POSSIBLE,
        confidence=0.7,
        reasoning="Heart rate rose quickly while the watch recorded almost no steps.",
        doctor_note="8 min at up to 127 bpm while seated (baseline 64).",
        model="llama3.1:8b",
    )


@pytest.fixture
def symptom():
    return SymptomReport(
        profile="Dad",
        reported_at=datetime(2026, 9, 27, 6, 12, tzinfo=UTC),
        event_time=datetime(2026, 9, 27, 6, 10, tzinfo=UTC),  # 14:10 Singapore
        note="fluttering after lunch <3 & more",
        hr_at_time=98,
        baseline_hr=64.0,
        id=1,
    )


@pytest.fixture
def empty_snapshot():
    """A snapshot with nothing but a profile/day: every optional block missing."""
    return DaySnapshot(profile="Dad", day=DAY, tz=TZ, fetched_at=datetime(2026, 9, 27, 1, 0, tzinfo=UTC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_esc_escapes_html_and_handles_none():
    assert M.esc("<Dad&Co>") == "&lt;Dad&amp;Co&gt;"
    assert M.esc(None) == "n/a"
    assert M.esc(42) == "42"
    assert M.esc("Dad's") == "Dad's"  # apostrophes stay readable


def test_confidence_label():
    assert M.confidence_label(0.1) == "Low"
    assert M.confidence_label(0.5) == "Medium"
    assert M.confidence_label(0.9) == "High"
    assert M.confidence_label(None) == "n/a"


def test_coaching_block_lists_and_disclaimer(coaching):
    text = M.coaching_block(coaching)
    assert "🧭" in text
    assert "Do more" in text and "Do less" in text and "Watch out" in text
    assert "Take a second short walk after lunch" in text
    assert "Long sitting stretches" in text
    assert "llama3.1:8b" in text
    assert "not medical advice" in text
    assert M.coaching_block(None) == ""


def test_coaching_block_tolerates_odd_shapes():
    advice = CoachingAdvice(summary="", do_more=None, do_less="one string", watch_outs=[None, "  "], heart_note=None, model="")
    text = M.coaching_block(advice)
    assert "one string" in text
    assert "Watch out" not in text
    _assert_valid_html(text)


# ---------------------------------------------------------------------------
# Scheduled messages
# ---------------------------------------------------------------------------


def test_morning_brief_content(profile, quiet_day, rows_7d, coaching):
    text = M.morning_brief(profile, quiet_day, rows_7d[-1], [], coaching)
    assert text.startswith("🌅 <b>Good morning, Dad</b>")
    assert "Slept 7h 40m (22:30–06:30)" in text
    assert "score 78 (Good)" in text
    assert "Deep 1h 36m" in text and "REM" in text and "Awake 20m" in text
    assert "HRV overnight: 42 ms (weekly avg 44 ms, ▼2 ms) — Balanced" in text
    assert "Resting HR: 58 bpm (7-day avg 57, ▲1)" in text
    assert "Body battery on waking" in text
    assert "Training readiness: 62 (Moderate)" in text
    assert "Lowest SpO2 overnight: 92%" in text
    assert "Yesterday:" in text
    assert "No at-rest heart-rate excursions overnight" in text
    assert "Coaching for today" in text
    _assert_valid_html(text)


def test_morning_brief_lists_overnight_episodes(profile, episode_day, episodes):
    text = M.morning_brief(profile, episode_day, None, episodes[:1], None)
    assert "Overnight / early-morning heart-rate excursions: 1" in text
    assert "10:00–10:08" in text
    assert "peak 127 bpm" in text
    assert "✅ felt" in text
    assert "Coaching" not in text


def test_morning_brief_without_any_data(profile, empty_snapshot):
    text = M.morning_brief(profile, empty_snapshot, None, None, None)
    assert "No sleep data yet" in text
    assert "HRV overnight: n/a" in text
    assert "Resting HR: n/a" in text
    assert "Body battery on waking: n/a" in text
    assert "Training readiness: n/a" in text
    assert "SpO2 overnight: n/a" in text
    _assert_valid_html(text)


def test_morning_brief_falls_back_to_yesterday_rhr(profile, empty_snapshot):
    text = M.morning_brief(profile, empty_snapshot, {"resting_hr": 61, "resting_hr_7d_avg": 58}, [], None)
    assert "Resting HR: 61 bpm (yesterday) (7-day avg 58, ▲3)" in text


def test_evening_summary_content(profile, episode_day, rows_7d, episodes, symptom, coaching):
    text = M.evening_summary(profile, episode_day, rows_7d, episodes, [symptom], coaching)
    assert text.startswith("🌙 <b>Evening summary — Dad</b>")
    assert "▰▰▰▰▰▰▱▱▱▱ 3,757 / 6,000 (63%)" in text
    assert "7-day avg" in text and ("▲" in text or "▼" in text)
    assert "Distance 2.6 km" in text and "Floors 6" in text and "1,900 kcal" in text
    assert "Intensity minutes 35" in text
    assert "Stress avg 28" in text and "high stress 20m" in text
    assert "Body battery high 85" in text and "low 25" in text
    assert "Morning Walk · 30m · avg HR 105 bpm" in text
    assert "At-rest heart-rate excursions today: 2" in text
    assert "10:00–10:08" in text and "15:30–15:36" in text
    assert "✅ felt" in text and "❌ not noticed" in text
    assert "possible palpitation" in text
    assert "Reported symptoms today: 1" in text
    assert "14:10" in text and "fluttering after lunch &lt;3 &amp; more" in text
    assert "Coaching for today" in text
    _assert_valid_html(text)


def test_evening_summary_hides_palpitation_section_when_feature_off(quiet_day, rows_7d):
    profile = make_profile(palpitations=False)
    text = M.evening_summary(profile, quiet_day, rows_7d, [], [], None)
    assert "excursions" not in text
    assert "symptoms" not in text.lower()
    assert "No recorded activities today" in text


def test_evening_summary_unanswered_marker_and_empty_lists(profile, episode_day, episodes):
    for ep in episodes:
        ep.felt = None
    text = M.evening_summary(profile, episode_day, [], episodes, [], None)
    assert "❓ not answered" in text
    assert "No symptoms reported today" in text
    assert "7-day avg" not in text.split("Stress")[0]  # no history -> no steps delta


def test_evening_summary_without_data(profile, empty_snapshot):
    text = M.evening_summary(profile, empty_snapshot, None, None, None, None)
    assert "▱▱▱▱▱▱▱▱▱▱ n/a / n/a" in text
    assert "Sleep last night n/a" in text
    _assert_valid_html(text)


def test_weekly_review_table_and_deltas(profile, rows_7d, rows_prev_week, episodes, coaching):
    coaching.period = "weekly"
    text = M.weekly_review(profile, rows_7d, rows_prev_week, episodes, coaching)
    assert text.startswith("📊 <b>Weekly review — Dad</b>")
    assert "<pre>" in text and "</pre>" in text
    assert "Day     Steps Sleep  RHR  Str   BB" in text
    assert "Sat 26" in text  # last row of the 7-day window
    assert "Steps total 33,600 (prev 24,500, ▲9,100)" in text
    assert "Sleep avg 7.0h (prev 7.0h, ±0.0h)" in text
    assert "Resting HR avg 57 bpm (prev 59, ▼2)" in text
    assert "At-rest heart-rate excursions this week: 2" in text
    assert "Sun 27 ×2" in text
    assert "Coaching for this week" in text
    _assert_valid_html(text)


def test_weekly_review_without_rows(profile):
    text = M.weekly_review(profile, [], [], [], None)
    assert "No daily data stored for this week yet" in text
    assert "Steps total n/a" in text
    assert "No at-rest heart-rate excursions this week" in text
    _assert_valid_html(text)


# ---------------------------------------------------------------------------
# Interactive commands
# ---------------------------------------------------------------------------


def test_today_status_live_snapshot(profile, episode_day, episodes):
    now = datetime(2026, 9, 27, 16, 5, tzinfo=UTC)  # 00:05 local, 35 min after the last sync
    text = M.today_status(profile, episode_day, episodes, now=now)
    assert "Right now — Dad" in text
    assert "Steps so far: ▰▰▰▰▰▰▱▱▱▱ 3,757 / 6,000" in text
    assert "Body battery: 5 (at 23:57)" in text
    assert "Last HR: 67 bpm at 23:58 · resting 58" in text
    assert "Stress avg: 28" in text
    assert "Morning Walk" in text
    assert "At-rest heart-rate excursions today: 2" in text
    assert "Last sync: Sun 27 Sep 23:30 (Venu 3)" in text
    assert "has not synced" not in text


def test_today_status_warns_when_sync_is_stale(profile, episode_day):
    now = datetime(2026, 9, 27, 21, 5, tzinfo=UTC)  # 5h35m after the last sync
    text = M.today_status(profile, episode_day, [], now=now)
    assert "⚠️ Watch has not synced for 5h 35m" in text


def test_today_status_without_data(profile, empty_snapshot):
    text = M.today_status(profile, empty_snapshot, None, now=datetime(2026, 9, 27, 4, 0, tzinfo=UTC))
    assert "Steps so far: ▱▱▱▱▱▱▱▱▱▱ n/a / n/a" in text
    assert "Body battery: n/a" in text
    assert "Last HR: n/a" in text
    assert "Last sync: n/a" in text
    assert "No sync time reported" in text
    _assert_valid_html(text)


def test_sleep_message(profile, quiet_day):
    text = M.sleep_message(profile, quiet_day)
    assert "😴 <b>Sleep — Dad</b>" in text
    assert "Bed 22:30 → wake 06:30 · 7h 40m asleep" in text
    assert "Score 78 (Good)" in text
    assert "Deep 1h 36m (21%)" in text and "Light 4h 00m (52%)" in text and "REM 1h 45m (23%)" in text
    assert "Woke 2 times · restless moments 18" in text
    assert "Sleeping HR 56 bpm · HRV 42 ms (Balanced)" in text
    assert "SpO2 avg 95% · lowest 91%" in text
    assert "Body battery +55 overnight" in text


def test_sleep_message_without_sleep(profile):
    snap = make_snapshot(profile="Dad", sleep=None)
    text = M.sleep_message(profile, snap)
    assert "No sleep data for this night" in text


def test_hr_message_lists_excursions_and_fits_caption(profile, episode_day, episodes):
    text = M.hr_message(profile, episode_day, episodes)
    assert "❤️ <b>Heart rate — Dad</b>" in text
    assert "Resting 58 bpm (7-day avg 57, ▲1)" in text
    assert "Min 50 · Max 127" in text
    assert "Watch abnormal-HR alerts: 2" in text
    assert "At-rest excursions: 2" in text
    assert "10:00–10:08 · 8m · peak 127 (base 64, +63) · sustained rise · ✅ felt" in text
    assert len(text) <= M.MAX_CAPTION_LEN


def test_hr_message_without_episodes(profile, quiet_day):
    text = M.hr_message(profile, quiet_day, None)
    assert "At-rest excursions: none detected" in text


def test_steps_message(profile, episode_day, rows_7d):
    text = M.steps_message(profile, episode_day, rows_7d)
    assert "👟 <b>Steps — Dad</b>" in text
    assert "▰▰▰▰▰▰▱▱▱▱ 3,757 / 6,000 (63%)" in text
    assert "7-day avg 4,800 · today ▼1,043" in text
    assert "2,243 more to reach the goal" in text
    assert "<pre>" in text and "Goal met on 0 of the last 7 days" in text


def test_steps_message_without_history(profile, empty_snapshot):
    text = M.steps_message(profile, empty_snapshot, [])
    assert "7-day avg n/a" in text
    assert "<pre>" not in text
    _assert_valid_html(text)


# ---------------------------------------------------------------------------
# Palpitation diary
# ---------------------------------------------------------------------------


def test_episode_alert_content(profile, episodes, assessment):
    ep = episodes[0]
    text = M.episode_alert(profile, ep, assessment)
    assert text.startswith("❤️ <b>Possible palpitation episode (Dad)</b>")
    assert "Sun 27 Sep 10:00–10:08 · 8m" in text
    assert "Peak 127 bpm vs baseline 64 bpm (+63)" in text
    assert "Onset jump: +60 bpm" in text
    assert "Context: at rest, 40 steps in the 15-min window" in text
    assert "Heuristic confidence: High (0.75)" in text
    assert "Model view: <b>possible palpitation</b> (confidence 0.70)" in text
    assert assessment.reasoning in text
    assert "Doctor note: 8 min at up to 127 bpm" in text
    assert "<b>Was it felt?</b> Tap a button below — it goes into the doctor report." in text
    assert "not a diagnosis" in text
    assert "chest pain, fainting or breathlessness" in text
    _assert_valid_html(text)


def test_episode_alert_without_assessment_uses_stored_fields(profile, episodes):
    ep = episodes[1]
    ep.llm_assessment = None
    text = M.episode_alert(profile, ep, None)
    assert "Model view" not in text
    assert "Was it felt?" in text
    ep.llm_assessment = ASSESSMENT_EXERTION
    ep.llm_model = "rules"
    text = M.episode_alert(profile, ep, None)
    assert "Model view: <b>likely exertion or movement</b>" in text


def test_episode_alert_nocturnal_context(profile):
    ep = Episode(
        profile="Dad",
        start=datetime(2026, 9, 26, 19, 0, tzinfo=UTC),
        end=datetime(2026, 9, 26, 19, 6, tzinfo=UTC),
        duration_min=6.0,
        peak_hr=96,
        mean_hr=94.0,
        baseline_hr=54.0,
        delta_hr=42.0,
        asleep=True,
        kind="nocturnal",
        confidence=0.3,
    )
    text = M.episode_alert(profile, ep, None)
    assert "Context: asleep" in text
    assert "type: during sleep" in text
    assert "Heuristic confidence: Low (0.30)" in text


def test_episodes_list_content(profile, episodes, symptom):
    text = M.episodes_list(profile, episodes, [symptom], 7)
    assert "🩺 <b>Palpitation diary — Dad</b> — last 7 days" in text
    assert "Detected episodes: 2" in text
    assert "Sun 27 Sep 10:00–10:08 · 8m · peak 127 bpm · ✅ felt · possible palpitation" in text
    assert "Sun 27 Sep 15:30–15:36 · 6m · peak 114 bpm · ❌ not noticed · not assessed yet" in text
    assert "Reported symptoms: 1" in text
    assert "Sun 27 Sep 14:10 — “fluttering after lunch &lt;3 &amp; more” · HR 98 bpm" in text
    assert "about 2.0 per week" in text
    assert "most common hour" in text
    assert "felt 1, not noticed 1, unanswered 0" in text
    assert "/report 30" in text
    _assert_valid_html(text)


def test_episodes_list_empty(profile):
    text = M.episodes_list(profile, [], [], 14)
    assert "Nothing recorded in the last 14 days" in text
    assert "/palp HH:MM note" in text


def test_episodes_list_truncates_long_lists(profile, episodes):
    many = []
    for i in range(120):
        ep = Episode(
            profile="Dad",
            start=datetime(2026, 9, 1, 2, 0, tzinfo=UTC) + timedelta(hours=5 * i),
            end=datetime(2026, 9, 1, 2, 8, tzinfo=UTC) + timedelta(hours=5 * i),
            duration_min=8.0,
            peak_hr=120,
            mean_hr=115.0,
            baseline_hr=60.0,
            delta_hr=60.0,
        )
        many.append(ep)
    syms = [
        SymptomReport(profile="Dad", reported_at=e.start, event_time=e.start, note="x" * 40, hr_at_time=100)
        for e in many[:40]
    ]
    text = M.episodes_list(profile, many, syms, 30)
    assert len(text) < 4000
    assert "… and 100 more" in text
    assert "… and 30 more" in text
    assert "about 28.0 per week" in text
    _assert_valid_html(text)


def test_symptom_logged(profile, symptom):
    text = M.symptom_logged(profile, symptom)
    assert "📝 <b>Symptom logged — Dad</b>" in text
    assert "Sun 27 Sep 14:10" in text
    assert "Note: “fluttering after lunch &lt;3 &amp; more”" in text
    assert "Heart rate near that time: 98 bpm (at-rest baseline 64, ▲34)" in text
    assert "doctor report" in text


def test_symptom_logged_without_hr(profile):
    rep = SymptomReport(profile="Dad", reported_at=datetime(2026, 9, 27, 6, 0, tzinfo=UTC), event_time=datetime(2026, 9, 27, 6, 0, tzinfo=UTC), episode_id=7)
    text = M.symptom_logged(profile, rep)
    assert "Heart rate near that time: n/a" in text
    assert "Linked to detected episode #7" in text
    assert "Note:" not in text


@pytest.mark.parametrize("severity,emoji", [("info", "ℹ️"), ("warning", "⚠️"), ("critical", "🚨"), ("weird", "ℹ️")])
def test_alert_message(severity, emoji):
    alert = Alert(key="k", severity=severity, title="Resting HR <high>", body="RHR 68 & rising", profile="Dad")
    text = M.alert_message(alert)
    assert text.startswith(f"{emoji} <b>Resting HR &lt;high&gt; (Dad)</b>")
    assert "RHR 68 &amp; rising" in text


def test_doctor_report_caption(profile):
    text = M.doctor_report_caption(profile, 30, 12, 3)
    assert "🩺 <b>Doctor report — Dad</b> — last 30 days" in text
    assert "12 detected episode(s), 3 reported symptom(s)" in text
    assert "not a diagnosis" in text
    assert len(text) <= M.MAX_CAPTION_LEN


def test_help_text_lists_commands_and_admin_extras():
    user = M.help_text(False)
    admin = M.help_text(True)
    for cmd in ("/today", "/yesterday", "/sleep", "/hr", "/steps", "/episodes", "/palp", "/note", "/report", "/analyze", "/profiles", "/help"):
        assert cmd in user
    assert "/status" not in user and "/mfa" not in user
    assert "/status" in admin and "/mfa &lt;code&gt;" in admin
    assert "<code>/today Dad</code>" in admin
    _assert_valid_html(user)
    _assert_valid_html(admin)


def test_mfa_request(profile):
    text = M.mfa_request(profile)
    assert "verification code (Dad)" in text
    assert "<code>/mfa &lt;code&gt;</code>" in text


# ---------------------------------------------------------------------------
# Cross-cutting: escaping, length, robustness
# ---------------------------------------------------------------------------


def _all_messages(profile, snap, rows, episodes, symptoms, coaching, assessment):
    return {
        "morning_brief": M.morning_brief(profile, snap, rows[-1] if rows else None, episodes, coaching),
        "evening_summary": M.evening_summary(profile, snap, rows, episodes, symptoms, coaching),
        "weekly_review": M.weekly_review(profile, rows, rows, episodes, coaching),
        "today_status": M.today_status(profile, snap, episodes),
        "sleep_message": M.sleep_message(profile, snap),
        "hr_message": M.hr_message(profile, snap, episodes),
        "steps_message": M.steps_message(profile, snap, rows),
        "episode_alert": M.episode_alert(profile, episodes[0], assessment) if episodes else "",
        "episodes_list": M.episodes_list(profile, episodes, symptoms, 7),
        "symptom_logged": M.symptom_logged(profile, symptoms[0]) if symptoms else "",
        "doctor_report_caption": M.doctor_report_caption(profile, 30, len(episodes), len(symptoms)),
        "mfa_request": M.mfa_request(profile),
    }


def test_profile_name_is_escaped_everywhere(episode_day, rows_7d, episodes, symptom, coaching, assessment):
    profile = make_profile(name="<Dad&Co>")
    coaching.summary = "Walk more <b>now</b> & rest"
    assessment.reasoning = "HR > baseline & steady"
    for name, text in _all_messages(profile, episode_day, rows_7d, episodes, [symptom], coaching, assessment).items():
        assert "<Dad&Co>" not in text, name
        assert "&lt;Dad&amp;Co&gt;" in text, name
        assert "<b>now</b>" not in text, name
        _assert_valid_html(text)


def test_every_message_stays_under_telegram_limit(profile, episode_day, rows_7d, episodes, symptom, coaching, assessment):
    coaching.do_more = ["a fairly long suggestion " * 8] * 10
    coaching.do_less = ["another long suggestion " * 8] * 10
    for name, text in _all_messages(profile, episode_day, rows_7d, episodes, [symptom], coaching, assessment).items():
        assert len(text) <= 4000, f"{name} is {len(text)} chars"
        assert len(text) <= M.MAX_LEN, f"{name} is {len(text)} chars"


def test_every_message_renders_with_empty_inputs(profile, empty_snapshot):
    empty_snapshot.summary = DaySummary()
    out = _all_messages(profile, empty_snapshot, [], [], [], None, None)
    for name, text in out.items():
        assert isinstance(text, str), name
        _assert_valid_html(text)
    assert "n/a" in out["morning_brief"]
    assert "n/a" in out["evening_summary"]
    assert "n/a" in out["today_status"]
    assert "n/a" in out["hr_message"]


def test_rows_with_missing_keys_do_not_crash(profile, quiet_day, coaching):
    rows = [{"day": "2026-09-20"}, {"day": None, "total_steps": "oops"}, {}]
    text = M.evening_summary(profile, quiet_day, rows, [], [], coaching)
    assert "Evening summary" in text
    text = M.weekly_review(profile, rows, rows, [], None)
    assert "Weekly review" in text
    text = M.steps_message(profile, quiet_day, rows)
    assert "Steps — Dad" in text


def test_assemble_truncates_without_breaking_tags():
    parts = [f"<b>line {i}</b> " + "x" * 50 for i in range(200)]
    text = M._assemble(parts, limit=1000)
    assert len(text) <= 1000
    assert text.endswith("… (message shortened)")
    _assert_valid_html(text)
