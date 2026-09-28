"""Tests for :mod:`garmin_health_monitor.analysis` with a fake Ollama client."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from garmin_health_monitor import analysis
from garmin_health_monitor.analysis import (
    COACHING_SCHEMA,
    EPISODE_SCHEMA,
    assess_episode,
    build_episode_prompt,
    build_history_payload,
    daily_coaching,
    doctor_narrative,
    episode_stats,
    fallback_doctor_note,
    parse_goals,
    rule_based_coaching,
    rule_based_narrative,
    weekly_review,
)
from garmin_health_monitor.models import (
    ASSESSMENT_POSSIBLE,
    ASSESSMENT_UNCLEAR,
    ASSESSMENTS,
    CoachingAdvice,
    DaySnapshot,
    EpisodeAssessment,
    SymptomReport,
)
from garmin_health_monitor.ollama_client import OllamaError
from garmin_health_monitor.palpitations import detect_episodes
from garmin_health_monitor.utils import now_utc

from .conftest import DAY, TZ, make_snapshot


class FakeClient:
    """Stand-in for OllamaClient: canned answers, records prompts."""

    def __init__(
        self,
        json_answer: dict[str, Any] | None = None,
        text_answer: str = "",
        error: Exception | None = None,
    ):
        self.model = "fake-model"
        self.json_answer = json_answer or {}
        self.text_answer = text_answer
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append({"kind": "json", "system": system, "user": user, "schema": schema})
        if self.error:
            raise self.error
        return self.json_answer

    def chat_text(self, system: str, user: str) -> str:
        self.calls.append({"kind": "text", "system": system, "user": user})
        if self.error:
            raise self.error
        return self.text_answer


@pytest.fixture
def rows_7d(storage):
    """Seven stored days (ascending), the last one being DAY."""
    for i in range(7):
        snap = make_snapshot(profile="Dad", day=DAY - timedelta(days=6 - i), seed=i + 1)
        storage.save_snapshot(snap)
    return storage.get_snapshot_rows("Dad", DAY - timedelta(days=6), DAY)


@pytest.fixture
def episodes(episode_day, profile):
    eps = detect_episodes(episode_day, profile.palpitations)
    assert len(eps) >= 2
    return eps


def _symptom(when, note="fluttering", hr=120):
    return SymptomReport(
        profile="Dad", reported_at=now_utc(), event_time=when, note=note, hr_at_time=hr
    )


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


def test_schemas_have_required_fields():
    assert set(COACHING_SCHEMA["required"]) == {
        "summary",
        "do_more",
        "do_less",
        "watch_outs",
        "heart_note",
    }
    assert COACHING_SCHEMA["properties"]["do_more"]["maxItems"] == 4
    assert set(EPISODE_SCHEMA["required"]) == {
        "assessment",
        "confidence",
        "reasoning",
        "doctor_note",
    }
    assert EPISODE_SCHEMA["properties"]["assessment"]["enum"] == list(ASSESSMENTS)
    json.dumps(COACHING_SCHEMA)
    json.dumps(EPISODE_SCHEMA)


# ---------------------------------------------------------------------------
# history payload
# ---------------------------------------------------------------------------


def test_build_history_payload_uses_rows_and_today(rows_7d, quiet_day):
    payload = build_history_payload(rows_7d, quiet_day)
    assert payload["today"]["day"] == DAY.isoformat()
    assert payload["today"]["steps"] == quiet_day.summary.total_steps
    assert payload["today"]["sleep_h"] == pytest.approx(27600 / 3600, abs=0.1)
    assert payload["today"]["activities"] == []
    # averages exclude today (the stored DAY row is replaced by the fresher snapshot)
    assert payload["averages"]["n_days"] == 6
    assert payload["averages"]["resting_hr"] == 58
    assert payload["averages"]["sleep_h"] == pytest.approx(7.7, abs=0.1)
    assert len(payload["days"]) == 7
    assert payload["days"][-1]["day"] == DAY.isoformat()
    json.dumps(payload)  # must be serialisable for the prompt


def test_build_history_payload_handles_empty_and_missing():
    payload = build_history_payload([], None)
    assert payload["today"] is None
    assert payload["days"] == []
    assert payload["averages"]["n_days"] == 0
    assert payload["averages"]["steps"] is None
    empty = DaySnapshot(profile="Dad", day=DAY, tz=TZ, fetched_at=now_utc())
    payload = build_history_payload([{"day": "2026-09-20", "total_steps": None}], empty)
    assert payload["today"]["steps"] is None
    assert payload["today"]["sleep_h"] is None
    assert payload["averages"]["n_days"] == 1
    # a row with the same day as today is superseded by the snapshot
    payload = build_history_payload([{"day": DAY.isoformat(), "total_steps": 99}], empty)
    assert payload["averages"]["n_days"] == 1  # falls back to today's own numbers
    assert len(payload["days"]) == 1


def test_build_history_payload_latest_without_today(rows_7d):
    payload = build_history_payload(rows_7d, None)
    assert payload["today"] is None
    assert payload["latest"]["day"] == DAY.isoformat()
    assert payload["averages"]["n_days"] == 7


# ---------------------------------------------------------------------------
# rule-based coaching
# ---------------------------------------------------------------------------


def test_parse_goals():
    assert parse_goals("6000 steps, 7h sleep") == (6000, 7.0)
    assert parse_goals("walk 6,000 steps a day and sleep 7.5 hours") == (6000, 7.5)
    assert parse_goals("8k steps") == (8000, None)
    assert parse_goals("") == (None, None)
    assert parse_goals(None) == (None, None)


def test_rule_based_coaching_quiet_day(profile, rows_7d, quiet_day):
    adv = rule_based_coaching(profile, rows_7d, quiet_day)
    assert isinstance(adv, CoachingAdvice)
    assert adv.model == "rules" and adv.period == "daily"
    assert 2 <= len(adv.do_more) <= 4
    assert 2 <= len(adv.do_less) <= 4
    assert f"{quiet_day.summary.total_steps:,}" in adv.summary
    assert "6,000" in adv.summary  # goal from profile.goals
    assert "Resting HR 58" in adv.summary
    assert "n/a" not in adv.summary
    # below the step goal -> a walking suggestion
    assert any("walk" in item.lower() for item in adv.do_more)


def test_rule_based_coaching_flags_problems(profile, rows_7d):
    today = make_snapshot(profile="Dad", rhr=75, sleep=(23.5, 5.0))  # ~5.2h sleep, RHR 75 vs avg 58
    today.summary.avg_stress = 60
    today.summary.body_battery_low = 8
    today.spo2.lowest = 85  # the spo2 block wins over summary.lowest_spo2, as in storage
    today.summary.abnormal_hr_alerts = 1
    adv = rule_based_coaching(profile, rows_7d, today)
    text = " ".join(adv.watch_outs)
    assert "Resting heart rate 75" in text and "above the recent average" in text
    assert "Short sleep" in text
    assert "High stress" in text
    assert "Body battery dropped to 8" in text
    assert "85%" in text
    assert "abnormal heart-rate alert" in text
    assert adv.heart_note and "75" in adv.heart_note
    assert len(adv.do_more) == 4 and len(adv.do_less) == 4  # truncated to 4
    assert all(len(item) <= 220 for item in adv.do_more + adv.do_less)


def test_rule_based_coaching_with_episodes(profile, rows_7d, episode_day, episodes):
    adv = rule_based_coaching(profile, rows_7d, episode_day, episodes_today=episodes)
    assert adv.heart_note and "10:00" in adv.heart_note  # local time in Asia/Singapore
    assert str(max(e.peak_hr for e in episodes)) in adv.heart_note
    assert any("episode" in w for w in adv.watch_outs)


def test_rule_based_coaching_survives_missing_data(profile):
    empty = DaySnapshot(profile="Dad", day=DAY, tz=TZ, fetched_at=now_utc())
    adv = rule_based_coaching(profile, [], empty)
    assert "n/a" in adv.summary
    assert 2 <= len(adv.do_more) <= 4 and 2 <= len(adv.do_less) <= 4
    adv = rule_based_coaching(profile, [], None)
    assert adv.model == "rules"
    adv = rule_based_coaching(profile, [{"day": "2026-09-27"}], None)
    assert adv.summary


# ---------------------------------------------------------------------------
# LLM-backed coaching
# ---------------------------------------------------------------------------


def test_daily_coaching_builds_prompt_and_validates(profile, rows_7d, episode_day, episodes):
    canned = {
        "summary": "  Nice day.\n\n Keep going. ",
        "do_more": ["walk", "walk", "sleep", "water", "stretch", "too many"],
        "do_less": "sitting; late snacks",
        "watch_outs": [{"item": "resting HR"}, 42, None],
        "heart_note": "",
    }
    client = FakeClient(json_answer=canned)
    sym = _symptom(episodes[0].start + timedelta(minutes=2))
    adv = daily_coaching(client, profile, rows_7d, episode_day, episodes, [sym])
    assert isinstance(adv, CoachingAdvice)
    assert adv.model == "fake-model" and adv.period == "daily"
    assert adv.summary == "Nice day. Keep going."
    assert adv.do_more == ["walk", "sleep", "water", "stretch"]  # de-duplicated + truncated to 4
    assert adv.do_less == ["sitting", "late snacks"]
    assert adv.watch_outs == ["resting HR", "42"]
    assert adv.heart_note is None
    call = client.calls[0]
    assert call["schema"] is COACHING_SCHEMA
    assert profile.persona in call["system"] and profile.goals in call["system"]
    assert "English" in call["system"]
    assert "not medical advice" in call["system"] and "diagnosis" in call["system"]
    assert str(episode_day.summary.total_steps) in call["user"]
    assert '"resting_hr":58' in call["user"]
    assert "27 Sep 10:00" in call["user"]  # episode rendered in local time
    assert "fluttering" in call["user"]
    assert "null = not measured" in call["user"]


def test_daily_coaching_propagates_ollama_error(profile, rows_7d, quiet_day):
    client = FakeClient(error=OllamaError("down"))
    with pytest.raises(OllamaError):
        daily_coaching(client, profile, rows_7d, quiet_day, [], [])


def test_daily_coaching_tolerates_garbage_answer(profile, rows_7d, quiet_day):
    client = FakeClient(json_answer={"unexpected": True})
    adv = daily_coaching(client, profile, rows_7d, quiet_day, None, None)
    assert adv.summary and adv.do_more == [] and adv.do_less == [] and adv.heart_note is None


def test_weekly_review_compares_two_weeks(profile, rows_7d, episodes):
    prev = [
        dict(r, day=(DAY - timedelta(days=13 - i)).isoformat(), total_steps=1000)
        for i, r in enumerate(rows_7d)
    ]
    canned = {
        "summary": "Week",
        "do_more": ["a", "b"],
        "do_less": ["c", "d"],
        "watch_outs": [],
        "heart_note": "2 episodes",
    }
    client = FakeClient(json_answer=canned)
    adv = weekly_review(client, profile, rows_7d, prev, episodes)
    assert adv.period == "weekly" and adv.model == "fake-model"
    assert adv.heart_note == "2 episodes"
    user = client.calls[0]["user"]
    assert '"this_week"' in user and '"previous_week"' in user
    assert '"change_this_minus_previous"' in user
    assert '"count":2' in user
    assert '"per_week":2.0' in user  # 2 episodes over a 7-day span
    assert "week compared with the week before" in client.calls[0]["system"]


# ---------------------------------------------------------------------------
# episode assessment
# ---------------------------------------------------------------------------


def test_episode_prompt_contains_required_facts(profile, episode_day, episodes):
    ep = episodes[0]
    sym = _symptom(ep.start + timedelta(minutes=3))
    user = build_episode_prompt(profile, ep, episode_day, [sym])
    facts = json.loads(user.split("\n")[1])
    assert facts["start_local"] == "27 Sep 10:00"
    assert facts["duration_min"] == ep.duration_min
    assert facts["peak_hr"] == ep.peak_hr
    assert facts["baseline_hr"] == round(ep.baseline_hr)
    assert facts["delta_hr"] == round(ep.delta_hr)
    assert facts["max_jump_bpm"] == ep.max_jump_bpm
    assert facts["steps_in_window"] == ep.steps_in_window
    assert facts["activity_level"] == ep.activity_level
    assert facts["asleep"] is False
    assert facts["movement_fraction"] == ep.movement_fraction
    assert facts["stress_avg"] == round(ep.stress_avg)
    assert facts["person_reported_feeling_it"] == "yes"
    assert facts["reported_symptoms_near_episode"][0]["note"] == "fluttering"
    ctx = facts["day_context"]
    assert ctx["resting_hr"] == 58
    assert ctx["activities_that_day"] == ["08:00 Morning Walk, 30 min, avg HR 105"]
    assert ctx["sleep_window"] == "22:30-06:30"
    assert any(s.startswith("09:5") for s in facts["hr_trace_2min"])  # samples before onset
    assert "10:00=" in " ".join(facts["hr_trace_2min"])
    system = analysis._episode_system(profile)
    for name in ASSESSMENTS:
        assert name in system
    assert "never a diagnosis" in system and "Do not invent numbers" in system
    assert "27 Sep 10:00, at rest (0 steps)" in system  # doctor_note example format


def test_episode_prompt_felt_flags(profile, episode_day, episodes):
    ep = episodes[1]
    user = build_episode_prompt(profile, ep, episode_day, [])
    assert '"person_reported_feeling_it":"not asked / unknown"' in user
    ep.felt = False
    assert '"did not notice"' in build_episode_prompt(profile, ep, episode_day, [])
    ep.felt = True
    assert '"person_reported_feeling_it":"yes"' in build_episode_prompt(
        profile, ep, episode_day, []
    )
    # a symptom far away from the episode does not count as "felt"
    ep.felt = None
    far = _symptom(ep.start + timedelta(hours=3))
    user = build_episode_prompt(profile, ep, episode_day, [far])
    assert '"not asked / unknown"' in user and "Other palpitations" in user


def test_episode_prompt_without_snapshot(profile, episodes):
    user = build_episode_prompt(profile, episodes[0], None, None)
    facts = json.loads(user.split("\n")[1])
    assert facts["day_context"]["resting_hr"] is None
    assert facts["hr_trace_2min"]  # falls back to the episode's own samples


def test_assess_episode_validates_output(profile, episode_day, episodes):
    ep = episodes[0]
    client = FakeClient(
        json_answer={
            "assessment": " Possible-Palpitation ",
            "confidence": 1.7,
            "reasoning": "  At rest,\nsudden.  ",
            "doctor_note": "27 Sep 10:00, at rest: HR 64 to 127 bpm for 8 min.",
        }
    )
    result = assess_episode(client, profile, ep, episode_day, [])
    assert isinstance(result, EpisodeAssessment)
    assert result.assessment == ASSESSMENT_POSSIBLE
    assert result.confidence == 1.0
    assert result.reasoning == "At rest, sudden."
    assert result.doctor_note.startswith("27 Sep 10:00")
    assert result.model == "fake-model"
    assert client.calls[0]["schema"] is EPISODE_SCHEMA


def test_assess_episode_unknown_assessment_and_missing_note(profile, episode_day, episodes):
    ep = episodes[0]
    client = FakeClient(
        json_answer={
            "assessment": "anxiety",
            "confidence": "85",
            "reasoning": "",
            "doctor_note": "",
        }
    )
    sym = _symptom(ep.start + timedelta(minutes=1), note="heart racing")
    result = assess_episode(client, profile, ep, episode_day, [sym])
    assert result.assessment == ASSESSMENT_UNCLEAR
    assert result.confidence == 0.85  # percent answer normalised
    assert result.reasoning  # never empty
    note = result.doctor_note
    assert note == fallback_doctor_note(ep, profile, [sym])
    assert "27 Sep 10:00" in note and f"{ep.peak_hr} bpm" in note and "8 min" in note
    assert "person reported heart racing" in note
    assert "at rest" in note


def test_assess_episode_garbage_and_errors(profile, episode_day, episodes):
    result = assess_episode(FakeClient(json_answer={}), profile, episodes[1], episode_day, None)
    assert result.assessment == ASSESSMENT_UNCLEAR and result.confidence == 0.0
    assert result.doctor_note.endswith("gradual onset.") or "sudden onset" in result.doctor_note
    result = assess_episode(
        FakeClient(json_answer={"confidence": -3}), profile, episodes[1], episode_day, None
    )
    assert result.confidence == 0.0
    with pytest.raises(OllamaError):
        assess_episode(
            FakeClient(error=OllamaError("down")), profile, episodes[0], episode_day, None
        )


def test_fallback_doctor_note_variants(profile, episodes):
    ep = episodes[1]
    ep.felt = False
    assert fallback_doctor_note(ep, profile).endswith("person did not notice it.")
    ep.felt = None
    ep.asleep = True
    assert "asleep" in fallback_doctor_note(ep, profile)
    ep.asleep = False
    ep.steps_in_window = 500
    assert "moving (500 steps)" in fallback_doctor_note(ep, profile)


# ---------------------------------------------------------------------------
# doctor narrative
# ---------------------------------------------------------------------------


def test_episode_stats(profile, episodes):
    sym = _symptom(episodes[0].start + timedelta(minutes=2))
    stats = episode_stats(episodes, TZ, span_days=30, symptoms=[sym])
    assert stats["count"] == 2 and stats["span_days"] == 30
    assert stats["per_week"] == pytest.approx(2 * 7 / 30, abs=0.05)
    assert stats["felt_yes"] == 1 and stats["felt_unknown"] == 1
    assert stats["by_time_of_day"] == {"morning (06-12)": 1, "afternoon (12-18)": 1}
    assert stats["peak_hr_max"] == max(e.peak_hr for e in episodes)
    assert episode_stats([], TZ)["count"] == 0


def test_doctor_narrative_uses_client_text(profile, rows_7d, episodes):
    client = FakeClient(text_answer="  Two episodes were recorded.  ")
    sym = _symptom(episodes[0].start + timedelta(minutes=2))
    text = doctor_narrative(client, profile, episodes, [sym], rows_7d)
    assert text == "Two episodes were recorded."
    call = client.calls[0]
    assert call["kind"] == "text"
    assert "No diagnosis" in call["system"] and "2 minutes" in call["system"]
    facts = json.loads(call["user"].split("\n")[1])
    assert facts["episodes"]["count"] == 2
    assert facts["episodes"]["span_days"] == 7
    assert facts["symptom_reports"] == {
        "count": 1,
        "with_detected_episode": 1,
        "without_detected_episode": 0,
        "examples": facts["symptom_reports"]["examples"],
    }
    assert len(facts["episode_notes"]) == 2 and "bpm" in facts["episode_notes"][0]
    assert facts["period"]["first_day"] == (DAY - timedelta(days=6)).isoformat()


def test_doctor_narrative_falls_back_when_empty(profile, rows_7d, episodes):
    text = doctor_narrative(FakeClient(text_answer="   "), profile, episodes, [], rows_7d)
    assert text == rule_based_narrative(profile, episodes, [], rows_7d)
    assert "not a diagnosis" in text


def test_rule_based_narrative_contents(profile, rows_7d, episodes):
    sym_far = _symptom(episodes[0].start - timedelta(days=2))
    text = rule_based_narrative(profile, episodes, [sym_far], rows_7d)
    assert "2 at-rest heart-rate excursions" in text
    assert "optical wrist sensor" in text
    assert "1 palpitation(s) themselves, 1 of which had no matching" in text
    assert "Average resting heart rate 58 bpm" in text
    empty = rule_based_narrative(profile, [], [], [])
    assert "No at-rest heart-rate excursions" in empty and "n/a" in empty
