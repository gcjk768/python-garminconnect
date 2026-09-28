"""Charts: every public function returns PNG bytes, also for empty or partial input."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import matplotlib
from matplotlib.figure import Figure

from garmin_health_monitor import charts
from garmin_health_monitor.models import DaySnapshot, Episode, SymptomReport
from garmin_health_monitor.palpitations import detect_episodes

from .conftest import DAY, TZ, make_snapshot


def _is_png(data: object) -> bool:
    return isinstance(data, bytes) and data.startswith(charts.PNG_SIGNATURE) and len(data) > 500


def _symptoms_for(episode: Episode) -> list[SymptomReport]:
    return [
        SymptomReport(
            profile="Dad",
            reported_at=episode.start + timedelta(minutes=5),
            event_time=episode.start + timedelta(minutes=2),
            note="fluttering",
            hr_at_time=118,
            baseline_hr=64.0,
        ),
        # no HR known and far from any sample -> placed near the top of the axis
        SymptomReport(
            profile="Dad",
            reported_at=episode.start,
            event_time=episode.start + timedelta(hours=9),
            note="skipped beat",
        ),
    ]


def test_agg_backend_selected():
    assert matplotlib.get_backend().lower() == "agg"


# -- hr_day_chart ------------------------------------------------------------


def test_hr_day_chart_full_day(episode_day):
    episodes = detect_episodes(episode_day)
    assert episodes, "fixture should yield episodes"
    png = charts.hr_day_chart(episode_day, episodes, _symptoms_for(episodes[0]), TZ)
    assert _is_png(png)


def test_hr_day_chart_quiet_day(quiet_day):
    assert _is_png(charts.hr_day_chart(quiet_day, [], [], TZ))
    assert _is_png(charts.hr_day_chart(quiet_day, None, None, TZ))


def test_hr_day_chart_empty_snapshot():
    snap = DaySnapshot(profile="Dad", day=DAY, tz=TZ, fetched_at=datetime.now(UTC))
    assert _is_png(charts.hr_day_chart(snap, None, None, TZ))


def test_hr_day_chart_missing_values(episode_day):
    episode_day.summary.resting_hr = None
    episode_day.summary.resting_hr_7d_avg = None
    episode_day.sleep = None
    episode_day.activities[0].name = ""
    episodes = detect_episodes(episode_day)
    # tz=None falls back to the snapshot's own zone
    assert _is_png(charts.hr_day_chart(episode_day, episodes, [], None))


def test_hr_day_chart_uses_local_time_and_title(episode_day):
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot()
    charts.draw_hr_day(ax, episode_day, detect_episodes(episode_day), [], TZ)
    assert "Dad" in ax.get_title()
    lo, hi = ax.get_xlim()
    # a full local day is exactly one matplotlib date unit wide
    assert abs((hi - lo) - 1.0) < 1e-6
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "00:00" in labels and "12:00" in labels


# -- episode_chart -----------------------------------------------------------


def test_episode_chart(episode_day):
    ep = detect_episodes(episode_day)[0]
    assert _is_png(charts.episode_chart(episode_day.hr, ep, TZ))
    assert _is_png(charts.episode_chart(episode_day.hr, ep, TZ, minutes_around=10))


def test_episode_chart_title_is_local(episode_day):
    ep = detect_episodes(episode_day)[0]  # 10:00 Singapore time == 02:00 UTC
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot()
    charts.draw_episode(ax, episode_day.hr, ep, TZ)
    title = ax.get_title()
    assert "10:00" in title and "peak" in title and "min" in title


def test_episode_chart_falls_back_to_episode_samples(episode_day):
    ep = detect_episodes(episode_day)[0]
    assert ep.samples
    assert _is_png(charts.episode_chart([], ep, TZ))
    assert _is_png(charts.episode_chart(None, ep, TZ))


def test_episode_chart_without_any_samples():
    start = datetime(2026, 9, 27, 2, 0, tzinfo=UTC)
    ep = Episode(
        profile="Dad",
        start=start,
        end=start + timedelta(minutes=6),
        duration_min=6.0,
        peak_hr=120,
        mean_hr=115.0,
        baseline_hr=60.0,
        delta_hr=60.0,
    )
    assert _is_png(charts.episode_chart([], ep, TZ))


# -- weekly_trend_chart ------------------------------------------------------


def test_weekly_trend_chart_from_storage_rows(storage):
    for i in range(7):
        snap = make_snapshot(day=DAY - timedelta(days=6 - i), seed=i + 1, sleep=None if i == 3 else (22.5, 6.5))
        storage.save_snapshot(snap)
    rows = storage.get_snapshot_rows("Dad", DAY - timedelta(days=6), DAY)
    assert len(rows) == 7
    rows[1]["resting_hr"] = None
    rows[2]["total_steps"] = None
    rows[4]["sleep_score"] = None
    rows[5]["step_goal"] = None
    assert _is_png(charts.weekly_trend_chart(rows))


def test_weekly_trend_chart_empty_and_garbage():
    assert _is_png(charts.weekly_trend_chart([]))
    assert _is_png(charts.weekly_trend_chart(None))
    assert _is_png(charts.weekly_trend_chart([{"day": None}, {"day": "not-a-date"}, {"total_steps": 5}]))
    # rows with a day but every metric missing
    assert _is_png(charts.weekly_trend_chart([{"day": DAY.isoformat()}]))


def test_weekly_trend_chart_long_period():
    rows = [
        {
            "day": (DAY - timedelta(days=59 - i)).isoformat(),
            "total_steps": 3000 + 40 * i,
            "step_goal": 6000,
            "sleep_seconds": 25000,
            "sleep_score": 70,
            "resting_hr": 58 + (i % 3),
            "resting_hr_7d_avg": 58,
        }
        for i in range(60)
    ]
    assert _is_png(charts.weekly_trend_chart(rows))


# -- episode_histograms ------------------------------------------------------


def test_episode_histograms(episode_day):
    episodes = detect_episodes(episode_day)
    episodes[0].asleep = True
    assert _is_png(charts.episode_histograms(episodes, TZ))
    assert _is_png(charts.episode_histograms(episodes, TZ, period=(DAY - timedelta(days=29), DAY)))
    # long period -> weekly bins
    assert _is_png(charts.episode_histograms(episodes, TZ, period=(DAY - timedelta(days=120), DAY)))


def test_episode_histograms_hours_are_local(episode_day):
    episodes = detect_episodes(episode_day)  # 10:00 and 15:30 Singapore time
    fig = Figure(figsize=(8, 4))
    ax_days, ax_hours = fig.subplots(1, 2)
    charts.draw_episode_histograms(ax_days, ax_hours, episodes, TZ)
    heights = {}
    for bar in ax_hours.patches:
        if bar.get_height() > 0:
            heights[round(bar.get_x() + bar.get_width() / 2)] = heights.get(round(bar.get_x() + bar.get_width() / 2), 0) + bar.get_height()
    assert heights.get(10) == 1 and heights.get(15) == 1
    assert 2 not in heights  # would be the UTC hour


def test_episode_histograms_empty():
    assert _is_png(charts.episode_histograms([], TZ))
    assert _is_png(charts.episode_histograms(None, TZ))
