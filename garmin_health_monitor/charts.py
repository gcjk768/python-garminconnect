"""Matplotlib charts for Telegram photos and the doctor report.

Every public function returns PNG bytes (Agg backend, 150 dpi, tight bounding
box) and tolerates empty or partial input: a chart with nothing to show draws a
short "no data" note instead of raising, so a rendering problem can never block
a Telegram message.

The ``draw_*`` helpers draw the same content onto caller-supplied axes; the
doctor report uses them to place charts on A4 PDF pages.

Data arrives in UTC; conversion to the profile's wall-clock time happens here
(``utils.to_local``).  Matplotlib is handed *naive* local datetimes so the axis
formatters never re-convert them.
"""

from __future__ import annotations

import io
import logging
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .models import DaySnapshot, Episode, HRSample, SymptomReport  # noqa: E402
from .utils import as_float, as_int, local_day_bounds, parse_date, to_local  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette (colour-blind safe; adjacent pairs validated with the dataviz checker)
# ---------------------------------------------------------------------------

BLUE = "#2a78d6"  # heart-rate line, steps
ORANGE = "#eb6834"  # symptom reports, resting-HR trend
AQUA = "#1baf7a"  # activity windows
VIOLET = "#4a3aa7"  # sleep, nocturnal episodes
RED = "#e34948"  # episode spans
RED_DARK = "#b3302f"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#9a9993"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_STYLE: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": INK3,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
}

DASHED = (0, (4, 3))
GAP_TOLERANCE = timedelta(minutes=10)  # longer gaps between samples are drawn as breaks


@contextmanager
def chart_style() -> Iterator[None]:
    """rc context with the monitor's chart look (spines off, light grid, readable fonts)."""
    with matplotlib.rc_context(_STYLE):
        yield


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _png(fig: Figure) -> bytes:
    """Render ``fig`` to PNG bytes and close it."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    finally:
        plt.close(fig)
    return buf.getvalue()


def _local_naive(dt: datetime, tz: str) -> datetime:
    return to_local(dt, tz).replace(tzinfo=None)


def no_data(ax: Axes, text: str = "No data") -> None:
    """Blank the axes and print a centred note."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=INK2, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _with_gaps(points: Sequence[tuple[datetime, float]]) -> tuple[list[datetime], list[float]]:
    """Insert NaNs where consecutive samples are far apart so the line breaks instead of bridging."""
    xs: list[datetime] = []
    ys: list[float] = []
    prev: datetime | None = None
    for x, y in points:
        if prev is not None and (x - prev) > GAP_TOLERANCE:
            xs.append(prev + timedelta(seconds=1))
            ys.append(float("nan"))
        xs.append(x)
        ys.append(float(y))
        prev = x
    return xs, ys


def _clip(start: datetime, end: datetime, lo: datetime, hi: datetime) -> tuple[datetime, datetime] | None:
    s, e = max(start, lo), min(end, hi)
    if e <= s:
        return None
    return s, e


def _fmt_minutes(minutes: float | None) -> str:
    if minutes is None:
        return "n/a"
    return f"{minutes:.0f} min" if float(minutes).is_integer() else f"{minutes:.1f} min"


def _valid_samples(samples: Sequence[HRSample] | None) -> list[HRSample]:
    out = [s for s in (samples or []) if s is not None and s.ts is not None and s.hr is not None]
    out.sort(key=lambda s: s.ts)
    return out


def _peak_time(ep: Episode, samples: Sequence[HRSample], fallback: datetime) -> datetime:
    """UTC time of the highest sample inside the episode window (or ``fallback``)."""
    inside = [s for s in samples if ep.start <= s.ts <= ep.end]
    if not inside:
        inside = _valid_samples(ep.samples)
    if not inside:
        return fallback
    return max(inside, key=lambda s: s.hr).ts


# ---------------------------------------------------------------------------
# Heart rate over one local day
# ---------------------------------------------------------------------------


def draw_hr_day(
    ax: Axes,
    snap: DaySnapshot,
    episodes: Sequence[Episode] | None,
    symptoms: Sequence[SymptomReport] | None,
    tz: str | None,
) -> None:
    """Draw the day's HR line with sleep, activities, episodes and symptom markers on ``ax``."""
    zone = tz or snap.tz or "UTC"
    day_start, day_end = local_day_bounds(snap.day, zone)
    x0, x1 = day_start.replace(tzinfo=None), day_end.replace(tzinfo=None)
    title = f"{snap.profile} · {snap.day.strftime('%a %d %b %Y')} · heart rate"
    samples = _valid_samples(snap.hr)
    if not samples:
        no_data(ax, f"No heart-rate data for {snap.day.isoformat()}")
        ax.set_title(title)
        return

    points = [(_local_naive(s.ts, zone), s.hr) for s in samples]
    xs, ys = _with_gaps(points)
    hr_values = [s.hr for s in samples]
    lo = max(20, min(hr_values) - 10)
    hi = max(hr_values) + 18
    ax.plot(xs, ys, color=BLUE, linewidth=1.4, zorder=3)
    handles: list[Any] = [Line2D([], [], color=BLUE, linewidth=1.4, label="Heart rate")]

    # sleep window
    sleep = snap.sleep
    if sleep is not None and sleep.start and sleep.end and sleep.end > sleep.start:
        span = _clip(_local_naive(sleep.start, zone), _local_naive(sleep.end, zone), x0, x1)
        if span:
            ax.axvspan(span[0], span[1], color=VIOLET, alpha=0.08, linewidth=0, zorder=0)
            handles.append(Patch(facecolor=VIOLET, alpha=0.25, label="Sleep"))

    # recorded / auto-detected activities
    activity_labelled = False
    for act in snap.activities or []:
        if not (act.start and act.end and act.end > act.start):
            continue
        span = _clip(_local_naive(act.start, zone), _local_naive(act.end, zone), x0, x1)
        if not span:
            continue
        ax.axvspan(span[0], span[1], color=AQUA, alpha=0.15, linewidth=0, zorder=0)
        name = (act.name or act.type_key or "activity").strip()
        if act.source == "auto_detected":
            name += " (auto)"
        mid = span[0] + (span[1] - span[0]) / 2
        ax.text(mid, hi - 1, name, ha="center", va="top", fontsize=7, color=INK2, clip_on=True, zorder=4)
        activity_labelled = True
    if activity_labelled:
        handles.append(Patch(facecolor=AQUA, alpha=0.35, label="Activity"))

    # resting HR line
    rhr = snap.summary.resting_hr
    if rhr is None and sleep is not None:
        rhr = sleep.resting_hr
    if rhr is None:
        rhr = snap.summary.resting_hr_7d_avg
    if rhr:
        ax.axhline(rhr, color=INK2, linestyle=DASHED, linewidth=1, zorder=2)
        handles.append(Line2D([], [], color=INK2, linestyle="--", label=f"Resting HR {rhr}"))

    # episodes
    episode_drawn = False
    for ep in episodes or []:
        if not (ep.start and ep.end):
            continue
        span = _clip(_local_naive(ep.start, zone), _local_naive(ep.end, zone), x0, x1)
        if not span:
            continue
        ax.axvspan(span[0], span[1], color=RED, alpha=0.22, linewidth=0, zorder=1)
        peak_x = _local_naive(_peak_time(ep, samples, ep.start + (ep.end - ep.start) / 2), zone)
        peak_x = min(max(peak_x, span[0]), span[1])
        ax.annotate(
            f"{ep.peak_hr}",
            xy=(peak_x, ep.peak_hr),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=RED_DARK,
            zorder=6,
        )
        episode_drawn = True
    if episode_drawn:
        handles.append(Patch(facecolor=RED, alpha=0.4, label="Episode (peak bpm)"))

    # person-reported symptoms
    symptom_drawn = False
    for rep in symptoms or []:
        if rep.event_time is None:
            continue
        x = _local_naive(rep.event_time, zone)
        if not (x0 <= x <= x1):
            continue
        y: float | None = float(rep.hr_at_time) if rep.hr_at_time is not None else None
        if y is None:
            nearest = min(samples, key=lambda s: abs((s.ts - rep.event_time).total_seconds()))
            if abs((nearest.ts - rep.event_time).total_seconds()) <= 6 * 60:
                y = float(nearest.hr)
        if y is None:
            y = lo + 0.9 * (hi - lo)
        ax.plot(
            [x],
            [y],
            marker="^",
            markersize=9,
            color=ORANGE,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linestyle="none",
            zorder=5,
        )
        symptom_drawn = True
    if symptom_drawn:
        handles.append(
            Line2D(
                [],
                [],
                marker="^",
                color=ORANGE,
                markeredgecolor="white",
                markersize=8,
                linestyle="none",
                label="Reported symptom",
            )
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("bpm")
    ax.set_title(title)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(len(handles), 6),
        handlelength=1.6,
        columnspacing=1.4,
    )


def hr_day_chart(
    snap: DaySnapshot,
    episodes: Sequence[Episode] | None,
    symptoms: Sequence[SymptomReport] | None,
    tz: str | None,
) -> bytes:
    """Heart rate over the local day with sleep, activities, episodes and symptom markers (PNG)."""
    with chart_style():
        fig, ax = plt.subplots(figsize=(11, 4.6))
        try:
            draw_hr_day(ax, snap, episodes, symptoms, tz)
        except Exception:  # noqa: BLE001 - never let a chart detail break the message
            logger.exception("hr_day_chart: drawing failed, sending a blank chart")
            ax.cla()
            no_data(ax, "Chart could not be drawn")
        return _png(fig)


# ---------------------------------------------------------------------------
# Zoom on one episode
# ---------------------------------------------------------------------------


def draw_episode(
    ax: Axes,
    samples: Sequence[HRSample] | None,
    ep: Episode,
    tz: str | None,
    minutes_around: int = 45,
) -> None:
    """Draw +/- ``minutes_around`` of heart rate around ``ep`` on ``ax``."""
    zone = tz or "UTC"
    around = timedelta(minutes=max(1, int(minutes_around)))
    lo_t, hi_t = ep.start - around, ep.end + around
    start_l, end_l = to_local(ep.start, zone), to_local(ep.end, zone)
    title = (
        f"{start_l:%a %d %b %Y} · {start_l:%H:%M}–{end_l:%H:%M} · "
        f"{_fmt_minutes(ep.duration_min)} · peak {ep.peak_hr} bpm"
    )
    pts = [s for s in _valid_samples(samples) if lo_t <= s.ts <= hi_t]
    if not pts:
        pts = _valid_samples(ep.samples)
    if not pts:
        no_data(ax, "No heart-rate samples stored around this episode")
        ax.set_title(title)
        return

    xs, ys = _with_gaps([(_local_naive(s.ts, zone), s.hr) for s in pts])
    ax.plot(xs, ys, color=BLUE, linewidth=1.6, marker="o", markersize=3, zorder=3)
    handles: list[Any] = [Line2D([], [], color=BLUE, marker="o", markersize=3, label="Heart rate")]
    ax.axvspan(start_l.replace(tzinfo=None), end_l.replace(tzinfo=None), color=RED, alpha=0.2, linewidth=0, zorder=1)
    handles.append(Patch(facecolor=RED, alpha=0.4, label="Episode"))
    baseline = as_float(ep.baseline_hr)
    if baseline:
        ax.axhline(baseline, color=INK2, linestyle=DASHED, linewidth=1, zorder=2)
        handles.append(Line2D([], [], color=INK2, linestyle="--", label=f"Baseline {baseline:.0f}"))

    peak_x = _local_naive(_peak_time(ep, pts, ep.start + (ep.end - ep.start) / 2), zone)
    ax.annotate(
        f"peak {ep.peak_hr}",
        xy=(peak_x, ep.peak_hr),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=RED_DARK,
        zorder=6,
    )

    hr_values = [s.hr for s in pts] + [ep.peak_hr] + ([int(baseline)] if baseline else [])
    ax.set_ylim(max(20, min(hr_values) - 8), max(hr_values) + 14)
    ax.set_xlim(_local_naive(lo_t, zone), _local_naive(hi_t, zone))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("bpm")
    ax.set_title(title)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(handles), handlelength=1.6)


def episode_chart(
    samples: Sequence[HRSample] | None,
    ep: Episode,
    tz: str | None,
    minutes_around: int = 45,
) -> bytes:
    """Zoomed heart-rate chart around one episode (PNG)."""
    with chart_style():
        fig, ax = plt.subplots(figsize=(9, 4))
        try:
            draw_episode(ax, samples, ep, tz, minutes_around)
        except Exception:  # noqa: BLE001
            logger.exception("episode_chart: drawing failed, sending a blank chart")
            ax.cla()
            no_data(ax, "Chart could not be drawn")
        return _png(fig)


# ---------------------------------------------------------------------------
# Daily trend: steps, sleep, resting HR
# ---------------------------------------------------------------------------


def _row_day(row: dict[str, Any]) -> date | None:
    try:
        return parse_date(row.get("day"))  # type: ignore[arg-type]
    except (TypeError, ValueError, AttributeError):
        return None


def draw_weekly_trend(axes: Sequence[Axes], rows: Sequence[dict[str, Any]] | None) -> None:
    """Three stacked panels (steps + goal, sleep hours + score, resting HR) from snapshot rows."""
    ax_steps, ax_sleep, ax_rhr = axes[0], axes[1], axes[2]
    days: list[tuple[date, dict[str, Any]]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        d = _row_day(row)
        if d is not None:
            days.append((d, row))
    days.sort(key=lambda p: p[0])
    if not days:
        for ax, what in ((ax_steps, "steps"), (ax_sleep, "sleep"), (ax_rhr, "resting heart rate")):
            no_data(ax, f"No {what} data")
        ax_steps.set_title("Steps")
        ax_sleep.set_title("Sleep")
        ax_rhr.set_title("Resting heart rate")
        return

    xs = [datetime.combine(d, time.min) for d, _ in days]
    span_days = (days[-1][0] - days[0][0]).days + 1
    bar_width = 0.7

    # -- steps ------------------------------------------------------------
    goals = Counter(g for g in (as_int(r.get("step_goal")) for _, r in days) if g)
    step_pts = [(x, as_int(r.get("total_steps"))) for x, (_, r) in zip(xs, days, strict=False)]
    step_pts = [(x, v) for x, v in step_pts if v is not None]
    if step_pts:
        ax_steps.bar([x for x, _ in step_pts], [v for _, v in step_pts], width=bar_width, color=BLUE, linewidth=0, zorder=2)
        handles: list[Any] = []
        if goals:
            goal = goals.most_common(1)[0][0]
            ax_steps.axhline(goal, color=INK2, linestyle=DASHED, linewidth=1, zorder=3)
            handles.append(Line2D([], [], color=INK2, linestyle="--", label=f"Goal {goal:,}"))
            ax_steps.legend(handles=handles, loc="upper right")
        ax_steps.set_ylim(0, max(max(v for _, v in step_pts), *(goals or [0])) * 1.15 + 1)
    else:
        no_data(ax_steps, "No step data")
    ax_steps.set_title("Steps per day")
    ax_steps.set_ylabel("steps")

    # -- sleep ------------------------------------------------------------
    sleep_pts = []
    for x, (_, r) in zip(xs, days, strict=False):
        secs = as_float(r.get("sleep_seconds"))
        if secs is None or secs <= 0:
            continue
        sleep_pts.append((x, secs / 3600.0, as_int(r.get("sleep_score"))))
    if sleep_pts:
        ax_sleep.bar([x for x, _, _ in sleep_pts], [h for _, h, _ in sleep_pts], width=bar_width, color=VIOLET, linewidth=0, zorder=2)
        top = max(h for _, h, _ in sleep_pts)
        for x, h, score in sleep_pts:
            if score is not None:
                ax_sleep.text(x, h + top * 0.03, f"{score}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax_sleep.set_ylim(0, top * 1.25 + 0.5)
    else:
        no_data(ax_sleep, "No sleep data")
    ax_sleep.set_title("Sleep hours (number above bar = Garmin sleep score)")
    ax_sleep.set_ylabel("hours")

    # -- resting HR -------------------------------------------------------
    rhr_pts = [(x, as_int(r.get("resting_hr"))) for x, (_, r) in zip(xs, days, strict=False)]
    rhr_pts = [(x, v) for x, v in rhr_pts if v]
    if rhr_pts:
        rx = [x for x, _ in rhr_pts]
        ry = [float(v) for _, v in rhr_pts]
        ax_rhr.plot(rx, ry, color=ORANGE, linewidth=1.6, marker="o", markersize=4, markeredgecolor="white", zorder=3)
        avg_pts = [(x, as_int(r.get("resting_hr_7d_avg"))) for x, (_, r) in zip(xs, days, strict=False)]
        avg_pts = [(x, v) for x, v in avg_pts if v]
        if avg_pts:
            ax_rhr.plot([x for x, _ in avg_pts], [float(v) for _, v in avg_pts], color=INK2, linestyle=DASHED, linewidth=1, zorder=2)
            ax_rhr.legend(
                handles=[
                    Line2D([], [], color=ORANGE, marker="o", markersize=4, label="Resting HR"),
                    Line2D([], [], color=INK2, linestyle="--", label="7-day average"),
                ],
                loc="upper right",
            )
        all_vals = ry + [float(v) for _, v in avg_pts]
        ax_rhr.set_ylim(min(all_vals) - 6, max(all_vals) + 8)
        ax_rhr.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        no_data(ax_rhr, "No resting heart-rate data")
    ax_rhr.set_title("Resting heart rate")
    ax_rhr.set_ylabel("bpm")

    # -- shared x formatting ---------------------------------------------
    fmt = mdates.DateFormatter("%a %d" if span_days <= 14 else "%d %b")
    for ax in (ax_steps, ax_sleep, ax_rhr):
        if not ax.get_xticks().size and not ax.lines and not ax.patches:
            continue  # blanked "no data" panel
        ax.set_xlim(xs[0] - timedelta(hours=14), xs[-1] + timedelta(hours=14))
        if span_days <= 14:
            ax.xaxis.set_major_locator(mdates.DayLocator())
        elif span_days <= 45:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(axis="x", labelrotation=0)
    ax_steps.tick_params(labelbottom=False)
    ax_sleep.tick_params(labelbottom=False)


def weekly_trend_chart(rows: Sequence[dict[str, Any]] | None) -> bytes:
    """Steps / sleep / resting-HR trend over the given snapshot rows (PNG)."""
    with chart_style():
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        fig.subplots_adjust(hspace=0.45)
        try:
            draw_weekly_trend(axes, rows)
        except Exception:  # noqa: BLE001
            logger.exception("weekly_trend_chart: drawing failed, sending a blank chart")
            for ax in axes:
                ax.cla()
                no_data(ax, "Chart could not be drawn")
        return _png(fig)


# ---------------------------------------------------------------------------
# Episode frequency
# ---------------------------------------------------------------------------


def draw_episode_histograms(
    ax_days: Axes,
    ax_hours: Axes,
    episodes: Sequence[Episode] | None,
    tz: str | None,
    period: tuple[date, date] | None = None,
) -> None:
    """Episodes per day (or per week for long periods) and by local hour of day, awake vs asleep."""
    zone = tz or "UTC"
    eps = [e for e in (episodes or []) if e is not None and e.start is not None]
    ax_days.set_title("Episodes per day")
    ax_hours.set_title("Episodes by hour of day (local time)")
    if not eps:
        no_data(ax_days, "No episodes in this period")
        no_data(ax_hours, "No episodes in this period")
        return

    local_starts = [(to_local(e.start, zone), bool(e.asleep)) for e in eps]
    dates = [d.date() for d, _ in local_starts]
    if period and period[0] and period[1]:
        d0, d1 = period
    else:
        d0, d1 = min(dates), max(dates)
    if d1 < d0:
        d0, d1 = d1, d0
    d0 = min(d0, min(dates))
    d1 = max(d1, max(dates))
    n_days = (d1 - d0).days + 1

    # -- per day / per week ----------------------------------------------
    weekly = n_days > 62
    if weekly:
        bins = [d0 + timedelta(days=7 * i) for i in range((n_days + 6) // 7)]
        awake = Counter()
        asleep = Counter()
        for d, is_asleep in zip(dates, (a for _, a in local_starts), strict=False):
            idx = (d - d0).days // 7
            (asleep if is_asleep else awake)[idx] += 1
        xs = [datetime.combine(b, time.min) for b in bins]
        width = 6.2
    else:
        bins = [d0 + timedelta(days=i) for i in range(n_days)]
        awake = Counter()
        asleep = Counter()
        for d, is_asleep in zip(dates, (a for _, a in local_starts), strict=False):
            idx = (d - d0).days
            (asleep if is_asleep else awake)[idx] += 1
        xs = [datetime.combine(b, time.min) for b in bins]
        width = 0.7
    awake_vals = [awake.get(i, 0) for i in range(len(bins))]
    asleep_vals = [asleep.get(i, 0) for i in range(len(bins))]
    ax_days.bar(xs, awake_vals, width=width, color=BLUE, linewidth=0, zorder=2, align="edge" if weekly else "center")
    ax_days.bar(
        xs,
        asleep_vals,
        width=width,
        bottom=awake_vals,
        color=VIOLET,
        linewidth=0,
        zorder=2,
        align="edge" if weekly else "center",
    )
    ax_days.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_days.set_ylabel("episodes per week" if weekly else "episodes")
    if weekly:
        ax_days.set_xlim(xs[0], datetime.combine(d1, time.min) + timedelta(days=1))
        ax_days.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=max(1, len(bins) // 8)))
        ax_days.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    else:
        ax_days.set_xlim(xs[0] - timedelta(hours=14), xs[-1] + timedelta(hours=14))
        ax_days.xaxis.set_major_locator(mdates.DayLocator(interval=1 if n_days <= 14 else 7 if n_days > 35 else 3))
        ax_days.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d" if n_days <= 14 else "%d %b"))
    ax_days.set_ylim(0, max(1, max(a + b for a, b in zip(awake_vals, asleep_vals, strict=False))) * 1.2 + 0.2)

    # -- by hour of day --------------------------------------------------
    hours_awake = Counter(d.hour for d, a in local_starts if not a)
    hours_asleep = Counter(d.hour for d, a in local_starts if a)
    h = list(range(24))
    va = [hours_awake.get(i, 0) for i in h]
    vs = [hours_asleep.get(i, 0) for i in h]
    ax_hours.bar(h, va, width=0.75, color=BLUE, linewidth=0, zorder=2)
    ax_hours.bar(h, vs, width=0.75, bottom=va, color=VIOLET, linewidth=0, zorder=2)
    ax_hours.set_xlim(-0.6, 23.6)
    ax_hours.set_xticks(range(0, 24, 3))
    ax_hours.set_xticklabels([f"{i:02d}:00" for i in range(0, 24, 3)])
    ax_hours.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_hours.set_ylim(0, max(1, max(a + b for a, b in zip(va, vs, strict=False))) * 1.2 + 0.2)
    ax_hours.set_ylabel("episodes")
    ax_hours.legend(
        handles=[Patch(facecolor=BLUE, label="Awake"), Patch(facecolor=VIOLET, label="Asleep")],
        loc="upper right",
        ncol=2,
    )


def episode_histograms(
    episodes: Sequence[Episode] | None,
    tz: str | None,
    period: tuple[date, date] | None = None,
) -> bytes:
    """Two panels: episodes per day over the period and by local hour of day (PNG)."""
    with chart_style():
        fig, (ax_days, ax_hours) = plt.subplots(1, 2, figsize=(11, 3.8))
        fig.subplots_adjust(wspace=0.3)
        try:
            draw_episode_histograms(ax_days, ax_hours, episodes, tz, period)
        except Exception:  # noqa: BLE001
            logger.exception("episode_histograms: drawing failed, sending a blank chart")
            for ax in (ax_days, ax_hours):
                ax.cla()
                no_data(ax, "Chart could not be drawn")
        return _png(fig)


# ---------------------------------------------------------------------------
# Month calendar: one coloured square per day, easy to read for an older person
# ---------------------------------------------------------------------------

CAL_NONE = "#cdeedd"  # soft green: a normal day
CAL_ONE = "#f6b26b"  # orange: 1 episode
CAL_MANY = "#e34948"  # red: 2 or more
CAL_EMPTY = "#f1f0ec"  # no data yet / future


def heart_calendar(
    year: int,
    month: int,
    counts: dict[date, int],
    has_data: set[date],
    title: str,
    marker: tuple[date, str] | None = None,
) -> bytes:
    """PNG month calendar: big day squares coloured by episode count, count printed large.

    ``marker`` outlines one day and writes a short label under its number (e.g. a medicine start).
    """
    import calendar

    weeks = calendar.Calendar(firstweekday=0).monthdatescalendar(year, month)
    with chart_style():
        fig, ax = plt.subplots(figsize=(7, 1.2 + 1.05 * len(weeks)))
        ax.set_xlim(0, 7)
        ax.set_ylim(len(weeks) + 0.6, 0)
        ax.axis("off")
        for col, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            ax.text(col + 0.5, 0.25, name, ha="center", va="center", fontsize=13, color=INK2, fontweight="bold")
        for row, week in enumerate(weeks, start=1):
            for col, d in enumerate(week):
                if d.month != month:
                    continue
                n = counts.get(d, 0)
                colour = CAL_EMPTY if d not in has_data else CAL_MANY if n >= 2 else CAL_ONE if n == 1 else CAL_NONE
                edge, lw = (VIOLET, 3.0) if marker and marker[0] == d else ("white", 2.0)
                ax.add_patch(plt.Rectangle((col + 0.04, row - 0.46), 0.92, 0.92,
                                           facecolor=colour, edgecolor=edge, linewidth=lw))
                ax.text(col + 0.12, row - 0.33, str(d.day), ha="left", va="center", fontsize=11, color=INK2)
                if n:
                    ax.text(col + 0.5, row + 0.05, str(n), ha="center", va="center", fontsize=24,
                            fontweight="bold", color="white" if n >= 2 else INK)
                if marker and marker[0] == d:
                    ax.text(col + 0.5, row + 0.34, marker[1], ha="center", va="center", fontsize=8.5,
                            color=VIOLET, fontweight="bold")
        ax.set_title(title, fontsize=17, loc="left", pad=14)
        legend = [
            Patch(facecolor=CAL_NONE, label="Normal day"),
            Patch(facecolor=CAL_ONE, label="1 fast-heartbeat episode"),
            Patch(facecolor=CAL_MANY, label="2 or more"),
        ]
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=11,
                  handlelength=1.4, handleheight=1.4, frameon=False)
        return _png(fig)
