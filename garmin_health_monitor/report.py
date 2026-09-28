"""Doctor report: a PDF diary of heart-rate excursions plus a CSV of every record.

The PDF is built page by page with :class:`matplotlib.backends.backend_pdf.PdfPages`
(A4 portrait figures; text is placed with ``Figure.text``, charts with
``Figure.add_axes`` using the ``draw_*`` helpers from :mod:`charts`).  It is a
*symptom diary* for a clinician: every number comes from the stored data, the
wording stays plain, and each page carries the reminder that this is not a
medical device and not a diagnosis.

The CSV has one row per detected episode and one per person-reported symptom
(``record_type`` column) so a doctor or the person can open it in a spreadsheet.
"""

from __future__ import annotations

import csv
import logging
import textwrap
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from . import charts
from .config import ProfileConfig
from .models import (
    ASSESSMENT_ARTIFACT,
    ASSESSMENT_EXERTION,
    ASSESSMENT_POSSIBLE,
    ASSESSMENT_UNCLEAR,
    Episode,
    HRSample,
    SymptomReport,
)
from .normalize import RAW_DEVICE
from .storage import Storage
from .utils import local_day_bounds, median, now_utc, to_local

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass and constants
# ---------------------------------------------------------------------------


@dataclass
class ReportFiles:
    """Paths of the generated files plus a plain-text summary for the Telegram caption."""

    pdf_path: Path
    csv_path: Path
    summary: str
    n_episodes: int
    n_symptoms: int


CSV_COLUMNS = [
    "record_type",
    "date",
    "start_local",
    "end_local",
    "duration_min",
    "kind",
    "peak_hr",
    "baseline_hr",
    "delta_hr",
    "mean_hr",
    "max_jump_bpm",
    "asleep",
    "steps_in_window",
    "activity_level",
    "confidence",
    "felt",
    "llm_assessment",
    "llm_confidence",
    "doctor_note",
    "note",
]

MAX_ZOOM_CHARTS = 12
ZOOM_MINUTES_AROUND = 45
EPISODE_ROWS_PER_PAGE = 28

ASSESSMENT_LABELS = {
    ASSESSMENT_POSSIBLE: "possible palpitation",
    ASSESSMENT_EXERTION: "likely exertion",
    ASSESSMENT_ARTIFACT: "likely sensor artifact",
    ASSESSMENT_UNCLEAR: "unclear",
}
NOT_ASSESSED = "not assessed"

DISCLAIMER_PARAGRAPHS = [
    "The numbers in this diary come from a wrist-worn optical heart-rate sensor read through "
    "Garmin Connect at 2-minute resolution. The sensor measures pulse rate only; it cannot see "
    "heart rhythm and cannot detect or rule out an arrhythmia. Short events between two samples "
    "are invisible, and movement, a loose strap or cold skin can produce false readings.",
    "Episodes are picked out automatically by a simple rule: the heart rate rose well above the "
    "resting level while no steps or activity were recorded. Some episodes will be sensor noise "
    "and some real events will be missed. The 'confidence' value is a heuristic score for that "
    "rule, not a probability that anything is wrong with the heart.",
    "The 'model assessment' and any narrative text were produced by a local language model that "
    "was given only the numbers shown here. That text describes the data in plain words; it is "
    "not a medical opinion.",
    "This is not a medical device and nothing in this document is a diagnosis. It is a symptom "
    "diary, prepared so that the pattern, frequency and timing of episodes can be discussed with "
    "a clinician who can decide whether further tests are useful.",
]

GLOSSARY = [
    ("Episode", "a stretch of 2-minute samples at rest where the heart rate exceeded the thresholds."),
    ("Baseline", "median heart rate over the preceding 30 minutes at rest (or the day's resting HR)."),
    ("Delta", "peak heart rate minus baseline, in beats per minute."),
    ("Onset jump", "largest change between two consecutive samples in the episode."),
    ("Confidence", "detector heuristic from 0 to 1 (more rest, longer, sharper onset = higher)."),
    ("Felt", "the person's own answer to 'did you notice it?' sent from the phone."),
    ("Symptom report", "a palpitation the person logged themselves, with the heart rate near that time."),
]

# ---------------------------------------------------------------------------
# Page layout (A4 portrait, inches)
# ---------------------------------------------------------------------------

A4 = (8.27, 11.69)
MARGIN_X = 0.7
MARGIN_TOP = 0.7
MARGIN_BOTTOM = 0.7
USABLE_WIDTH = A4[0] - 2 * MARGIN_X
SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"
SANS_EM = 0.56  # average glyph advance as a fraction of the font size (conservative)
MONO_EM = 0.61
ROW_SHADE = "#f3f3f0"
TABLE_GUTTER = 1  # characters between table columns


def _chars_per_inch(size: float, mono: bool) -> float:
    return 72.0 / (size * (MONO_EM if mono else SANS_EM))


def _wrap(text: Any, chars: int) -> list[str]:
    s = "" if text is None else str(text)
    if not s.strip():
        return [""]
    lines: list[str] = []
    for para in s.splitlines():
        lines.extend(textwrap.wrap(para, max(1, chars), break_long_words=True) or [""])
    return lines


class _Page:
    """One A4 figure with a downward cursor (``y`` = inches from the bottom edge)."""

    def __init__(self) -> None:
        self.fig = Figure(figsize=A4)
        self.fig.patch.set_facecolor("white")
        self.y = A4[1] - MARGIN_TOP

    @staticmethod
    def fx(inches: float) -> float:
        return inches / A4[0]

    @staticmethod
    def fy(inches: float) -> float:
        return inches / A4[1]

    @property
    def remaining(self) -> float:
        return self.y - MARGIN_BOTTOM

    def line(self, text: str, size: float, x: float = MARGIN_X, weight: str = "normal", color: str = charts.INK, mono: bool = False) -> None:
        """Draw one already-wrapped line at the cursor (the caller moves the cursor)."""
        self.fig.text(
            self.fx(x),
            self.fy(self.y),
            text,
            fontsize=size,
            fontweight=weight,
            color=color,
            family=MONO if mono else SANS,
            ha="left",
            va="baseline",
        )

    def rect(self, y_bottom: float, height: float, color: str, x: float = MARGIN_X, width: float = USABLE_WIDTH, zorder: int = 1) -> None:
        self.fig.add_artist(
            Rectangle(
                (self.fx(x), self.fy(y_bottom)),
                self.fx(width),
                self.fy(height),
                transform=self.fig.transFigure,
                facecolor=color,
                edgecolor="none",
                zorder=zorder,
            )
        )

    def footer(self, left: str, right: str) -> None:
        self.fig.text(self.fx(MARGIN_X), self.fy(0.35), left, fontsize=7, color=charts.INK2, ha="left", va="baseline", family=SANS)
        self.fig.text(self.fx(A4[0] - MARGIN_X), self.fy(0.35), right, fontsize=7, color=charts.INK2, ha="right", va="baseline", family=SANS)
        self.rect(0.5, 0.008, charts.GRID)


@dataclass(slots=True)
class _Col:
    header: str
    chars: int  # width budget in monospace characters (before scaling)


class _Flow:
    """A sequence of pages with a text cursor; content that does not fit continues on a new page."""

    def __init__(self) -> None:
        self.pages: list[_Page] = []
        self.page: _Page | None = None
        self._continued: str | None = None

    # -- pages --------------------------------------------------------------

    def new_page(self, heading: str | None = None) -> _Page:
        page = _Page()
        self.pages.append(page)
        self.page = page
        self._continued = f"{heading} (continued)" if heading else None
        if heading:
            self.heading(heading, size=13)
        return page

    def _current(self) -> _Page:
        if self.page is None:
            self.new_page()
        assert self.page is not None
        return self.page

    def _overflow(self) -> _Page:
        continued = self._continued
        page = self.new_page(continued)
        self._continued = continued
        return page

    def ensure(self, height: float) -> _Page:
        """Return a page with at least ``height`` inches free below the cursor."""
        page = self._current()
        if page.remaining < height:
            page = self._overflow()
        return page

    # -- text -----------------------------------------------------------------

    def gap(self, inches: float) -> None:
        page = self._current()
        page.y -= inches

    def text(
        self,
        s: str,
        size: float = 9.5,
        weight: str = "normal",
        color: str = charts.INK,
        mono: bool = False,
        indent: float = 0.0,
        leading: float = 1.4,
    ) -> None:
        width = USABLE_WIDTH - indent
        cpl = max(8, int(width * _chars_per_inch(size, mono)))
        line_h = size / 72.0 * leading
        for text in _wrap(s, cpl):
            page = self.ensure(line_h)
            page.y -= line_h
            page.line(text, size, x=MARGIN_X + indent, weight=weight, color=color, mono=mono)

    def heading(self, s: str, size: float = 15) -> None:
        self.text(s, size=size, weight="bold")
        self.gap(0.1)

    def subheading(self, s: str, size: float = 11) -> None:
        self.ensure(1.0)  # never leave a heading alone at the foot of a page
        self.gap(0.12)
        self.text(s, size=size, weight="bold")
        self.gap(0.05)

    def rule(self, color: str = charts.GRID) -> None:
        page = self._current()
        page.rect(page.y, 0.012, color)

    # -- charts ---------------------------------------------------------------

    def axes(self, height: float, title_room: float = 0.28, label_room: float = 0.35) -> Axes:
        """Reserve room for an axes (plus its title above and tick labels / legend below)."""
        page = self.ensure(height + title_room + label_room)
        page.y -= title_room + height
        ax = page.fig.add_axes([page.fx(MARGIN_X), page.fy(page.y), page.fx(USABLE_WIDTH), page.fy(height)])
        page.y -= label_room
        return ax

    # -- tables ---------------------------------------------------------------

    def table(self, cols: Sequence[_Col], rows: Sequence[Sequence[Any]], size: float = 6.6, max_rows_per_page: int | None = None) -> None:
        """Monospace table with wrapped cells; repeats the header on every page it spans."""
        cpi = _chars_per_inch(size, mono=True)
        total_chars = sum(c.chars for c in cols) + TABLE_GUTTER * (len(cols) - 1)
        scale = min(1.0, (USABLE_WIDTH * cpi) / max(1, total_chars))
        char_w = scale / cpi  # inches per character after scaling
        budgets = [max(1, int(c.chars * scale)) for c in cols]
        xs: list[float] = []
        x = MARGIN_X
        for col in cols:
            xs.append(x)
            x += (col.chars + TABLE_GUTTER) * char_w
        line_h = size / 72.0 * 1.35
        header_h = line_h + 0.15

        def draw_header(page: _Page) -> None:
            page.y -= line_h
            for col, cx in zip(cols, xs, strict=False):
                page.line(col.header, size, x=cx, weight="bold", mono=True)
            page.y -= 0.05
            page.rect(page.y, 0.012, charts.INK3)
            page.y -= 0.1

        wrapped_rows = [[_wrap(cell, budget) for cell, budget in zip(row, budgets, strict=False)] for row in rows]
        if not wrapped_rows:
            self.ensure(header_h)
            draw_header(self._current())
            return
        first_h = max(len(c) for c in wrapped_rows[0]) * line_h + 0.06
        page = self.ensure(header_h + first_h)
        draw_header(page)
        on_page = 0
        for i, cells in enumerate(wrapped_rows):
            n_lines = max(len(c) for c in cells) if cells else 1
            row_h = n_lines * line_h + 0.06
            if page.remaining < row_h or (max_rows_per_page and on_page >= max_rows_per_page):
                page = self._overflow()
                draw_header(page)
                on_page = 0
            if i % 2 == 1:
                page.rect(page.y - row_h, row_h, ROW_SHADE, x=MARGIN_X - 0.04, width=USABLE_WIDTH + 0.08, zorder=0)
            page.y -= 0.03
            for k in range(n_lines):
                page.y -= line_h
                for cx, lines in zip(xs, cells, strict=False):
                    if k < len(lines) and lines[k]:
                        page.line(lines[k], size, x=cx, mono=True)
            page.y -= 0.03
            on_page += 1


# ---------------------------------------------------------------------------
# Formatting helpers (never raise on None)
# ---------------------------------------------------------------------------


def _fmt_dt(dt: datetime | None, tz: str) -> str:
    return to_local(dt, tz).strftime("%Y-%m-%d %H:%M") if dt else "n/a"


def _fmt_hm(dt: datetime | None, tz: str) -> str:
    return to_local(dt, tz).strftime("%H:%M") if dt else "--:--"


def _fmt_day(dt: datetime | None, tz: str) -> str:
    return to_local(dt, tz).strftime("%Y-%m-%d") if dt else "n/a"


def _fmt_num(value: Any, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_signed(value: Any, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_minutes(minutes: float | None) -> str:
    if minutes is None:
        return "n/a"
    try:
        m = float(minutes)
    except (TypeError, ValueError):
        return "n/a"
    return f"{m:.0f} min" if m.is_integer() else f"{m:.1f} min"


def _felt_label(felt: bool | None) -> str:
    if felt is None:
        return "—"
    return "yes" if felt else "no"


def _assessment_label(code: str | None) -> str:
    if not code:
        return NOT_ASSESSED
    return ASSESSMENT_LABELS.get(code, str(code).replace("_", " "))


def _context_label(ep: Episode) -> str:
    parts = ["asleep" if ep.asleep else "awake", f"{ep.steps_in_window or 0} steps"]
    level = (ep.activity_level or "").strip()
    if level and level.lower() != "unknown":
        parts.append(level)
    if ep.movement_fraction and ep.movement_fraction >= 0.3:
        parts.append("movement")
    return ", ".join(parts)


def _csv_value(value: Any, digits: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}"
    return str(value)


def _pct(part: int, whole: int) -> str:
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:.0f}%"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class _Week:
    start: date
    end: date
    count: int = 0
    asleep: int = 0
    felt: int = 0


@dataclass
class _Stats:
    n_days: int
    n_episodes: int
    n_symptoms: int
    n_symptoms_linked: int
    per_week_rate: float | None
    weeks: list[_Week]
    hours: Counter
    n_asleep: int
    n_felt: int
    n_not_noticed: int
    n_unanswered: int
    assessments: Counter
    kinds: Counter
    peak_max: int | None
    duration_median: float | None
    duration_max: float | None
    delta_median: float | None = None
    top_hours: list[tuple[int, int]] = field(default_factory=list)


def _compute_stats(
    episodes: Sequence[Episode],
    symptoms: Sequence[SymptomReport],
    start: date,
    end: date,
    tz: str,
) -> _Stats:
    n_days = max(1, (end - start).days + 1)
    weeks: list[_Week] = []
    d = start
    while d <= end:
        weeks.append(_Week(start=d, end=min(end, d + timedelta(days=6))))
        d += timedelta(days=7)
    hours: Counter = Counter()
    assessments: Counter = Counter()
    kinds: Counter = Counter()
    n_asleep = n_felt = n_not = n_unanswered = 0
    peaks: list[int] = []
    durations: list[float] = []
    deltas: list[float] = []
    for ep in episodes:
        local = to_local(ep.start, tz)
        hours[local.hour] += 1
        idx = (local.date() - start).days // 7
        if 0 <= idx < len(weeks):
            weeks[idx].count += 1
            if ep.asleep:
                weeks[idx].asleep += 1
            if ep.felt:
                weeks[idx].felt += 1
        if ep.asleep:
            n_asleep += 1
        if ep.felt is None:
            n_unanswered += 1
        elif ep.felt:
            n_felt += 1
        else:
            n_not += 1
        assessments[_assessment_label(ep.llm_assessment)] += 1
        kinds[ep.kind or "unknown"] += 1
        if ep.peak_hr is not None:
            peaks.append(int(ep.peak_hr))
        if ep.duration_min is not None:
            durations.append(float(ep.duration_min))
        if ep.delta_hr is not None:
            deltas.append(float(ep.delta_hr))
    n_symptoms_linked = sum(1 for s in symptoms if s.episode_id is not None)
    return _Stats(
        n_days=n_days,
        n_episodes=len(episodes),
        n_symptoms=len(symptoms),
        n_symptoms_linked=n_symptoms_linked,
        per_week_rate=len(episodes) / (n_days / 7.0),
        weeks=weeks,
        hours=hours,
        n_asleep=n_asleep,
        n_felt=n_felt,
        n_not_noticed=n_not,
        n_unanswered=n_unanswered,
        assessments=assessments,
        kinds=kinds,
        peak_max=max(peaks) if peaks else None,
        duration_median=median(durations),
        duration_max=max(durations) if durations else None,
        delta_median=median(deltas),
        top_hours=sorted(hours.items(), key=lambda kv: (-kv[1], kv[0]))[:3],
    )


def _summary_lines(profile: ProfileConfig, stats: _Stats, start: date, end: date, tz: str) -> list[str]:
    """Plain-language statistics; every number is computed from the stored records."""
    lines = [f"Period: {start.isoformat()} to {end.isoformat()} ({stats.n_days} days, times in {tz})."]
    if stats.n_episodes == 0:
        lines.append("Detected episodes: 0.")
    else:
        rate = f"about {stats.per_week_rate:.1f} per week" if stats.per_week_rate is not None else "rate n/a"
        lines.append(f"Detected episodes: {stats.n_episodes} ({rate}).")
        hours = ", ".join(f"{h:02d}:00-{h + 1:02d}:00 ({n})" for h, n in stats.top_hours)
        lines.append(f"Most common hours: {hours}.")
        lines.append(
            f"While asleep: {stats.n_asleep} of {stats.n_episodes} ({_pct(stats.n_asleep, stats.n_episodes)}); "
            f"awake: {stats.n_episodes - stats.n_asleep}."
        )
        lines.append(
            f"Felt by {profile.name}: {stats.n_felt} of {stats.n_episodes} ({_pct(stats.n_felt, stats.n_episodes)}); "
            f"not noticed: {stats.n_not_noticed}; not answered: {stats.n_unanswered}."
        )
        lines.append(
            f"Highest peak: {_fmt_num(stats.peak_max)} bpm; median rise over baseline: "
            f"{_fmt_signed(stats.delta_median)} bpm; median duration: {_fmt_minutes(stats.duration_median)}; "
            f"longest: {_fmt_minutes(stats.duration_max)}."
        )
        lines.append("Type: " + ", ".join(f"{k} {n}" for k, n in stats.kinds.most_common()) + ".")
        lines.append("Model assessment: " + ", ".join(f"{k} {n}" for k, n in stats.assessments.most_common()) + ".")
    if stats.n_symptoms == 0:
        lines.append("Symptom reports logged by the person: 0.")
    else:
        lines.append(
            f"Symptom reports logged by the person: {stats.n_symptoms}, "
            f"of which {stats.n_symptoms_linked} matched a detected episode."
        )
    return lines


def _summary_text(profile: ProfileConfig, stats: _Stats, start: date, end: date, tz: str) -> str:
    return "\n".join([f"Heart-rate excursion diary for {profile.name}.", *_summary_lines(profile, stats, start, end, tz)])


# ---------------------------------------------------------------------------
# Report context
# ---------------------------------------------------------------------------


@dataclass
class _Context:
    profile: ProfileConfig
    tz: str
    start: date
    end: date
    episodes: list[Episode]
    symptoms: list[SymptomReport]
    rows: list[dict[str, Any]]
    stats: _Stats
    narrative: str | None
    device_note: str
    generated_at: datetime
    zoom_samples: dict[int, list[HRSample]]
    episode_index: dict[int, int]  # episode id -> row number in the table (1-based)


def _device_note(storage: Storage, profile: ProfileConfig, rows: Sequence[dict[str, Any]]) -> str:
    """Best-effort device name from the newest stored raw payload."""
    name = None
    for row in list(rows)[-3:][::-1]:
        try:
            raw = storage.get_snapshot_raw(profile.name, row.get("day"))
        except Exception:  # noqa: BLE001 - the device name is decoration only
            raw = None
        dev = raw.get(RAW_DEVICE) if isinstance(raw, dict) else None
        if isinstance(dev, dict) and dev.get("lastUsedDeviceName"):
            name = str(dev["lastUsedDeviceName"])
            break
    device = f"Garmin {name}" if name else "Garmin wrist device"
    return f"{device} (wrist optical sensor, heart rate sampled every 2 minutes, synced via Garmin Connect)"


def _ranked_for_zoom(episodes: Sequence[Episode]) -> list[Episode]:
    return sorted(episodes, key=lambda e: (-(e.confidence or 0.0), e.start))[:MAX_ZOOM_CHARTS]


def _load_context(profile: ProfileConfig, storage: Storage, start: date, end: date, narrative: str | None) -> _Context:
    tz = profile.timezone or "UTC"
    utc_start, _ = local_day_bounds(start, tz)
    _, utc_end = local_day_bounds(end, tz)
    episodes = sorted(storage.get_episodes(profile.name, utc_start, utc_end), key=lambda e: e.start)
    symptoms = sorted(storage.get_symptoms(profile.name, utc_start, utc_end), key=lambda s: s.event_time)
    rows = storage.get_snapshot_rows(profile.name, start, end)
    stats = _compute_stats(episodes, symptoms, start, end, tz)
    around = timedelta(minutes=ZOOM_MINUTES_AROUND)
    zoom: dict[int, list[HRSample]] = {}
    for ep in _ranked_for_zoom(episodes):
        if ep.id is None:
            continue
        try:
            zoom[ep.id] = storage.get_hr_samples(profile.name, ep.start - around, ep.end + around)
        except Exception as exc:  # noqa: BLE001 - fall back to the episode's own samples
            logger.warning("%s: could not load HR samples for episode %s: %s", profile.name, ep.id, exc)
            zoom[ep.id] = []
    return _Context(
        profile=profile,
        tz=tz,
        start=start,
        end=end,
        episodes=episodes,
        symptoms=symptoms,
        rows=rows,
        stats=stats,
        narrative=(narrative or "").strip() or None,
        device_note=_device_note(storage, profile, rows),
        generated_at=now_utc(),
        zoom_samples=zoom,
        episode_index={ep.id: i + 1 for i, ep in enumerate(episodes) if ep.id is not None},
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _cover(flow: _Flow, ctx: _Context) -> None:
    flow.new_page()
    flow.text("Heart-rate excursion diary", size=21, weight="bold")
    flow.gap(0.02)
    flow.text(f"for {ctx.profile.name}", size=14, color=charts.INK2)
    flow.gap(0.15)
    flow.text(
        f"Period: {ctx.start.strftime('%d %b %Y')} to {ctx.end.strftime('%d %b %Y')} ({ctx.stats.n_days} days). "
        f"All times are local ({ctx.tz}).",
        size=10,
    )
    flow.text(f"Generated: {to_local(ctx.generated_at, ctx.tz).strftime('%d %b %Y %H:%M')}.", size=10)
    flow.text(f"Device: {ctx.device_note}.", size=10)
    if ctx.profile.persona:
        flow.text(f"About the person (as configured by the family): {ctx.profile.persona}", size=10)

    flow.subheading("Summary")
    for line in _summary_lines(ctx.profile, ctx.stats, ctx.start, ctx.end, ctx.tz):
        flow.text(line, size=9.5, indent=0.15)

    if ctx.narrative:
        flow.subheading("Plain-language summary (local language model, from the numbers above)")
        flow.text(ctx.narrative, size=9.5, indent=0.15)
        flow.gap(0.04)
        flow.text("This text describes the data. It is not a medical opinion.", size=8.5, color=charts.INK2, indent=0.15)

    flow.subheading("Please read before interpreting")
    for para in DISCLAIMER_PARAGRAPHS:
        flow.text(para, size=8.6, indent=0.15, leading=1.32)
        flow.gap(0.04)

    flow.subheading("How to read the tables")
    for term, meaning in GLOSSARY:
        flow.text(f"{term}: {meaning}", size=8.2, indent=0.15, leading=1.25)


def _frequency(flow: _Flow, ctx: _Context) -> None:
    flow.new_page("Frequency and timing")
    flow.text(
        f"{ctx.stats.n_episodes} detected episode(s) over {ctx.stats.n_days} days. "
        "Bars show detected episodes; purple = while asleep.",
        size=9.5,
        color=charts.INK2,
    )
    flow.gap(0.1)
    ax_days = flow.axes(2.0)
    ax_hours = flow.axes(1.9)
    charts.draw_episode_histograms(ax_days, ax_hours, ctx.episodes, ctx.tz, (ctx.start, ctx.end))

    flow.subheading("Per week")
    week_rows = [
        [f"{w.start.isoformat()} to {w.end.isoformat()}", str(w.count), str(w.asleep), str(w.count - w.asleep), str(w.felt)]
        for w in ctx.stats.weeks
    ]
    flow.table([_Col("Week", 26), _Col("Episodes", 9), _Col("Asleep", 7), _Col("Awake", 7), _Col("Felt", 6)], week_rows, size=7.5)

    flow.subheading("Breakdown")
    st = ctx.stats
    breakdown = [
        ["Asleep vs awake", f"{st.n_asleep} asleep, {st.n_episodes - st.n_asleep} awake"],
        ["Felt vs not", f"{st.n_felt} felt, {st.n_not_noticed} not noticed, {st.n_unanswered} not answered"],
        ["Type", ", ".join(f"{k} {n}" for k, n in st.kinds.most_common()) or "n/a"],
        ["Model assessment", ", ".join(f"{k} {n}" for k, n in st.assessments.most_common()) or "n/a"],
        ["Symptom reports", f"{st.n_symptoms} logged, {st.n_symptoms_linked} matched a detected episode"],
    ]
    flow.table([_Col("", 18), _Col("", 88)], breakdown, size=7.5)


EPISODE_COLS = [
    _Col("#", 3),
    _Col("Date", 10),
    _Col("Start-end", 11),
    _Col("Dur", 6),
    _Col("Peak", 4),
    _Col("Base", 4),
    _Col("Delta", 5),
    _Col("Jump", 4),
    _Col("Context", 14),
    _Col("Conf", 4),
    _Col("Felt", 4),
    _Col("Assessment", 12),
    _Col("Doctor note", 34),
]


def _episode_row(ctx: _Context, ep: Episode) -> list[str]:
    number = ctx.episode_index.get(ep.id, 0) if ep.id is not None else 0
    assess = _assessment_label(ep.llm_assessment)
    if ep.llm_assessment and ep.llm_confidence is not None:
        assess += f" ({ep.llm_confidence:.1f})"
    note = ep.doctor_note or ""
    if ep.notes:
        note = f"{note} [{ep.notes}]" if note else f"[{ep.notes}]"
    return [
        str(number) if number else "",
        _fmt_day(ep.start, ctx.tz),
        f"{_fmt_hm(ep.start, ctx.tz)}-{_fmt_hm(ep.end, ctx.tz)}",
        _fmt_minutes(ep.duration_min),
        _fmt_num(ep.peak_hr),
        _fmt_num(ep.baseline_hr),
        _fmt_signed(ep.delta_hr),
        _fmt_num(ep.max_jump_bpm),
        _context_label(ep),
        _fmt_num(ep.confidence, 2),
        _felt_label(ep.felt),
        assess,
        note,
    ]


def _episodes(flow: _Flow, ctx: _Context) -> None:
    flow.new_page("Detected episodes")
    if not ctx.episodes:
        flow.text("No episodes were detected in this period.", size=10, color=charts.INK2)
        return
    flow.text(
        "Heart rate in bpm. Times local. Dur = duration, Base = baseline, Delta = peak minus baseline, "
        "Jump = largest change between consecutive 2-minute samples, Conf = detector heuristic 0-1.",
        size=8,
        color=charts.INK2,
    )
    flow.gap(0.1)
    flow.table(EPISODE_COLS, [_episode_row(ctx, ep) for ep in ctx.episodes], size=6.6, max_rows_per_page=EPISODE_ROWS_PER_PAGE)


SYMPTOM_COLS = [
    _Col("Time", 16),
    _Col("What the person reported", 46),
    _Col("HR near that time", 22),
    _Col("Matched episode", 30),
]


def _symptom_row(ctx: _Context, rep: SymptomReport) -> list[str]:
    if rep.hr_at_time is not None:
        hr = f"{rep.hr_at_time} bpm"
        if rep.baseline_hr is not None:
            hr += f" (baseline {rep.baseline_hr:.0f})"
    else:
        hr = "n/a"
    linked = "—"
    if rep.episode_id is not None:
        number = ctx.episode_index.get(rep.episode_id)
        match = next((e for e in ctx.episodes if e.id == rep.episode_id), None)
        if match is not None and number:
            linked = f"#{number} {_fmt_hm(match.start, ctx.tz)}-{_fmt_hm(match.end, ctx.tz)}, peak {match.peak_hr}"
        else:
            linked = f"episode id {rep.episode_id} (outside this period)"
    note = rep.note or "(no note)"
    if rep.source == "button":
        note += " [answered from an alert]"
    return [_fmt_dt(rep.event_time, ctx.tz), note, hr, linked]


def _symptoms(flow: _Flow, ctx: _Context) -> None:
    flow.new_page("Symptom reports from the person")
    flow.text(
        "Palpitations the person logged themselves from the phone (/palp), with the heart rate stored nearest to that time.",
        size=8.5,
        color=charts.INK2,
    )
    flow.gap(0.1)
    if not ctx.symptoms:
        flow.text("No symptom reports were logged in this period.", size=10, color=charts.INK2)
        return
    flow.table(SYMPTOM_COLS, [_symptom_row(ctx, rep) for rep in ctx.symptoms], size=6.8)


def _trend(flow: _Flow, ctx: _Context) -> None:
    flow.new_page("Daily context: steps, sleep and resting heart rate")
    flow.text(
        f"{len(ctx.rows)} day(s) with data in this period. Resting heart rate and sleep are Garmin's daily values.",
        size=9,
        color=charts.INK2,
    )
    flow.gap(0.05)
    ax1 = flow.axes(2.3, label_room=0.3)
    ax2 = flow.axes(2.3, label_room=0.3)
    ax3 = flow.axes(2.3, label_room=0.35)
    charts.draw_weekly_trend((ax1, ax2, ax3), ctx.rows)


def _zooms(flow: _Flow, ctx: _Context) -> None:
    ranked = _ranked_for_zoom(ctx.episodes)
    if not ranked:
        return
    flow.new_page("Episode charts")
    flow.text(
        f"Up to {MAX_ZOOM_CHARTS} episodes, highest detector confidence first. "
        f"Each chart shows {ZOOM_MINUTES_AROUND} minutes before and after the episode.",
        size=8.5,
        color=charts.INK2,
    )
    for ep in ranked:
        flow.ensure(4.5)  # keep the caption and its chart together
        number = ctx.episode_index.get(ep.id, 0) if ep.id is not None else 0
        label = f"Episode #{number}" if number else "Episode"
        flow.gap(0.18)
        flow.text(
            f"{label} · baseline {_fmt_num(ep.baseline_hr)} bpm · rise {_fmt_signed(ep.delta_hr)} bpm · "
            f"{_context_label(ep)} · confidence {_fmt_num(ep.confidence, 2)} · felt: {_felt_label(ep.felt)}",
            size=8.5,
            weight="bold",
        )
        note = ep.doctor_note or ""
        flow.text(f"Model assessment: {_assessment_label(ep.llm_assessment)}. {note}".strip(), size=8.5, color=charts.INK2)
        flow.gap(0.05)
        ax = flow.axes(3.0, title_room=0.3, label_room=0.6)
        samples = ctx.zoom_samples.get(ep.id, []) if ep.id is not None else []
        charts.draw_episode(ax, samples, ep, ctx.tz, ZOOM_MINUTES_AROUND)


def _build_pages(ctx: _Context) -> list[_Page]:
    flow = _Flow()
    _cover(flow, ctx)
    _frequency(flow, ctx)
    _episodes(flow, ctx)
    _symptoms(flow, ctx)
    _trend(flow, ctx)
    _zooms(flow, ctx)
    return flow.pages


def _render_pages(ctx: _Context) -> Iterator[Figure]:
    """Yield finished A4 figures in report order, footers and page numbers applied."""
    with charts.chart_style():
        pages = _build_pages(ctx)
        left = (
            f"{ctx.profile.name} · heart-rate excursion diary · {ctx.start.isoformat()} to {ctx.end.isoformat()} · "
            "not a medical device, for discussion with a clinician"
        )
        for number, page in enumerate(pages, 1):
            page.footer(left, f"page {number} of {len(pages)}")
            yield page.fig


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _episode_csv_row(ep: Episode, tz: str) -> dict[str, str]:
    return {
        "record_type": "episode",
        "date": _fmt_day(ep.start, tz),
        "start_local": _fmt_dt(ep.start, tz),
        "end_local": _fmt_dt(ep.end, tz),
        "duration_min": _csv_value(ep.duration_min, 1),
        "kind": _csv_value(ep.kind),
        "peak_hr": _csv_value(ep.peak_hr),
        "baseline_hr": _csv_value(ep.baseline_hr, 1),
        "delta_hr": _csv_value(ep.delta_hr, 1),
        "mean_hr": _csv_value(ep.mean_hr, 1),
        "max_jump_bpm": _csv_value(ep.max_jump_bpm),
        "asleep": _csv_value(bool(ep.asleep)),
        "steps_in_window": _csv_value(ep.steps_in_window),
        "activity_level": _csv_value(ep.activity_level),
        "confidence": _csv_value(ep.confidence, 2),
        "felt": _csv_value(ep.felt),
        "llm_assessment": _csv_value(ep.llm_assessment),
        "llm_confidence": _csv_value(ep.llm_confidence, 2),
        "doctor_note": _csv_value(ep.doctor_note),
        "note": _csv_value(ep.notes),
    }


def _symptom_csv_row(rep: SymptomReport, tz: str) -> dict[str, str]:
    delta = None
    if rep.hr_at_time is not None and rep.baseline_hr is not None:
        delta = float(rep.hr_at_time) - float(rep.baseline_hr)
    return {
        "record_type": "symptom",
        "date": _fmt_day(rep.event_time, tz),
        "start_local": _fmt_dt(rep.event_time, tz),
        "end_local": "",
        "duration_min": "",
        "kind": "reported",
        "peak_hr": _csv_value(rep.hr_at_time),
        "baseline_hr": _csv_value(rep.baseline_hr, 1),
        "delta_hr": _csv_value(delta, 1),
        "mean_hr": "",
        "max_jump_bpm": "",
        "asleep": "",
        "steps_in_window": "",
        "activity_level": "",
        "confidence": "",
        "felt": "yes",
        "llm_assessment": "",
        "llm_confidence": "",
        "doctor_note": "",
        "note": _csv_value(rep.note),
    }


def _write_csv(ctx: _Context, path: Path) -> None:
    records: list[tuple[datetime, dict[str, str]]] = []
    records.extend((ep.start, _episode_csv_row(ep, ctx.tz)) for ep in ctx.episodes)
    records.extend((rep.event_time, _symptom_csv_row(rep, ctx.tz)) for rep in ctx.symptoms)
    records.sort(key=lambda r: r[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for _, row in records:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_doctor_report(
    profile: ProfileConfig,
    storage: Storage,
    start: date,
    end: date,
    out_dir: Path,
    narrative: str | None = None,
) -> ReportFiles:
    """Write ``<slug>-palpitation-report-<start>_<end>.pdf`` and ``.csv`` into ``out_dir``.

    ``start``/``end`` are inclusive local calendar dates of ``profile.timezone``.
    ``narrative`` is an optional paragraph (from the local language model) shown on the cover.
    """
    if end < start:
        start, end = end, start
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = _load_context(profile, storage, start, end, narrative)
    stem = f"{profile.slug}-palpitation-report-{start.isoformat()}_{end.isoformat()}"
    pdf_path = out_dir / f"{stem}.pdf"
    csv_path = out_dir / f"{stem}.csv"

    _write_csv(ctx, csv_path)
    n_pages = 0
    with PdfPages(pdf_path) as pdf:
        info = pdf.infodict()
        info["Title"] = f"Heart-rate excursion diary - {profile.name} - {start.isoformat()} to {end.isoformat()}"
        info["Subject"] = "Symptom diary from a wrist heart-rate sensor; not a medical device, not a diagnosis"
        info["Author"] = "Garmin Health Monitor"
        for fig in _render_pages(ctx):
            pdf.savefig(fig)
            fig.clear()
            n_pages += 1
    summary = _summary_text(profile, ctx.stats, start, end, ctx.tz)
    logger.info(
        "%s: doctor report %s (%d pages, %d episodes, %d symptoms)",
        profile.name,
        pdf_path.name,
        n_pages,
        ctx.stats.n_episodes,
        ctx.stats.n_symptoms,
    )
    return ReportFiles(
        pdf_path=pdf_path,
        csv_path=csv_path,
        summary=summary,
        n_episodes=ctx.stats.n_episodes,
        n_symptoms=ctx.stats.n_symptoms,
    )
