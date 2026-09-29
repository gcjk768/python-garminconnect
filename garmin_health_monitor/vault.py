"""Obsidian vault log of possible palpitations.

One note per day (``Episodes/YYYY-MM-DD.md``) plus a ``Home.md`` index. Every
write rebuilds the day note from the database, so re-runs never duplicate rows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from .models import Episode
from .utils import to_local

_COUNT_RE = re.compile(r"^episodes: (\d+)$", re.MULTILINE)


def _row(ep: Episode, tz: str) -> str:
    start, end = to_local(ep.start, tz), to_local(ep.end, tz)
    state = "asleep" if ep.asleep else "at rest"
    ai = (ep.llm_assessment or "").replace("_", " ") or "not assessed"
    felt = {True: "yes", False: "no"}.get(ep.felt, "?")
    return (
        f"| {start:%H:%M}–{end:%H:%M} | {round(ep.duration_min)} min | {ep.peak_hr} | "
        f"{round(ep.baseline_hr)} | {state} | {ep.steps_in_window} | {ai} | {felt} |"
    )


def write_day(vault_dir: str, name: str, day: date, episodes: Sequence[Episode], tz: str, today: date) -> Path:
    """Rewrite the note for ``day`` and refresh the index. Returns the note path."""
    root = Path(vault_dir)
    (root / "Episodes").mkdir(parents=True, exist_ok=True)
    eps = sorted(episodes, key=lambda e: e.start)
    lines = [
        "---",
        "tags: [heart]",
        f"date: {day.isoformat()}",
        f"episodes: {len(eps)}",
        f"updated: {today.isoformat()}",
        "---",
        f"# {day:%a %d %b %Y} · {len(eps)} possible palpitation{'s' if len(eps) != 1 else ''} ({name})",
        "",
        "| Time | Duration | Peak HR | HR before | State | Steps | AI view | Felt |",
        "|---|---|---|---|---|---|---|---|",
        *(_row(e, tz) for e in eps),
        "",
    ]
    notes = [f"- **{to_local(e.start, tz):%H:%M}** {e.doctor_note}" for e in eps if e.doctor_note]
    if notes:
        lines += ["## Notes for the doctor", *notes, ""]
    lines.append("Wrist-sensor readings for a symptom diary, not a diagnosis. Back to [[Home]].")
    path = root / "Episodes" / f"{day.isoformat()}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_index(root, name, today)
    return path


def write_index(root: Path, name: str, today: date) -> None:
    """``Home.md``: every day note, newest first, grouped by month with counts."""
    days = sorted((root / "Episodes").glob("*.md"), reverse=True)
    lines = [
        "---",
        "tags: [heart]",
        f"updated: {today.isoformat()}",
        "---",
        f"# {name}: heart palpitation log",
        "",
        "Written automatically by the Garmin monitor each time a possible palpitation is detected.",
    ]
    month = None
    for p in days:
        m = _COUNT_RE.search(p.read_text(encoding="utf-8"))
        count = int(m.group(1)) if m else 0
        if count == 0:
            continue
        if p.stem[:7] != month:
            month = p.stem[:7]
            lines += ["", f"## {date.fromisoformat(p.stem):%B %Y}"]
        lines.append(f"- [[{p.stem}]]: {count}")
    (root / "Home.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
