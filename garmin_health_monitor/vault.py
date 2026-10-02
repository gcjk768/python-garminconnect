"""Obsidian vault: movement log + memory (James's NAS app-vault standard).

Layout under ``vault_dir`` (flat, at most two levels):

* ``Home.md``: map of contents, rebuilt on every note write.
* ``Activity/YYYY/MM/YYYY-MM-DD.md``: append-only, one line per event,
  ``- HH:MM emoji **what** · detail · [[entity]]`` (local time, SGT on the NAS).
  Old flat ``Activity/YYYY-MM-DD.md`` notes are moved into ``YYYY/MM/`` by :func:`migrate`.
* ``Episodes/YYYY-MM-DD.md``: possible palpitations that day (table rebuilt from the database).
* ``Days/YYYY-MM-DD.md``: the day's numbers and coaching (profiles with coaching).
* ``Metrics/<metric>.md``: one note per metric (resting HR, HRV, sleep, stress...), one History
  line per day. ``Alerts/<type>.md``: one note per alert type, every send in its History.

Entity notes end with an append-only ``## History`` section that survives every rewrite.
:func:`memory` reads it all back as a capped, newest-first excerpt for the LLM prompts.

Every public function here is best-effort: it logs and returns on any error, so the vault can
never crash a poll or swallow an alert. Never pass secrets, tokens or whole prompts in.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from pathlib import Path

from .models import Episode
from .utils import to_local

logger = logging.getLogger(__name__)

MEMORY_CHARS = 4000  # cap on the excerpt passed to the LLM
RECENT_NOTES = 14  # newest Activity / entity notes read for memory
ENTITY_DIRS = ("Episodes", "Days")  # date-named notes read back as memory
NAMED_DIRS = ("Alerts", "Metrics")  # one note per alert type / metric
HISTORY = "## History"
_COUNT_RE = re.compile(r"^episodes: (\d+)$", re.MULTILINE)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _clean(text: object, limit: int = 240) -> str:
    """One line, no Markdown table/link breakers, capped."""
    s = " ".join(str(text or "").split()).replace("[[", "[").replace("]]", "]")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _write(path: Path, text: str) -> None:
    """Atomic write (tmp + rename), left editable by James on the share."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o664)
    os.replace(tmp, path)


def _split(path: Path) -> tuple[str, list[str]]:
    """A note as (text above ``## History`` without frontmatter, History lines)."""
    if not path.exists():
        return "", []
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        text = text[end + 4:] if end >= 0 else text
    head, _, past = text.partition("\n" + HISTORY)
    return head.strip("\n"), [ln for ln in past.splitlines() if ln.strip()]


def _history(path: Path) -> list[str]:
    """The existing ``## History`` lines of a note (kept across rewrites)."""
    return _split(path)[1]


def _with_history(path: Path, lines: list[str]) -> str:
    return "\n".join([*lines, "", HISTORY, *_history(path)]) + "\n"


# --------------------------------------------------------------------- write


def _activity_path(root: Path, day: date) -> Path:
    return root / "Activity" / f"{day:%Y}" / f"{day:%m}" / f"{day.isoformat()}.md"


def migrate(vault_dir: str | None) -> list[str]:
    """Move old flat ``Activity/YYYY-MM-DD.md`` notes into ``Activity/YYYY/MM/`` (never deletes)."""
    moved: list[str] = []
    if not vault_dir:
        return moved
    try:
        root = Path(vault_dir)
        for old in sorted((root / "Activity").glob("*.md")):
            if not _DATE_RE.match(old.stem):
                continue
            new = _activity_path(root, date.fromisoformat(old.stem))
            if new.exists():
                continue
            new.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old, new)
            moved.append(old.stem)
        if moved:
            logger.info("vault: moved %d Activity notes into YYYY/MM folders", len(moved))
            write_index(root, date.today())
    except Exception as exc:  # noqa: BLE001
        logger.warning("vault migrate failed: %s", exc)
    return moved


def history(vault_dir: str | None, entity: str, line: str, today: date, replace: str | None = None) -> None:
    """Add ``line`` to ``<entity>.md``'s ``## History`` (note created if missing; text above History
    kept verbatim; atomic write). ``replace``: an existing line with that prefix is replaced, so a
    re-run for the same day never duplicates a reading."""
    if not vault_dir:
        return
    try:
        path = Path(vault_dir) / f"{entity}.md"
        head, past = _split(path)
        if replace:
            past = [ln for ln in past if not ln.startswith(replace)]
        new = not head
        head = head or f"# {Path(entity).name}"
        _write(path, f"---\ntags: [active]\nupdated: {today.isoformat()}\n---\n{head}\n\n{HISTORY}\n"
                     + "".join(f"{ln}\n" for ln in [*past, line]))
        if new:
            write_index(Path(vault_dir), today)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vault history %s failed: %s", entity, exc)


def log(vault_dir: str | None, when: datetime, emoji: str, what: str, detail: str = "",
        entity: str | Sequence[str] = "") -> None:
    """Append one event to ``Activity/YYYY/MM/<day>.md`` (and to each entity note's History)."""
    if not vault_dir:
        return
    try:
        root = Path(vault_dir)
        day = when.date().isoformat()
        entities = [entity] if isinstance(entity, str) else list(entity)
        parts = [f"- {when:%H:%M} {emoji} **{_clean(what, 60)}**"]
        if detail:
            parts.append(_clean(detail))
        parts += [f"[[{e}]]" for e in entities if e]
        line = " · ".join(parts)
        path = _activity_path(root, when.date())
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            if new:
                fh.write(f"---\ntags: [activity]\nupdated: {day}\n---\n# Activity {day}\n\n")
            fh.write(line + "\n")
        if new:
            with contextlib.suppress(OSError):
                os.chmod(path, 0o664)
            write_index(root, when.date())
        for e in entities:
            if e:
                history(vault_dir, e, line, when.date())
    except Exception as exc:  # noqa: BLE001 - the vault is best-effort
        logger.warning("vault log failed: %s", exc)


def _row(ep: Episode, tz: str) -> str:
    start, end = to_local(ep.start, tz), to_local(ep.end, tz)
    state = "asleep" if ep.asleep else "at rest"
    ai = (ep.llm_assessment or "").replace("_", " ") or "not assessed"
    felt = {True: "yes", False: "no"}.get(ep.felt, "?")
    return (
        f"| {start:%H:%M}–{end:%H:%M} | {round(ep.duration_min)} min | {ep.peak_hr} | "
        f"{round(ep.baseline_hr)} | {state} | {ep.steps_in_window} | {ai} | {felt} |"
    )


def write_day(vault_dir: str, name: str, day: date, episodes: Sequence[Episode], tz: str, today: date) -> Path | None:
    """Rewrite ``Episodes/<day>.md`` from the database (History kept) and refresh Home."""
    try:
        root = Path(vault_dir)
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
        notes = [f"- **{to_local(e.start, tz):%H:%M}** {_clean(e.doctor_note, 400)}" for e in eps if e.doctor_note]
        if notes:
            lines += ["## Notes for the doctor", *notes, ""]
        lines.append("Wrist-sensor readings for a symptom diary, not a diagnosis. Back to [[Home]].")
        path = root / "Episodes" / f"{day.isoformat()}.md"
        _write(path, _with_history(path, lines))
        write_index(root, today)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("vault episode note failed: %s", exc)
        return None


def write_note(vault_dir: str | None, folder: str, day: date, title: str, body: Iterable[str], today: date) -> None:
    """Rewrite ``<folder>/<day>.md`` (e.g. ``Days``) with ``body``; History kept; Home refreshed."""
    if not vault_dir:
        return
    try:
        root = Path(vault_dir)
        path = root / folder / f"{day.isoformat()}.md"
        head = ["---", "tags: [active]", f"date: {day.isoformat()}", f"updated: {today.isoformat()}", "---", f"# {title}", ""]
        _write(path, _with_history(path, [*head, *body, "", "Back to [[Home]]."]))
        write_index(root, today)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vault %s note failed: %s", folder, exc)


def _notes(root: Path, folder: str) -> list[Path]:
    """Date-named notes under ``folder`` (any depth), newest first."""
    return sorted((p for p in (root / folder).rglob("*.md") if _DATE_RE.match(p.stem)),
                  key=lambda p: p.stem, reverse=True)


def _stems(root: Path, folder: str) -> list[str]:
    return [p.stem for p in _notes(root, folder)]


def _link(root: Path, path: Path) -> str:
    return f"[[{path.relative_to(root).with_suffix('').as_posix()}|{path.stem}]]"


def write_index(root: Path, today: date) -> None:
    """``Home.md`` (MOC): this month's Activity folder, latest activity and days, episodes by
    month, alerts, metrics, other notes."""
    lines = [
        "---", "tags: [active]", f"updated: {today.isoformat()}", "---",
        "# Garmin health monitor",
        "",
        "Written automatically by the Garmin monitor: `Activity/YYYY/MM/` is the movement log (one note "
        "per day, one line per event), `Episodes/` the possible palpitations per day, `Days/` the daily "
        "numbers and coaching, `Metrics/` one note per metric, `Alerts/` one note per alert type. "
        "Recent notes are read back into the AI prompts as memory.",
        "",
        f"This month: `Activity/{today:%Y/%m}/`",
    ]
    if acts := _notes(root, "Activity")[:14]:
        lines += ["", "## Latest activity", *(f"- {_link(root, p)}" for p in acts)]
    if days := _stems(root, "Days")[:14]:
        lines += ["", "## Days", *(f"- [[Days/{s}|{s}]]" for s in days)]
    month = None
    for s in _stems(root, "Episodes"):
        m = _COUNT_RE.search((root / "Episodes" / f"{s}.md").read_text(encoding="utf-8"))
        count = int(m.group(1)) if m else 0
        if count == 0:
            continue
        if s[:7] != month:
            month = s[:7]
            lines += ["", f"## Episodes · {date.fromisoformat(s):%B %Y}"]
        lines.append(f"- [[Episodes/{s}|{s}]]: {count}")
    for folder in NAMED_DIRS:
        if named := sorted((root / folder).glob("*.md")):
            lines += ["", f"## {folder}", *(f"- {_link(root, p)}" for p in named)]
    if other := sorted(p.stem for p in root.glob("*.md") if p.name != "Home.md"):
        lines += ["", "## Other notes", *(f"- [[{s}]]" for s in other)]
    _write(root / "Home.md", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------- read


def _body(path: Path) -> list[str]:
    """Note lines without frontmatter, the History section (it mirrors Activity) or blanks."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        text = text[end + 4:] if end >= 0 else text
    text = text.split("\n" + HISTORY)[0]
    return [ln for ln in text.splitlines() if ln.strip() and "Back to [[Home]]" not in ln]


def _take(lines: list[str], budget: int) -> list[str]:
    """Leading ``lines`` whose joined length (newlines included) fits in ``budget``."""
    out: list[str] = []
    for ln in lines:
        budget -= len(ln) + 1
        if budget < -1:  # the last line needs no newline
            break
        out.append(ln)
    return out


def memory(vault_dir: str | None, cap: int = MEMORY_CHARS) -> str:
    """Newest-first excerpt of the vault for a prompt, at most ``cap`` chars ("" when empty/off).

    First half of the budget: Activity lines, newest first (prefixed with their date). The rest:
    entity notes (Episodes, Days), newest first. Never raises.
    """
    if not vault_dir:
        return ""
    try:
        root = Path(vault_dir)
        acts = [f"- {p.stem} {ln[2:]}" for p in _notes(root, "Activity")[:RECENT_NOTES]
                for ln in reversed(p.read_text(encoding="utf-8").splitlines())
                if ln.startswith("- ")]
        out = _take(["Recent activity (newest first):", *acts] if acts else [], cap // 2)
        notes = sorted(((s, d) for d in ENTITY_DIRS for s in _stems(root, d)), reverse=True)[:RECENT_NOTES]
        ents = [ln for s, d in notes for ln in (f"[{d}/{s}]", *_body(root / d / f"{s}.md"))]
        out += _take(ents, cap - len("\n".join(out)) - 1)
        return "\n".join(out)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vault memory read failed: %s", exc)
        return ""
