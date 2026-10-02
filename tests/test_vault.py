"""Vault movement log + memory: line format, capped newest-first excerpt, best-effort I/O,
and the excerpt reaching the LLM prompts."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from garmin_health_monitor import vault
from garmin_health_monitor.utils import get_tz
from tests.conftest import DAY, TZ
from tests.test_service import svc  # noqa: F401 - fixture

LINE_RE = re.compile(r"^- \d\d:\d\d \S+ \*\*[^*]+\*\*( · [^\n]+)?$")


def at(day: date, h: int, m: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, h, m, tzinfo=get_tz(TZ))


def test_activity_line_format_and_entity_history(tmp_path):
    root = tmp_path / "v"
    vault.log(str(root), at(DAY, 14, 5), "💓", "Possible palpitation",
              "14:02–14:20, peak 132 bpm\nsecond line [[x]]", f"Episodes/{DAY}")
    vault.log(str(root), at(DAY, 14, 6), "🚨", "Episode alert sent")
    text = (root / "Activity" / "2026" / "09" / f"{DAY}.md").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert lines == [
        f"- 14:05 💓 **Possible palpitation** · 14:02–14:20, peak 132 bpm second line [x] · [[Episodes/{DAY}]]",
        "- 14:06 🚨 **Episode alert sent**",
    ]
    assert all(LINE_RE.match(ln) for ln in lines)
    assert text.startswith("---\ntags: [activity]\n")
    note = (root / "Episodes" / f"{DAY}.md").read_text(encoding="utf-8")
    assert note.rstrip().endswith(lines[0]) and "## History" in note
    home = (root / "Home.md").read_text(encoding="utf-8")
    assert f"[[Activity/2026/09/{DAY}|{DAY}]]" in home and "This month: `Activity/2026/09/`" in home


def test_flat_activity_notes_migrate_and_named_entities_keep_history(tmp_path):
    root = tmp_path / "v"
    (root / "Activity").mkdir(parents=True)
    (root / "Activity" / "2026-08-30.md").write_text("# old\n- 08:00 x **y**\n", encoding="utf-8")
    (root / "Activity" / "README.md").write_text("James's own note stays", encoding="utf-8")
    assert vault.migrate(str(root)) == ["2026-08-30"]
    assert (root / "Activity" / "2026" / "08" / "2026-08-30.md").exists()
    assert not (root / "Activity" / "2026-08-30.md").exists() and (root / "Activity" / "README.md").exists()
    assert vault.migrate(str(root)) == [] and vault.migrate(None) == []
    # one Activity line can feed several entity notes (Episodes + Alerts)
    vault.log(str(root), at(DAY, 9), "🚨", "Episode alert sent", "peak 132", [f"Episodes/{DAY}", "Alerts/episode"])
    alerts = (root / "Alerts" / "episode.md").read_text(encoding="utf-8")
    assert alerts.startswith("---\ntags: [active]\n") and "# episode\n\n## History\n- 09:00 🚨" in alerts
    assert "## History\n- 09:00 🚨" in (root / "Episodes" / f"{DAY}.md").read_text(encoding="utf-8")
    # Metrics: one line per day, a re-run replaces that day's line, James's text above History stays
    vault.history(str(root), "Metrics/resting_hr", f"- {DAY} 52", DAY, replace=f"- {DAY} ")
    note = root / "Metrics" / "resting_hr.md"
    note.write_text(note.read_text(encoding="utf-8").replace("# resting_hr", "# resting_hr\nMy note."), encoding="utf-8")
    vault.history(str(root), "Metrics/resting_hr", f"- {DAY} 54", DAY, replace=f"- {DAY} ")
    vault.history(str(root), "Metrics/resting_hr", f"- {DAY + timedelta(days=1)} 50", DAY)
    text = note.read_text(encoding="utf-8")
    assert text.endswith(f"# resting_hr\nMy note.\n\n## History\n- {DAY} 54\n- {DAY + timedelta(days=1)} 50\n")
    assert "- 2026-08-30 08:00 x **y**" in vault.memory(str(root))  # migrated notes still count as memory
    home = (root / "Home.md").read_text(encoding="utf-8")
    assert "[[Alerts/episode|episode]]" in home and "[[Metrics/resting_hr|resting_hr]]" in home


def test_entity_rewrite_keeps_history(tmp_path):
    root = tmp_path / "v"
    vault.write_note(str(root), "Days", DAY, "Sat", ["## Numbers", "steps 9000"], DAY)
    vault.log(str(root), at(DAY, 22), "🏋️", "Coaching", "first", f"Days/{DAY}")
    vault.write_note(str(root), "Days", DAY, "Sat", ["## Numbers", "steps 9500"], DAY)
    text = (root / "Days" / f"{DAY}.md").read_text(encoding="utf-8")
    assert "steps 9500" in text and "steps 9000" not in text
    assert text.count("**Coaching** · first") == 1


def test_memory_is_capped_and_newest_first(tmp_path):
    root = tmp_path / "v"
    for i in range(5):
        d = DAY - timedelta(days=i)
        for h in range(8, 20):
            vault.log(str(root), at(d, h), "📨", "Report sent", f"day {d} hour {h} " + "x" * 40)
        vault.write_note(str(root), "Days", d, f"Day {d}", [f"numbers for {d}"], DAY)
    mem = vault.memory(str(root), cap=1500)
    assert 0 < len(mem) <= 1500
    lines = mem.splitlines()
    assert lines[0] == "Recent activity (newest first):"
    assert lines[1].startswith(f"- {DAY} 19:00 📨") and lines[2].startswith(f"- {DAY} 18:00")
    days = [ln for ln in lines if ln.startswith("[Days/")]
    assert days and days[0] == f"[Days/{DAY}]"  # newest entity first, after the activity half
    assert "## History" not in mem and "tags:" not in mem
    assert len(vault.memory(str(root))) <= vault.MEMORY_CHARS
    assert vault.memory(None) == "" and vault.memory(str(tmp_path / "empty")) == ""


def test_vault_errors_never_raise(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file in the way", encoding="utf-8")
    bad = str(blocker / "vault")
    vault.log(bad, at(DAY, 9), "💊", "Medicine taken", entity=f"Episodes/{DAY}")
    assert vault.write_day(bad, "Dad", DAY, [], TZ, DAY) is None
    vault.write_note(bad, "Days", DAY, "x", ["y"], DAY)
    assert vault.memory(bad) == ""


def test_poll_with_broken_vault_still_alerts(svc, tmp_path):  # noqa: F811
    service, profile, _, _ = svc
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    service.config.vault_dir = str(blocker / "vault")
    result = service.poll(profile)
    assert result.error is None and result.to_notify


def test_memory_reaches_episode_and_coaching_prompts(svc, tmp_path):  # noqa: F811
    service, profile, _, _ = svc
    root = tmp_path / "vault"
    service.config.vault_dir = str(root)
    vault.log(str(root), at(DAY - timedelta(days=1), 21), "🚨", "Episode alert sent", "MARKER-YESTERDAY")
    service.poll(profile)
    prompts = [u for _, u in service.llm.calls]
    assert prompts and all("Your own log (Obsidian vault" in u for u in prompts)
    assert all("MARKER-YESTERDAY" in u for u in prompts)
    # the second assessment already sees the first one's Activity line
    assert any("**AI view**" in u for u in prompts[1:])
    activity = (root / "Activity" / "2026" / "09" / f"{DAY}.md").read_text(encoding="utf-8")
    assert "**Possible palpitation**" in activity and "**AI view**" in activity

    service.llm.calls.clear()
    service.evening_summary_text(profile)
    coaching = [u for _, u in service.llm.calls if "do_more" in u]
    assert coaching and "**Possible palpitation**" in coaching[0]
    assert "## Coaching (fake-model)" in (root / "Days" / f"{DAY}.md").read_text(encoding="utf-8")
    metrics = list((root / "Metrics").glob("*.md"))
    assert metrics and all(f"- {DAY} " in m.read_text(encoding="utf-8") for m in metrics)
