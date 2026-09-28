"""Doctor report: PDF + CSV generated from a populated Storage."""

from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from garmin_health_monitor import report
from garmin_health_monitor.models import (
    ASSESSMENT_EXERTION,
    ASSESSMENT_POSSIBLE,
    EPISODE_SOURCE_MANUAL,
    Episode,
    SymptomReport,
)
from garmin_health_monitor.palpitations import detect_episodes

from .conftest import DAY, make_profile, make_snapshot

START = DAY - timedelta(days=6)


@pytest.fixture
def populated(storage):
    """Seven days of snapshots; episodes on two days; assessments, felt answers and two symptom reports."""
    episodes = []
    for i in range(7):
        day = START + timedelta(days=i)
        spec = [(10, 0, 8, 125), (15, 30, 6, 112)] if i in (2, 5) else None
        snap = make_snapshot(
            profile="Dad",
            day=day,
            seed=i + 1,
            episodes=spec,
            activities=[(8, 30, 105)] if i == 2 else None,
        )
        storage.save_snapshot(snap)
        for ep in detect_episodes(snap):
            stored, _ = storage.upsert_episode(ep)
            episodes.append(stored)
    assert len(episodes) >= 2
    storage.update_episode_assessment(
        episodes[0].id, ASSESSMENT_POSSIBLE, 0.7, "reasoning", "Rate rose from 64 to 125 bpm at rest for 8 minutes.", "test-model"
    )
    storage.set_episode_felt(episodes[0].id, True, "felt fluttering")
    storage.update_episode_assessment(episodes[1].id, ASSESSMENT_EXERTION, 0.6, "reasoning", "Followed a walk.", "test-model")
    storage.set_episode_felt(episodes[1].id, False)
    first = episodes[0]
    storage.add_symptom(
        SymptomReport(
            profile="Dad",
            reported_at=first.start + timedelta(minutes=5),
            event_time=first.start + timedelta(minutes=3),
            note="fluttering while reading",
            hr_at_time=118,
            baseline_hr=64.0,
            episode_id=first.id,
        )
    )
    storage.add_symptom(
        SymptomReport(
            profile="Dad",
            reported_at=first.start + timedelta(days=1, hours=9),
            event_time=first.start + timedelta(days=1, hours=9),
            note="skipped beat after dinner",
            source="button",
        )
    )
    return storage, episodes


def _pdf_pages(path) -> int:
    """Count page objects in a matplotlib PDF (``/Type /Page`` but not ``/Type /Pages``)."""
    return len(re.findall(rb"/Type /Page\b(?!s)", path.read_bytes()))


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def test_generate_doctor_report_writes_pdf_and_csv(populated, tmp_path):
    storage, episodes = populated
    out = tmp_path / "reports"  # created on demand
    files = report.generate_doctor_report(make_profile(), storage, START, DAY, out, narrative="Two days had episodes.")

    assert files.pdf_path == out / f"dad-palpitation-report-{START.isoformat()}_{DAY.isoformat()}.pdf"
    assert files.csv_path == out / f"dad-palpitation-report-{START.isoformat()}_{DAY.isoformat()}.csv"
    assert files.pdf_path.exists() and files.csv_path.exists()
    assert files.pdf_path.read_bytes().startswith(b"%PDF")
    assert files.n_episodes == len(episodes)
    assert files.n_symptoms == 2

    # cover, frequency, episodes, symptoms, trend, zoom charts (2 per page)
    assert _pdf_pages(files.pdf_path) >= 5 + (len(episodes) + 1) // 2


def test_summary_text(populated, tmp_path):
    storage, episodes = populated
    files = report.generate_doctor_report(make_profile(), storage, START, DAY, tmp_path)
    s = files.summary
    assert s.startswith("Heart-rate excursion diary for Dad.")
    assert f"Detected episodes: {len(episodes)}" in s
    assert "per week" in s
    assert "Most common hours: 10:00-11:00" in s  # local (Singapore) hour, not UTC 02:00
    assert "Felt by Dad: 1 of" in s and "not noticed: 1" in s
    assert "Symptom reports logged by the person: 2, of which 1 matched a detected episode." in s
    assert "possible palpitation 1" in s and "likely exertion 1" in s
    assert "n/a" not in s


def test_csv_rows_and_columns(populated, tmp_path):
    storage, episodes = populated
    files = report.generate_doctor_report(make_profile(), storage, START, DAY, tmp_path)
    header, rows = _read_csv(files.csv_path)
    assert header == report.CSV_COLUMNS
    assert len(rows) == files.n_episodes + files.n_symptoms
    assert Counter(r["record_type"] for r in rows) == {"episode": len(episodes), "symptom": 2}

    ep_rows = [r for r in rows if r["record_type"] == "episode"]
    first = ep_rows[0]
    assert first["date"] == (START + timedelta(days=2)).isoformat()
    assert first["start_local"] == f"{(START + timedelta(days=2)).isoformat()} 10:00"
    assert first["peak_hr"] == str(episodes[0].peak_hr)
    assert first["kind"] == episodes[0].kind
    assert first["asleep"] == "no"
    assert first["felt"] == "yes"
    assert first["llm_assessment"] == ASSESSMENT_POSSIBLE
    assert first["llm_confidence"] == "0.70"
    assert first["doctor_note"].startswith("Rate rose")
    assert first["note"] == "felt fluttering"
    assert ep_rows[1]["felt"] == "no"
    assert ep_rows[2]["felt"] == ""  # not answered

    sym_rows = [r for r in rows if r["record_type"] == "symptom"]
    assert sym_rows[0]["peak_hr"] == "118"
    assert sym_rows[0]["baseline_hr"] == "64.0"
    assert sym_rows[0]["delta_hr"] == "54.0"
    assert sym_rows[0]["note"] == "fluttering while reading"
    assert sym_rows[0]["start_local"] == f"{(START + timedelta(days=2)).isoformat()} 10:03"
    assert sym_rows[1]["peak_hr"] == ""
    assert sym_rows[1]["note"] == "skipped beat after dinner"

    # rows are interleaved chronologically
    starts = [r["start_local"] for r in rows]
    assert starts == sorted(starts)


def test_report_for_empty_period(storage, tmp_path):
    start = DAY + timedelta(days=30)
    files = report.generate_doctor_report(make_profile(), storage, start, start + timedelta(days=6), tmp_path)
    assert files.pdf_path.read_bytes().startswith(b"%PDF")
    assert files.n_episodes == 0 and files.n_symptoms == 0
    header, rows = _read_csv(files.csv_path)
    assert header == report.CSV_COLUMNS and rows == []
    assert "Detected episodes: 0." in files.summary
    assert "Symptom reports logged by the person: 0." in files.summary


def test_report_period_filters_by_local_day(populated, tmp_path):
    storage, _ = populated
    # first two days carry snapshots but no episodes
    files = report.generate_doctor_report(make_profile(), storage, START, START + timedelta(days=1), tmp_path)
    assert files.n_episodes == 0 and files.n_symptoms == 0
    assert files.pdf_path.read_bytes().startswith(b"%PDF")
    # the day with the episodes alone
    files = report.generate_doctor_report(make_profile(), storage, START + timedelta(days=2), START + timedelta(days=2), tmp_path)
    assert files.n_episodes == 2 and files.n_symptoms == 1


def test_reversed_dates_are_swapped(populated, tmp_path):
    storage, episodes = populated
    files = report.generate_doctor_report(make_profile(), storage, DAY, START, tmp_path)
    assert files.pdf_path.name == f"dad-palpitation-report-{START.isoformat()}_{DAY.isoformat()}.pdf"
    assert files.n_episodes == len(episodes)


def test_manual_episode_without_samples(storage, tmp_path):
    start = datetime(2026, 9, 25, 3, 0, tzinfo=UTC)
    ep = Episode(
        profile="Dad",
        start=start,
        end=start + timedelta(minutes=5),
        duration_min=5.0,
        peak_hr=110,
        mean_hr=105.0,
        baseline_hr=60.0,
        delta_hr=50.0,
        confidence=0.5,
        source=EPISODE_SOURCE_MANUAL,
        fingerprint="manual-1",
    )
    storage.upsert_episode(ep)
    files = report.generate_doctor_report(make_profile(), storage, START, DAY, tmp_path)
    assert files.n_episodes == 1
    assert files.pdf_path.read_bytes().startswith(b"%PDF")
    _, rows = _read_csv(files.csv_path)
    assert rows[0]["record_type"] == "episode" and rows[0]["peak_hr"] == "110"
    assert rows[0]["llm_assessment"] == "" and rows[0]["felt"] == ""


def test_many_episodes_paginate(storage, tmp_path):
    """More than a page of episodes and more than 12 zoom charts still produce a PDF."""
    base = datetime(2026, 9, 1, 2, 0, tzinfo=UTC)
    for i in range(40):
        st = base + timedelta(hours=7 * i)
        storage.upsert_episode(
            Episode(
                profile="Dad",
                start=st,
                end=st + timedelta(minutes=6),
                duration_min=6.0,
                peak_hr=100 + i % 30,
                mean_hr=100.0,
                baseline_hr=60.0,
                delta_hr=40.0 + i % 30,
                confidence=0.3 + (i % 7) / 10,
                asleep=i % 4 == 0,
                fingerprint=f"fp-{i}",
                doctor_note="A sustained rise at rest with no steps recorded; the person did not answer." * 2,
            )
        )
    files = report.generate_doctor_report(make_profile(), storage, DAY - timedelta(days=29), DAY, tmp_path)
    assert files.n_episodes == 40
    assert files.pdf_path.read_bytes().startswith(b"%PDF")
    assert _pdf_pages(files.pdf_path) >= 5 + 6
    _, rows = _read_csv(files.csv_path)
    assert len(rows) == 40
