"""SQLite persistence (standard library only).

Tables
------
daily_snapshots  one row per profile+day: normalised columns for quick queries
                 plus the raw Garmin payloads as JSON so history can be
                 re-processed when the detector improves.
hr_samples       intraday heart rate (2-minute samples).
episodes         candidate palpitation episodes (detector, Garmin alert or manual).
symptoms         person-reported palpitations (``/palp``, inline buttons).
analyses         Ollama outputs (daily / weekly coaching, episode assessments).
notifications    dedupe keys for alerts that were already sent.
kv               small key/value state (last sync timestamps etc).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import DaySnapshot, Episode, HRSample, SymptomReport
from .utils import now_utc, to_utc

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_snapshots (
    profile TEXT NOT NULL,
    day TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    tz TEXT,
    total_steps INTEGER, step_goal INTEGER, distance_m REAL,
    total_kcal REAL, active_kcal REAL,
    min_hr INTEGER, max_hr INTEGER, resting_hr INTEGER, resting_hr_7d_avg INTEGER,
    sleep_seconds INTEGER, sleep_score INTEGER, sleep_deep_seconds INTEGER,
    sleep_light_seconds INTEGER, sleep_rem_seconds INTEGER, sleep_awake_seconds INTEGER,
    sleep_start TEXT, sleep_end TEXT,
    hrv_last_night REAL, hrv_weekly_avg REAL, hrv_status TEXT,
    avg_stress INTEGER, max_stress INTEGER,
    high_stress_seconds INTEGER, rest_stress_seconds INTEGER,
    body_battery_high INTEGER, body_battery_low INTEGER, body_battery_latest INTEGER,
    moderate_intensity_min INTEGER, vigorous_intensity_min INTEGER,
    active_seconds INTEGER, highly_active_seconds INTEGER, sedentary_seconds INTEGER,
    floors_up REAL,
    avg_spo2 REAL, lowest_spo2 INTEGER,
    avg_waking_respiration REAL, avg_sleep_respiration REAL,
    readiness_score INTEGER, readiness_level TEXT,
    abnormal_hr_alerts INTEGER,
    activities_count INTEGER, activities_json TEXT,
    last_sync TEXT,
    errors_json TEXT,
    raw_json TEXT,
    PRIMARY KEY (profile, day)
);

CREATE TABLE IF NOT EXISTS hr_samples (
    profile TEXT NOT NULL,
    ts TEXT NOT NULL,
    hr INTEGER NOT NULL,
    PRIMARY KEY (profile, ts)
);
CREATE INDEX IF NOT EXISTS idx_hr_samples_profile_ts ON hr_samples(profile, ts);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE,
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    duration_min REAL NOT NULL,
    peak_hr INTEGER NOT NULL,
    mean_hr REAL NOT NULL,
    baseline_hr REAL NOT NULL,
    delta_hr REAL NOT NULL,
    max_jump_bpm INTEGER DEFAULT 0,
    asleep INTEGER DEFAULT 0,
    steps_in_window INTEGER DEFAULT 0,
    activity_level TEXT,
    stress_avg REAL,
    movement_fraction REAL DEFAULT 0,
    confidence REAL DEFAULT 0,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    samples_json TEXT,
    llm_assessment TEXT,
    llm_confidence REAL,
    llm_reasoning TEXT,
    doctor_note TEXT,
    llm_model TEXT,
    felt INTEGER,
    notes TEXT,
    notified_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_profile_start ON episodes(profile, start);

CREATE TABLE IF NOT EXISTS symptoms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    event_time TEXT NOT NULL,
    note TEXT,
    hr_at_time INTEGER,
    baseline_hr REAL,
    episode_id INTEGER,
    chat_id INTEGER,
    source TEXT DEFAULT 'command',
    extracted TEXT
);
CREATE INDEX IF NOT EXISTS idx_symptoms_profile_time ON symptoms(profile, event_time);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    kind TEXT NOT NULL,
    day TEXT NOT NULL,
    model TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_lookup ON analyses(profile, kind, day);

CREATE TABLE IF NOT EXISTS notifications (
    key TEXT PRIMARY KEY,
    profile TEXT,
    sent_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return to_utc(dt).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    return str(obj)


class Storage:
    """Thread-safe SQLite wrapper.  One connection per thread via a lock."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            try:  # databases created before the extracted column existed
                self._conn.execute("ALTER TABLE symptoms ADD COLUMN extracted TEXT")
            except sqlite3.OperationalError:
                pass

    # -- low level -------------------------------------------------------

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- snapshots -------------------------------------------------------

    def save_snapshot(self, snap: DaySnapshot, keep_raw: bool = True) -> None:
        s = snap.summary
        sl = snap.sleep
        hrv = snap.hrv
        rd = snap.readiness
        resp = snap.respiration
        activities = [
            {
                "name": a.name,
                "type": a.type_key,
                "start": _iso(a.start),
                "end": _iso(a.end),
                "duration_s": a.duration_s,
                "avg_hr": a.avg_hr,
                "max_hr": a.max_hr,
                "distance_m": a.distance_m,
                "calories": a.calories,
                "source": a.source,
                "activity_id": a.activity_id,
            }
            for a in snap.activities
        ]
        row = {
            "profile": snap.profile,
            "day": snap.date_str,
            "fetched_at": _iso(snap.fetched_at),
            "tz": snap.tz,
            "total_steps": s.total_steps,
            "step_goal": s.step_goal,
            "distance_m": s.distance_m,
            "total_kcal": s.total_kcal,
            "active_kcal": s.active_kcal,
            "min_hr": s.min_hr,
            "max_hr": s.max_hr,
            "resting_hr": s.resting_hr,
            "resting_hr_7d_avg": s.resting_hr_7d_avg,
            "sleep_seconds": sl.total_seconds if sl else None,
            "sleep_score": sl.score if sl else None,
            "sleep_deep_seconds": sl.deep_seconds if sl else None,
            "sleep_light_seconds": sl.light_seconds if sl else None,
            "sleep_rem_seconds": sl.rem_seconds if sl else None,
            "sleep_awake_seconds": sl.awake_seconds if sl else None,
            "sleep_start": _iso(sl.start) if sl else None,
            "sleep_end": _iso(sl.end) if sl else None,
            "hrv_last_night": hrv.last_night_avg if hrv else None,
            "hrv_weekly_avg": hrv.weekly_avg if hrv else None,
            "hrv_status": hrv.status if hrv else None,
            "avg_stress": s.avg_stress,
            "max_stress": s.max_stress,
            "high_stress_seconds": s.high_stress_seconds,
            "rest_stress_seconds": s.rest_stress_seconds,
            "body_battery_high": s.body_battery_high,
            "body_battery_low": s.body_battery_low,
            "body_battery_latest": s.body_battery_latest,
            "moderate_intensity_min": s.moderate_intensity_min,
            "vigorous_intensity_min": s.vigorous_intensity_min,
            "active_seconds": s.active_seconds,
            "highly_active_seconds": s.highly_active_seconds,
            "sedentary_seconds": s.sedentary_seconds,
            "floors_up": s.floors_up,
            "avg_spo2": (snap.spo2.average if snap.spo2 and snap.spo2.average is not None else s.avg_spo2),
            "lowest_spo2": (snap.spo2.lowest if snap.spo2 and snap.spo2.lowest is not None else s.lowest_spo2),
            "avg_waking_respiration": (resp.avg_waking if resp else s.avg_waking_respiration),
            "avg_sleep_respiration": resp.avg_sleep if resp else None,
            "readiness_score": rd.score if rd else None,
            "readiness_level": rd.level if rd else None,
            "abnormal_hr_alerts": s.abnormal_hr_alerts,
            "activities_count": len(snap.activities),
            "activities_json": json.dumps(activities, default=_json_default),
            "last_sync": _iso(s.last_sync),
            "errors_json": json.dumps(snap.errors),
            "raw_json": json.dumps(snap.raw, default=_json_default) if keep_raw else None,
        }
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        updates = ", ".join(f"{k}=excluded.{k}" for k in row if k not in ("profile", "day"))
        with self.tx() as c:
            c.execute(
                f"INSERT INTO daily_snapshots ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(profile, day) DO UPDATE SET {updates}",
                row,
            )
        if snap.hr:
            self.save_hr_samples(snap.profile, snap.hr)

    def get_snapshot_row(self, profile: str, day: date | str) -> dict[str, Any] | None:
        day_str = day if isinstance(day, str) else day.isoformat()
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM daily_snapshots WHERE profile=? AND day=?", (profile, day_str)
            )
            row = cur.fetchone()
        return self._snapshot_row(row) if row else None

    def get_snapshot_rows(self, profile: str, start: date, end: date) -> list[dict[str, Any]]:
        """Normalised rows (no raw JSON) for ``start..end`` inclusive, ascending."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM daily_snapshots WHERE profile=? AND day BETWEEN ? AND ? ORDER BY day",
                (profile, start.isoformat(), end.isoformat()),
            )
            rows = cur.fetchall()
        return [self._snapshot_row(r) for r in rows]

    def get_snapshot_raw(self, profile: str, day: date | str) -> dict[str, Any] | None:
        day_str = day if isinstance(day, str) else day.isoformat()
        with self._lock:
            cur = self._conn.execute(
                "SELECT raw_json FROM daily_snapshots WHERE profile=? AND day=?", (profile, day_str)
            )
            row = cur.fetchone()
        if not row or not row["raw_json"]:
            return None
        return json.loads(row["raw_json"])

    @staticmethod
    def _snapshot_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d.pop("raw_json", None)
        d["activities"] = json.loads(d.pop("activities_json") or "[]")
        d["errors"] = json.loads(d.pop("errors_json") or "{}")
        for key in ("sleep_start", "sleep_end", "last_sync", "fetched_at"):
            d[key] = _parse_iso(d.get(key))
        return d

    # -- heart rate samples ---------------------------------------------

    def save_hr_samples(self, profile: str, samples: Iterable[HRSample]) -> int:
        rows = [(profile, _iso(s.ts), int(s.hr)) for s in samples if s.hr is not None]
        if not rows:
            return 0
        with self.tx() as c:
            c.executemany(
                "INSERT INTO hr_samples(profile, ts, hr) VALUES (?, ?, ?) "
                "ON CONFLICT(profile, ts) DO UPDATE SET hr=excluded.hr",
                rows,
            )
        return len(rows)

    def get_hr_samples(self, profile: str, start: datetime, end: datetime) -> list[HRSample]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT ts, hr FROM hr_samples WHERE profile=? AND ts>=? AND ts<? ORDER BY ts",
                (profile, _iso(start), _iso(end)),
            )
            rows = cur.fetchall()
        out: list[HRSample] = []
        for r in rows:
            ts = _parse_iso(r["ts"])
            if ts is not None:
                out.append(HRSample(ts=ts, hr=int(r["hr"])))
        return out

    def hr_near(self, profile: str, when: datetime, window_minutes: int = 6) -> HRSample | None:
        """Closest stored HR sample within +/- ``window_minutes`` of ``when``."""
        lo = when - timedelta(minutes=window_minutes)
        hi = when + timedelta(minutes=window_minutes)
        samples = self.get_hr_samples(profile, lo, hi)
        if not samples:
            return None
        return min(samples, key=lambda s: abs((s.ts - to_utc(when)).total_seconds()))

    # -- episodes --------------------------------------------------------

    def _episode_from_row(self, row: sqlite3.Row) -> Episode:
        samples = []
        if row["samples_json"]:
            for s in json.loads(row["samples_json"]):
                ts = _parse_iso(s.get("ts"))
                if ts is not None:
                    samples.append(HRSample(ts=ts, hr=int(s["hr"])))
        felt = row["felt"]
        return Episode(
            id=row["id"],
            profile=row["profile"],
            fingerprint=row["fingerprint"],
            start=_parse_iso(row["start"]) or now_utc(),
            end=_parse_iso(row["end"]) or now_utc(),
            duration_min=float(row["duration_min"]),
            peak_hr=int(row["peak_hr"]),
            mean_hr=float(row["mean_hr"]),
            baseline_hr=float(row["baseline_hr"]),
            delta_hr=float(row["delta_hr"]),
            max_jump_bpm=int(row["max_jump_bpm"] or 0),
            asleep=bool(row["asleep"]),
            steps_in_window=int(row["steps_in_window"] or 0),
            activity_level=row["activity_level"] or "unknown",
            stress_avg=row["stress_avg"],
            movement_fraction=float(row["movement_fraction"] or 0.0),
            confidence=float(row["confidence"] or 0.0),
            kind=row["kind"],
            source=row["source"],
            samples=samples,
            llm_assessment=row["llm_assessment"],
            llm_confidence=row["llm_confidence"],
            llm_reasoning=row["llm_reasoning"],
            doctor_note=row["doctor_note"],
            llm_model=row["llm_model"],
            felt=None if felt is None else bool(felt),
            notes=row["notes"],
            notified_at=_parse_iso(row["notified_at"]),
            created_at=_parse_iso(row["created_at"]),
            updated_at=_parse_iso(row["updated_at"]),
        )

    def find_overlapping_episode(
        self, profile: str, start: datetime, end: datetime, margin_minutes: int = 6
    ) -> Episode | None:
        """Existing episode of the same profile whose window overlaps ``[start, end]``."""
        lo = _iso(start - timedelta(minutes=margin_minutes))
        hi = _iso(end + timedelta(minutes=margin_minutes))
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM episodes WHERE profile=? AND source!='manual' AND start<=? AND end>=? "
                "ORDER BY start LIMIT 1",
                (profile, hi, lo),
            )
            row = cur.fetchone()
        return self._episode_from_row(row) if row else None

    def upsert_episode(self, ep: Episode) -> tuple[Episode, bool]:
        """Insert a detected episode or merge it into an overlapping stored one.

        Returns ``(stored_episode, is_new)``.  Merging keeps the stored id, LLM
        assessment and person feedback, and widens the window / raises the peak.
        """
        now = now_utc()
        existing = self.find_overlapping_episode(ep.profile, ep.start, ep.end)
        if existing is None:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT * FROM episodes WHERE fingerprint=?", (ep.fingerprint,)
                )
                row = cur.fetchone()
            existing = self._episode_from_row(row) if row else None
        if existing is None:
            with self.tx() as c:
                cur = c.execute(
                    """INSERT INTO episodes(profile, fingerprint, start, end, duration_min, peak_hr,
                        mean_hr, baseline_hr, delta_hr, max_jump_bpm, asleep, steps_in_window,
                        activity_level, stress_avg, movement_fraction, confidence, kind, source,
                        samples_json, llm_assessment, llm_confidence, llm_reasoning, doctor_note,
                        llm_model, felt, notes, notified_at, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ep.profile,
                        ep.fingerprint,
                        _iso(ep.start),
                        _iso(ep.end),
                        ep.duration_min,
                        ep.peak_hr,
                        ep.mean_hr,
                        ep.baseline_hr,
                        ep.delta_hr,
                        ep.max_jump_bpm,
                        int(ep.asleep),
                        ep.steps_in_window,
                        ep.activity_level,
                        ep.stress_avg,
                        ep.movement_fraction,
                        ep.confidence,
                        ep.kind,
                        ep.source,
                        json.dumps([{"ts": _iso(s.ts), "hr": s.hr} for s in ep.samples]),
                        ep.llm_assessment,
                        ep.llm_confidence,
                        ep.llm_reasoning,
                        ep.doctor_note,
                        ep.llm_model,
                        None if ep.felt is None else int(ep.felt),
                        ep.notes,
                        _iso(ep.notified_at),
                        _iso(now),
                        _iso(now),
                    ),
                )
                ep.id = cur.lastrowid
            ep.created_at = now
            ep.updated_at = now
            return ep, True

        # merge
        merged_start = min(existing.start, ep.start)
        merged_end = max(existing.end, ep.end)
        by_ts: dict[str, HRSample] = {_iso(s.ts) or "": s for s in existing.samples}
        for s in ep.samples:
            by_ts[_iso(s.ts) or ""] = s
        samples = sorted(by_ts.values(), key=lambda s: s.ts)
        peak = max(existing.peak_hr, ep.peak_hr)
        mean_hr = (sum(s.hr for s in samples) / len(samples)) if samples else max(existing.mean_hr, ep.mean_hr)
        baseline = existing.baseline_hr if existing.baseline_hr else ep.baseline_hr
        duration = max(existing.duration_min, ep.duration_min, (merged_end - merged_start).total_seconds() / 60.0)
        confidence = max(existing.confidence, ep.confidence)
        with self.tx() as c:
            c.execute(
                """UPDATE episodes SET start=?, end=?, duration_min=?, peak_hr=?, mean_hr=?, baseline_hr=?,
                   delta_hr=?, max_jump_bpm=?, asleep=?, steps_in_window=?, activity_level=?, stress_avg=?,
                   movement_fraction=?, confidence=?, kind=?, samples_json=?, updated_at=? WHERE id=?""",
                (
                    _iso(merged_start),
                    _iso(merged_end),
                    duration,
                    peak,
                    mean_hr,
                    baseline,
                    peak - baseline,
                    max(existing.max_jump_bpm, ep.max_jump_bpm),
                    int(existing.asleep or ep.asleep),
                    max(existing.steps_in_window, ep.steps_in_window),
                    existing.activity_level if existing.activity_level != "unknown" else ep.activity_level,
                    ep.stress_avg if ep.stress_avg is not None else existing.stress_avg,
                    max(existing.movement_fraction, ep.movement_fraction),
                    confidence,
                    existing.kind if existing.duration_min >= ep.duration_min else ep.kind,
                    json.dumps([{"ts": _iso(s.ts), "hr": s.hr} for s in samples]),
                    _iso(now),
                    existing.id,
                ),
            )
        stored = self.get_episode(existing.id)  # type: ignore[arg-type]
        assert stored is not None
        return stored, False

    def get_episode(self, episode_id: int) -> Episode | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,))
            row = cur.fetchone()
        return self._episode_from_row(row) if row else None

    def get_episodes(
        self,
        profile: str,
        start: datetime,
        end: datetime,
        min_confidence: float = 0.0,
        include_manual: bool = True,
    ) -> list[Episode]:
        q = "SELECT * FROM episodes WHERE profile=? AND start>=? AND start<? AND confidence>=?"
        params: list[Any] = [profile, _iso(start), _iso(end), min_confidence]
        if not include_manual:
            q += " AND source!='manual'"
        q += " ORDER BY start"
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [self._episode_from_row(r) for r in rows]

    def update_episode_assessment(
        self,
        episode_id: int,
        assessment: str,
        confidence: float | None,
        reasoning: str | None,
        doctor_note: str | None,
        model: str | None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE episodes SET llm_assessment=?, llm_confidence=?, llm_reasoning=?, doctor_note=?, "
                "llm_model=?, updated_at=? WHERE id=?",
                (assessment, confidence, reasoning, doctor_note, model, _iso(now_utc()), episode_id),
            )

    def mark_episode_notified(self, episode_id: int) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE episodes SET notified_at=?, updated_at=? WHERE id=?",
                (_iso(now_utc()), _iso(now_utc()), episode_id),
            )

    def set_episode_felt(self, episode_id: int, felt: bool | None, note: str | None = None) -> None:
        with self.tx() as c:
            if note:
                c.execute(
                    "UPDATE episodes SET felt=?, notes=COALESCE(notes || ' | ', '') || ?, updated_at=? WHERE id=?",
                    (None if felt is None else int(felt), note, _iso(now_utc()), episode_id),
                )
            else:
                c.execute(
                    "UPDATE episodes SET felt=?, updated_at=? WHERE id=?",
                    (None if felt is None else int(felt), _iso(now_utc()), episode_id),
                )

    def unassessed_episodes(self, profile: str, limit: int = 20) -> list[Episode]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE profile=? AND llm_assessment IS NULL AND source!='manual' "
                "ORDER BY start DESC LIMIT ?",
                (profile, limit),
            ).fetchall()
        return [self._episode_from_row(r) for r in rows]

    # -- symptoms --------------------------------------------------------

    def add_symptom(self, rep: SymptomReport) -> SymptomReport:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO symptoms(profile, reported_at, event_time, note, hr_at_time, baseline_hr, "
                "episode_id, chat_id, source, extracted) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rep.profile,
                    _iso(rep.reported_at),
                    _iso(rep.event_time),
                    rep.note,
                    rep.hr_at_time,
                    rep.baseline_hr,
                    rep.episode_id,
                    rep.chat_id,
                    rep.source,
                    json.dumps(rep.extracted) if rep.extracted else None,
                ),
            )
            rep.id = cur.lastrowid
        return rep

    def get_symptoms(self, profile: str, start: datetime, end: datetime) -> list[SymptomReport]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM symptoms WHERE profile=? AND event_time>=? AND event_time<? ORDER BY event_time",
                (profile, _iso(start), _iso(end)),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                SymptomReport(
                    id=r["id"],
                    profile=r["profile"],
                    reported_at=_parse_iso(r["reported_at"]) or now_utc(),
                    event_time=_parse_iso(r["event_time"]) or now_utc(),
                    note=r["note"] or "",
                    hr_at_time=r["hr_at_time"],
                    baseline_hr=r["baseline_hr"],
                    episode_id=r["episode_id"],
                    chat_id=r["chat_id"],
                    source=r["source"] or "command",
                    extracted=json.loads(r["extracted"]) if r["extracted"] else None,
                )
            )
        return out

    def symptoms_without_hr(self, profile: str, limit: int = 50) -> list[SymptomReport]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM symptoms WHERE profile=? AND hr_at_time IS NULL ORDER BY event_time DESC LIMIT ?",
                (profile, limit),
            ).fetchall()
        return [
            SymptomReport(
                id=r["id"],
                profile=r["profile"],
                reported_at=_parse_iso(r["reported_at"]) or now_utc(),
                event_time=_parse_iso(r["event_time"]) or now_utc(),
                note=r["note"] or "",
                hr_at_time=r["hr_at_time"],
                baseline_hr=r["baseline_hr"],
                episode_id=r["episode_id"],
                chat_id=r["chat_id"],
                source=r["source"] or "command",
            )
            for r in rows
        ]

    def update_symptom_hr(
        self, symptom_id: int, hr: int | None, baseline: float | None, episode_id: int | None
    ) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE symptoms SET hr_at_time=?, baseline_hr=?, episode_id=COALESCE(?, episode_id) WHERE id=?",
                (hr, baseline, episode_id, symptom_id),
            )

    # -- analyses --------------------------------------------------------

    def save_analysis(self, profile: str, kind: str, day: date | str, model: str | None, payload: dict[str, Any]) -> int:
        day_str = day if isinstance(day, str) else day.isoformat()
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO analyses(profile, kind, day, model, payload_json, created_at) VALUES (?,?,?,?,?,?)",
                (profile, kind, day_str, model, json.dumps(payload, default=_json_default), _iso(now_utc())),
            )
            return int(cur.lastrowid or 0)

    def get_latest_analysis(self, profile: str, kind: str, day: date | str | None = None) -> dict[str, Any] | None:
        q = "SELECT * FROM analyses WHERE profile=? AND kind=?"
        params: list[Any] = [profile, kind]
        if day is not None:
            q += " AND day=?"
            params.append(day if isinstance(day, str) else day.isoformat())
        q += " ORDER BY id DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(q, params).fetchone()
        if not row:
            return None
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json"))
        return d

    # -- notifications / kv --------------------------------------------

    def notification_sent(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM notifications WHERE key=?", (key,)).fetchone()
        return row is not None

    def mark_notification(self, key: str, profile: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO notifications(key, profile, sent_at) VALUES (?,?,?)",
                (key, profile, _iso(now_utc())),
            )

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str | None) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO kv(key, value) VALUES (?,?)", (key, value))

    # -- housekeeping ----------------------------------------------------

    def days_with_data(self, profile: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT day FROM daily_snapshots WHERE profile=? ORDER BY day", (profile,)
            ).fetchall()
        return [r["day"] for r in rows]

    def prune_raw(self, older_than_days: int) -> int:
        cutoff = (now_utc() - timedelta(days=older_than_days)).date().isoformat()
        with self.tx() as c:
            cur = c.execute("UPDATE daily_snapshots SET raw_json=NULL WHERE day<? AND raw_json IS NOT NULL", (cutoff,))
            return cur.rowcount
