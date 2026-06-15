"""Data access — SQLite behind a repository, the single source of truth.

Same discipline as the rest of the system: methods never raise into the request path
(they degrade to None/[]/False and the caller handles it), every write that matters
is wrapped with its audit event in ONE transaction, and connections are always
closed (the `_tx` context manager) with WAL + busy_timeout for safe concurrency.

The headline correctness claim — no double-booking — lives in `book_slot`: the SLOT
is the unit of concurrency (UNIQUE(doctor_id, start_at)), and booking is a single
conditional `UPDATE ... WHERE status='available'` that exactly one writer can win.

Authorization is NOT enforced here — it's enforced at the tool layer via
auth.can_access (the single choke point), so this stays pure data access. `get_repository`
is the swap point: a Postgres backend implements the same methods and plugs in here.
"""

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import config
import migrations


def _now() -> str:
    # INVARIANT: every timestamp is UTC ISO-8601 to seconds, produced here, so
    # lexicographic compares == chronological compares (used for slot ordering).
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteRepository:
    def __init__(self, db_path: str):
        self._path = db_path
        with self._tx() as conn:
            migrations.migrate(conn)

    def _connect(self):
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _tx(self):
        conn = self._connect()
        try:
            with conn:           # commit on success, rollback on error
                yield conn
        finally:
            conn.close()

    def _audit(self, conn, entity_type, entity_id, event_type, actor, detail=None):
        conn.execute(
            "INSERT INTO audit_events (entity_type, entity_id, event_type, actor, detail, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (entity_type, entity_id, event_type, actor, detail, _now()),
        )

    # -- seeding directory ----------------------------------------------------
    def add_patient(self, patient_id, name, dob=None):
        with self._tx() as conn:
            conn.execute("INSERT OR REPLACE INTO patients (patient_id, name, dob, created_at)"
                         " VALUES (?, ?, ?, ?)", (patient_id, name, dob, _now()))

    def add_doctor(self, doctor_id, name, specialty):
        with self._tx() as conn:
            conn.execute("INSERT OR REPLACE INTO doctors (doctor_id, name, specialty)"
                         " VALUES (?, ?, ?)", (doctor_id, name, specialty))

    def add_slot(self, doctor_id, start_at, slot_id=None):
        sid = slot_id or uuid.uuid4().hex[:10]
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO slots (slot_id, doctor_id, start_at, status, updated_at)"
                    " VALUES (?, ?, ?, 'available', ?)", (sid, doctor_id, start_at, _now()))
                if cur.rowcount == 1:
                    return sid
                # UNIQUE(doctor_id, start_at) conflict: return the EXISTING slot's id,
                # not the phantom one we just tried to insert.
                row = conn.execute(
                    "SELECT slot_id FROM slots WHERE doctor_id=? AND start_at=?",
                    (doctor_id, start_at)).fetchone()
                return row["slot_id"] if row else None
        except Exception:
            return None

    # -- reads ----------------------------------------------------------------
    def get_patient(self, patient_id):
        return self._one("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))

    def list_doctors(self, specialty=None):
        if specialty:
            return self._all("SELECT * FROM doctors WHERE lower(specialty) = lower(?) ORDER BY name",
                             (specialty,))
        return self._all("SELECT * FROM doctors ORDER BY specialty, name", ())

    def find_available_slots(self, specialty, limit=5):
        return self._all(
            "SELECT s.*, d.name AS doctor_name, d.specialty FROM slots s "
            "JOIN doctors d ON d.doctor_id = s.doctor_id "
            "WHERE s.status='available' AND lower(d.specialty)=lower(?) "
            "ORDER BY s.start_at ASC LIMIT ?", (specialty, limit))

    def list_appointments(self, patient_id=None, doctor_id=None):
        if patient_id is not None:
            return self._all("SELECT * FROM appointments WHERE patient_id = ? ORDER BY start_at",
                             (patient_id,))
        if doctor_id is not None:
            return self._all("SELECT * FROM appointments WHERE doctor_id = ? ORDER BY start_at",
                             (doctor_id,))
        return self._all("SELECT * FROM appointments ORDER BY start_at", ())

    def list_records(self, patient_id):
        # Append-only with corrections: a row that a newer row supersedes is excluded,
        # so reads always reflect the current chart (the raw history stays in the table).
        return self._all(
            "SELECT * FROM records WHERE patient_id = ? AND record_id NOT IN "
            "(SELECT supersedes FROM records WHERE patient_id = ? AND supersedes IS NOT NULL) "
            "ORDER BY record_id ASC", (patient_id, patient_id))

    def list_memory(self, patient_id, limit=20):
        return self._all("SELECT * FROM patient_memory WHERE patient_id = ? "
                         "ORDER BY memory_id DESC LIMIT ?", (patient_id, limit))

    def get_audit(self, entity_type, entity_id):
        return self._all("SELECT * FROM audit_events WHERE entity_type=? AND entity_id=? "
                         "ORDER BY event_id ASC", (entity_type, entity_id))

    # -- writes (with audit, transactional) -----------------------------------
    def book_slot(self, slot_id, patient_id, specialty, actor):
        """Atomically claim an available slot. Returns the appointment dict, or None
        if the slot was already taken (the loser of a race) / not found."""
        try:
            with self._tx() as conn:
                # Conditional claim: exactly one concurrent writer flips it from
                # 'available'. rowcount==0 => someone else won (or it's gone).
                cur = conn.execute(
                    "UPDATE slots SET status='booked', patient_id=?, updated_at=? "
                    "WHERE slot_id=? AND status='available'", (patient_id, _now(), slot_id))
                if cur.rowcount != 1:
                    return None
                slot = conn.execute("SELECT * FROM slots WHERE slot_id=?", (slot_id,)).fetchone()
                appt_id = uuid.uuid4().hex[:10]
                conn.execute(
                    "INSERT INTO appointments (appointment_id, slot_id, patient_id, doctor_id, "
                    "specialty, start_at, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'booked', ?)",
                    (appt_id, slot_id, patient_id, slot["doctor_id"], specialty, slot["start_at"], _now()))
                self._audit(conn, "appointment", appt_id, "booked", actor,
                            f"{specialty} @ {slot['start_at']} (slot {slot_id})")
                return {"appointment_id": appt_id, "doctor_id": slot["doctor_id"],
                        "specialty": specialty, "start_at": slot["start_at"], "status": "booked"}
        except Exception:
            return None

    def add_record(self, patient_id, record_type, label, value=None, note=None,
                   recorded_by="system", supersedes=None):
        """Append a clinical record (append-only) + audit, in one transaction."""
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    "INSERT INTO records (patient_id, record_type, label, value, note, "
                    "recorded_at, recorded_by, supersedes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (patient_id, record_type, label, value, note, _now(), recorded_by, supersedes))
                rid = cur.lastrowid
                self._audit(conn, "record", str(rid), "added", recorded_by,
                            f"{record_type}: {label}")
                return rid
        except Exception:
            return None

    def add_memory(self, patient_id, content):
        try:
            with self._tx() as conn:
                conn.execute("INSERT INTO patient_memory (patient_id, content, created_at)"
                             " VALUES (?, ?, ?)", (patient_id, content, _now()))
            return True
        except Exception:
            return False

    # -- helpers --------------------------------------------------------------
    def _one(self, sql, args):
        try:
            with self._tx() as conn:
                r = conn.execute(sql, args).fetchone()
                return dict(r) if r else None
        except Exception:
            return None

    def _all(self, sql, args):
        try:
            with self._tx() as conn:
                return [dict(r) for r in conn.execute(sql, args).fetchall()]
        except Exception:
            return []

    def reset(self):
        if os.path.exists(self._path):
            os.remove(self._path)
        with self._tx() as conn:
            migrations.migrate(conn)


def build_repository() -> SQLiteRepository:
    """Factory + Postgres swap point. A postgres:// DATABASE_URL would select a
    PostgresRepository implementing the same methods (the SQL is largely portable)."""
    if config.DATABASE_URL.startswith(("postgres://", "postgresql://")):
        raise NotImplementedError("Implement PostgresRepository against psycopg here.")
    return SQLiteRepository(config.DB_PATH)


_repo: SQLiteRepository | None = None


def get_repository() -> SQLiteRepository:
    # The singleton is just a connection-string holder: each call opens its own
    # connection and migrations are idempotent, so a lazy-init race under Streamlit's
    # per-session threads is harmless. The no-double-booking correctness guarantee does
    # NOT depend on this — it lives in the atomic conditional UPDATE in book_slot.
    global _repo
    if _repo is None:
        _repo = build_repository()
    return _repo


def reset_repository_singleton() -> None:
    global _repo
    _repo = None
