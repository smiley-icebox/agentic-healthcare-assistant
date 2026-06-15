"""Schema migrations — versioned, forward-only, crash-safe.

Records which schema version a database is at and applies only the steps it hasn't
seen. CRASH-SAFE: SQLite auto-commits DDL, so a crash between an ALTER and the
version bump is possible — re-running swallows "duplicate column"/"already exists"
instead of bricking every future boot. SQL is kept portable for a future Postgres
backend behind the repository seam.

Data model (the audit trail + append-only records are the production-minded parts):
  - patients, doctors            — directory
  - slots                        — UNIQUE(doctor_id, start_at); the unit of booking
  - appointments                 — a booking event referencing a slot
  - records                      — APPEND-ONLY clinical facts: typed columns + note;
                                   corrections are new rows referencing the prior id
  - audit_events                 — immutable event per write (actor + timestamp)
  - patient_memory               — long-term per-patient context (redacted)
"""

import sqlite3

MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id   TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                dob          TEXT,
                created_at   TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS doctors (
                doctor_id    TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                specialty    TEXT NOT NULL
            )
            """,
            # The SLOT is the unit of concurrency. UNIQUE(doctor_id, start_at) makes
            # double-booking structurally impossible; booking is a conditional UPDATE.
            """
            CREATE TABLE IF NOT EXISTS slots (
                slot_id      TEXT PRIMARY KEY,
                doctor_id    TEXT NOT NULL,
                start_at     TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'available',
                patient_id   TEXT,
                updated_at   TEXT NOT NULL,
                UNIQUE (doctor_id, start_at)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_slots_doc ON slots (doctor_id, status)",
            """
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id TEXT PRIMARY KEY,
                slot_id        TEXT NOT NULL,
                patient_id     TEXT NOT NULL,
                doctor_id      TEXT NOT NULL,
                specialty      TEXT NOT NULL,
                start_at       TEXT NOT NULL,
                status         TEXT NOT NULL,
                created_at     TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_appt_patient ON appointments (patient_id)",
            # APPEND-ONLY clinical records. Typed columns make "summarize diagnoses/
            # alerts" a structured READ (the LLM only phrases), never a parse-from-prose.
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   TEXT NOT NULL,
                record_type  TEXT NOT NULL,   -- diagnosis|treatment|medication|allergy|alert|note
                label        TEXT NOT NULL,
                value        TEXT,
                note         TEXT,            -- free-text / unstructured
                recorded_at  TEXT NOT NULL,
                recorded_by  TEXT NOT NULL,
                supersedes   INTEGER          -- record_id this corrects (append-only)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_records_patient ON records (patient_id)",
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type  TEXT NOT NULL,   -- appointment|record
                entity_id    TEXT NOT NULL,
                event_type   TEXT NOT NULL,
                actor        TEXT NOT NULL,
                detail       TEXT,
                created_at   TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_events (entity_type, entity_id)",
            """
            CREATE TABLE IF NOT EXISTS patient_memory (
                memory_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   TEXT NOT NULL,
                content      TEXT NOT NULL,   -- redacted long-term context
                created_at   TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_patient ON patient_memory (patient_id)",
        ],
    ),
]

CURRENT_VERSION = MIGRATIONS[-1][0]


def _current_version(conn) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    v = row["v"] if hasattr(row, "keys") else row[0]
    return v or 0


def migrate(conn) -> int:
    """Apply pending migrations idempotently. Returns the new version."""
    current = _current_version(conn)
    for version, statements in MIGRATIONS:
        if version <= current:
            continue
        for sql in statements:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as exc:
                m = str(exc).lower()
                if "duplicate column" in m or "already exists" in m:
                    continue
                raise
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        current = version
    conn.commit()
    return current
