from __future__ import annotations

import fcntl
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    participant_code TEXT NOT NULL UNIQUE,
    guardian_consent INTEGER NOT NULL,
    learner_assent INTEGER NOT NULL,
    consent_recorded_by TEXT NOT NULL,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'ready',      -- ready | active | finished | closed
    learner_token TEXT,
    current_index INTEGER NOT NULL DEFAULT 0,  -- position in the session (0-14)
    created_ms INTEGER NOT NULL,
    started_ms INTEGER,
    finished_ms INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    type TEXT NOT NULL,
    question_index INTEGER,
    client_ms INTEGER,
    server_ms INTEGER NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    question_index INTEGER NOT NULL,
    kind TEXT NOT NULL,
    difficulty INTEGER NOT NULL,
    given TEXT NOT NULL,
    correct INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    first_input_ms INTEGER,
    hints_used INTEGER NOT NULL,
    shown_client_ms INTEGER,
    answered_client_ms INTEGER,
    server_ms INTEGER NOT NULL,
    UNIQUE (session_id, question_index)
);
CREATE TABLE IF NOT EXISTS checkins (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    after_question INTEGER NOT NULL,
    choice TEXT NOT NULL,
    client_ms INTEGER,
    server_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    observer TEXT NOT NULL,
    code TEXT NOT NULL,
    client_ms INTEGER,
    server_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    question_index INTEGER NOT NULL,
    state TEXT NOT NULL,
    state_source TEXT NOT NULL,
    observation TEXT NOT NULL,
    model TEXT NOT NULL,
    raw_action_index INTEGER NOT NULL,
    raw_action_name TEXT NOT NULL,
    action_index INTEGER NOT NULL,
    action_name TEXT NOT NULL,
    guardrail TEXT,
    q_values TEXT NOT NULL,
    server_ms INTEGER NOT NULL
);
"""

# Columns added after the pilot. Added in place so the pilot database (21 October
# 2026) keeps working unchanged: old rows get the defaults shown.
MIGRATIONS = (
    ("sessions", "mode", "TEXT NOT NULL DEFAULT 'shadow'"),          # shadow | adaptive
    ("sessions", "current_question_id", "TEXT"),
    ("sessions", "pending_support", "TEXT"),                         # JSON shown with the next question
    ("answers", "question_id", "TEXT"),
)

TABLES = ("sessions", "events", "answers", "checkins", "observations", "decisions")


def now_ms() -> int:
    return int(time.time() * 1000)


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def initialise(self) -> None:
        """Create or upgrade the database. Safe to call from several worker
        processes at once: a file lock makes them take turns (without it the
        stress test hit "database is locked" when four workers started together)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".init.lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._create_or_upgrade()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _create_or_upgrade(self) -> None:
        with self.connect() as con:
            # Write-ahead logging lets many learners read while one answer is
            # written, instead of every request waiting for a single lock.
            # Found necessary by the stress test (docs/testing).
            con.execute("PRAGMA journal_mode = WAL")
            con.executescript(SCHEMA)
            for table, column, spec in MIGRATIONS:
                existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    try:
                        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")
                    except sqlite3.OperationalError as e:
                        # Several worker processes may upgrade at the same moment.
                        if "duplicate column" not in str(e):
                            raise

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA synchronous = NORMAL")   # safe with WAL; avoids a disk sync per request
        try:
            yield con
            con.commit()
        finally:
            con.close()
