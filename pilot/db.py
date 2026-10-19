"""SQLite storage for the pilot. One file, no server to run.

Nothing here holds a name or any other identifier: learners are known only
by a participant code (P01, P02, ...). The mapping from codes to children is
kept on paper by the researcher, outside this system.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.environ.get("PILOT_DB", Path(__file__).parent / "data" / "pilot.sqlite3"))

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
    current_index INTEGER NOT NULL DEFAULT 0,
    created_ms INTEGER NOT NULL,
    started_ms INTEGER,
    finished_ms INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    type TEXT NOT NULL,
    question_index INTEGER,
    client_ms INTEGER,                         -- learner device clock, corrected to server time
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
    client_ms INTEGER,                         -- observer device clock, corrected to server time
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
    raw_action_index INTEGER NOT NULL,         -- the DQN's own first choice
    raw_action_name TEXT NOT NULL,
    action_index INTEGER NOT NULL,             -- after guardrails
    action_name TEXT NOT NULL,
    guardrail TEXT,                            -- why the DQN's choice was ruled out, if it was
    q_values TEXT NOT NULL,
    server_ms INTEGER NOT NULL
);
"""

TABLES = ("sessions", "events", "answers", "checkins", "observations", "decisions")


def now_ms() -> int:
    return int(time.time() * 1000)


def init() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.executescript(SCHEMA)


@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def rows(con, sql: str, *args) -> list[dict]:
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def one(con, sql: str, *args) -> dict | None:
    r = con.execute(sql, args).fetchone()
    return dict(r) if r else None


def log_event(con, session_id: int, type_: str, question_index: int | None,
              client_ms: int | None, payload: dict | None = None) -> None:
    con.execute(
        "INSERT INTO events (session_id, type, question_index, client_ms, server_ms, payload) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, type_, question_index, client_ms, now_ms(), json.dumps(payload or {})),
    )
