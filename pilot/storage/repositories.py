"""One repository per table. Each works inside a single connection, so a
service can make several changes that are committed together."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pilot.core.telemetry import AnswerRecord, LearnerHistory
from pilot.storage.database import TABLES, now_ms

Row = dict[str, Any]


class _Repository:
    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con

    def _rows(self, sql: str, *args) -> list[Row]:
        return [dict(r) for r in self.con.execute(sql, args).fetchall()]

    def _one(self, sql: str, *args) -> Row | None:
        r = self.con.execute(sql, args).fetchone()
        return dict(r) if r else None

    def _scalar(self, sql: str, *args) -> Any:
        return self.con.execute(sql, args).fetchone()[0]


class SessionRepository(_Repository):
    def next_code(self) -> str:
        return f"P{self._scalar('SELECT COUNT(*) FROM sessions') + 1:02d}"

    def create(self, *, recorded_by: str, notes: str, mode: str) -> Row:
        code = self.next_code()
        self.con.execute(
            "INSERT INTO sessions (participant_code, guardian_consent, learner_assent, consent_recorded_by, "
            "notes, mode, created_ms) VALUES (?, 1, 1, ?, ?, ?, ?)",
            (code, recorded_by, notes, mode, now_ms()),
        )
        return self.by_code(code)

    def by_code(self, code: str) -> Row | None:
        return self._one("SELECT * FROM sessions WHERE participant_code = ?", code.upper())

    def by_id(self, session_id: int) -> Row | None:
        return self._one("SELECT * FROM sessions WHERE id = ?", session_id)

    def all(self, open_only: bool = False) -> list[Row]:
        where = "WHERE status IN ('ready', 'active')" if open_only else ""
        return self._rows(f"SELECT * FROM sessions {where} ORDER BY id")

    def activate(self, session_id: int, token: str, first_question_id: str | None) -> None:
        self.con.execute(
            "UPDATE sessions SET learner_token = ?, status = 'active', started_ms = COALESCE(started_ms, ?), "
            "current_question_id = COALESCE(current_question_id, ?) WHERE id = ?",
            (token, now_ms(), first_question_id, session_id),
        )

    def advance(self, session_id: int, next_index: int, next_question_id: str | None,
                support: dict | None) -> None:
        self.con.execute(
            "UPDATE sessions SET current_index = ?, current_question_id = ?, pending_support = ? WHERE id = ?",
            (next_index, next_question_id, json.dumps(support) if support else None, session_id),
        )

    def finish(self, session_id: int) -> None:
        self.con.execute(
            "UPDATE sessions SET status = 'finished', finished_ms = ?, learner_token = NULL WHERE id = ?",
            (now_ms(), session_id))

    def close(self, session_id: int) -> None:
        self.con.execute("UPDATE sessions SET status = 'closed', learner_token = NULL WHERE id = ?", (session_id,))


class AnswerRepository(_Repository):
    def add(self, *, session_id: int, question_index: int, question_id: str, kind: str, difficulty: int,
            given: str, correct: bool, response_time_ms: int, first_input_ms: int | None, hints_used: int,
            shown_client_ms: int | None, answered_client_ms: int | None) -> None:
        self.con.execute(
            "INSERT INTO answers (session_id, question_index, question_id, kind, difficulty, given, correct, "
            "response_time_ms, first_input_ms, hints_used, shown_client_ms, answered_client_ms, server_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, question_index, question_id, kind, difficulty, given, int(correct), response_time_ms,
             first_input_ms, hints_used, shown_client_ms, answered_client_ms, now_ms()),
        )

    def for_session(self, session_id: int) -> list[Row]:
        return self._rows("SELECT * FROM answers WHERE session_id = ? ORDER BY question_index", session_id)

    def history(self, session_id: int) -> LearnerHistory:
        return LearnerHistory(tuple(
            AnswerRecord(
                question_index=r["question_index"], difficulty=r["difficulty"], correct=bool(r["correct"]),
                response_time_s=r["response_time_ms"] / 1000,
                first_input_s=None if r["first_input_ms"] is None else r["first_input_ms"] / 1000,
                hints_used=r["hints_used"],
            ) for r in self.for_session(session_id)))

    def seen_question_ids(self, session_id: int) -> set[str]:
        return {r["question_id"] for r in self.for_session(session_id) if r["question_id"]}

    def counts(self, session_id: int) -> tuple[int, int]:
        r = self._one("SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS c FROM answers WHERE session_id = ?",
                      session_id)
        return r["n"], r["c"]


class EventRepository(_Repository):
    def log(self, session_id: int, type_: str, question_index: int | None, client_ms: int | None,
            payload: dict | None = None) -> None:
        self.con.execute(
            "INSERT INTO events (session_id, type, question_index, client_ms, server_ms, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, type_, question_index, client_ms, now_ms(), json.dumps(payload or {})),
        )

    def count(self, session_id: int, question_index: int, type_: str) -> int:
        return self._scalar("SELECT COUNT(*) FROM events WHERE session_id = ? AND question_index = ? AND type = ?",
                            session_id, question_index, type_)

    def for_session(self, session_id: int) -> list[Row]:
        return self._rows("SELECT * FROM events WHERE session_id = ? ORDER BY id", session_id)


class CheckinRepository(_Repository):
    def add(self, session_id: int, after_question: int, choice: str, client_ms: int | None) -> None:
        self.con.execute(
            "INSERT INTO checkins (session_id, after_question, choice, client_ms, server_ms) VALUES (?, ?, ?, ?, ?)",
            (session_id, after_question, choice, client_ms, now_ms()))

    def for_session(self, session_id: int) -> list[Row]:
        return self._rows("SELECT * FROM checkins WHERE session_id = ? ORDER BY id", session_id)


class ObservationRepository(_Repository):
    def add(self, session_id: int, observer: str, code: str, client_ms: int | None) -> None:
        self.con.execute(
            "INSERT INTO observations (session_id, observer, code, client_ms, server_ms) VALUES (?, ?, ?, ?, ?)",
            (session_id, observer, code, client_ms, now_ms()))

    def for_session(self, session_id: int) -> list[Row]:
        return self._rows("SELECT * FROM observations WHERE session_id = ? ORDER BY id", session_id)

    def count(self, session_id: int) -> int:
        return self._scalar("SELECT COUNT(*) FROM observations WHERE session_id = ?", session_id)


class DecisionRepository(_Repository):
    def add(self, *, session_id: int, question_index: int, state: str, state_source: str,
            observation: list[float], decision) -> None:
        self.con.execute(
            "INSERT INTO decisions (session_id, question_index, state, state_source, observation, model, "
            "raw_action_index, raw_action_name, action_index, action_name, guardrail, q_values, server_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, question_index, state, state_source, json.dumps(observation), decision.model,
             decision.raw_action_index, decision.raw_action, decision.action_index, decision.action,
             decision.guardrail, json.dumps(list(decision.scores)), now_ms()),
        )

    def for_session(self, session_id: int) -> list[Row]:
        return self._rows("SELECT * FROM decisions WHERE session_id = ? ORDER BY id", session_id)

    def latest(self, session_id: int) -> Row | None:
        return self._one(
            "SELECT question_index, state, raw_action_name, action_name, guardrail FROM decisions "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 1", session_id)


class Repositories:
    """All repositories bound to one connection (one unit of work)."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con
        self.sessions = SessionRepository(con)
        self.answers = AnswerRepository(con)
        self.events = EventRepository(con)
        self.checkins = CheckinRepository(con)
        self.observations = ObservationRepository(con)
        self.decisions = DecisionRepository(con)

    def export(self, table: str) -> list[Row]:
        if table not in TABLES:
            raise KeyError(table)
        rows = [dict(r) for r in self.con.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()]
        for r in rows:
            r.pop("learner_token", None)
        return rows
