"""SensAI pilot server: learner, observer and researcher APIs, plus the web app.

Run from the repository root:

    uv run uvicorn pilot.server:app --port 8001

Learner endpoints are reached with a participant code and the token issued
when the learner joins. Observer and researcher endpoints need the
researcher key (`PILOT_RESEARCHER_KEY`, sent as the `X-Researcher-Key`
header).
"""

from __future__ import annotations

import csv
import io
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pilot import db
from pilot.labeler import RECENT_WINDOW, STATE_LABELS, AnswerTelemetry, build_observation, provisional_state
from pilot.policy import shadow_decision
from pilot.tasks import CHECKIN_AFTER, FORM, N_QUESTIONS

RESEARCHER_KEY = os.environ.get("PILOT_RESEARCHER_KEY", "sensai-dev")
WEB_DIST = Path(__file__).parent / "web" / "dist"

LEARNER_EVENT_TYPES = {
    "question_shown", "first_input", "input_changed", "idle",
    "focus_lost", "focus_returned", "session_resumed",
}
CHECKIN_CHOICES = {"easy", "good", "hard", "lost"}
OBSERVER_CODES = {"on_task", "rushing", "struggling", "stuck", "off_task", "cant_see"}


@asynccontextmanager
async def lifespan(_app):
    db.init()
    yield


app = FastAPI(title="SensAI Pilot", lifespan=lifespan)


# --------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------

def require_researcher(x_researcher_key: str = Header(default="")) -> None:
    if not secrets.compare_digest(x_researcher_key, RESEARCHER_KEY):
        raise HTTPException(401, "Researcher key required.")


def learner_session(con, code: str, token: str) -> dict:
    s = db.one(con, "SELECT * FROM sessions WHERE participant_code = ?", code.upper())
    if not s or not s["learner_token"] or not secrets.compare_digest(s["learner_token"], token):
        raise HTTPException(403, "This session is not open on this device. Ask the researcher.")
    if s["status"] not in ("ready", "active"):
        raise HTTPException(409, "This session has ended.")
    return s


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------

@app.get("/api/time")
def server_time() -> dict:
    """Devices call this to work out their clock offset, so learner events
    and observer codes recorded on different devices line up in time."""
    return {"server_ms": db.now_ms()}


@app.get("/api/meta")
def meta() -> dict:
    return {
        "n_questions": N_QUESTIONS,
        "checkin_after": sorted(CHECKIN_AFTER),
        "states": STATE_LABELS,
        "observer_codes": sorted(OBSERVER_CODES),
    }


# --------------------------------------------------------------------------
# Researcher
# --------------------------------------------------------------------------

class NewSession(BaseModel):
    guardian_consent: bool
    learner_assent: bool
    recorded_by: str = Field(min_length=1, max_length=60)
    notes: str = Field(default="", max_length=500)


@app.post("/api/researcher/sessions", dependencies=[Depends(require_researcher)])
def create_session(body: NewSession) -> dict:
    if not (body.guardian_consent and body.learner_assent):
        raise HTTPException(400, "A session needs both signed guardian consent and the learner's assent.")
    with db.connect() as con:
        n = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        code = f"P{n + 1:02d}"
        con.execute(
            "INSERT INTO sessions (participant_code, guardian_consent, learner_assent, consent_recorded_by, notes, created_ms) "
            "VALUES (?, 1, 1, ?, ?, ?)",
            (code, body.recorded_by.strip(), body.notes.strip(), db.now_ms()),
        )
        return db.one(con, "SELECT * FROM sessions WHERE participant_code = ?", code)


def _session_summaries(con, where: str = "") -> list[dict]:
    sessions = db.rows(con, f"SELECT * FROM sessions {where} ORDER BY id")
    for s in sessions:
        sid = s["id"]
        s.pop("learner_token", None)
        s["answered"] = con.execute("SELECT COUNT(*) FROM answers WHERE session_id = ?", (sid,)).fetchone()[0]
        s["correct"] = con.execute("SELECT COUNT(*) FROM answers WHERE session_id = ? AND correct = 1", (sid,)).fetchone()[0]
        s["observations"] = con.execute("SELECT COUNT(*) FROM observations WHERE session_id = ?", (sid,)).fetchone()[0]
        s["latest_decision"] = db.one(
            con, "SELECT question_index, state, raw_action_name, action_name, guardrail FROM decisions "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 1", sid)
    return sessions


@app.get("/api/researcher/sessions", dependencies=[Depends(require_researcher)])
def list_sessions() -> list[dict]:
    with db.connect() as con:
        return _session_summaries(con)


@app.get("/api/researcher/sessions/{session_id}", dependencies=[Depends(require_researcher)])
def session_detail(session_id: int) -> dict:
    with db.connect() as con:
        s = db.one(con, "SELECT * FROM sessions WHERE id = ?", session_id)
        if not s:
            raise HTTPException(404, "No such session.")
        s.pop("learner_token", None)
        return {
            "session": s,
            "answers": db.rows(con, "SELECT * FROM answers WHERE session_id = ? ORDER BY question_index", session_id),
            "checkins": db.rows(con, "SELECT * FROM checkins WHERE session_id = ? ORDER BY id", session_id),
            "observations": db.rows(con, "SELECT * FROM observations WHERE session_id = ? ORDER BY id", session_id),
            "decisions": db.rows(con, "SELECT * FROM decisions WHERE session_id = ? ORDER BY id", session_id),
            "events": db.rows(con, "SELECT * FROM events WHERE session_id = ? ORDER BY id", session_id),
        }


@app.post("/api/researcher/sessions/{session_id}/close", dependencies=[Depends(require_researcher)])
def close_session(session_id: int) -> dict:
    with db.connect() as con:
        con.execute("UPDATE sessions SET status = 'closed', learner_token = NULL WHERE id = ?", (session_id,))
        return {"ok": True}


@app.get("/api/researcher/export/{table}.csv", dependencies=[Depends(require_researcher)])
def export_table(table: str):
    if table not in db.TABLES:
        raise HTTPException(404, f"Unknown table. Choose one of {list(db.TABLES)}.")
    with db.connect() as con:
        data = db.rows(con, f"SELECT * FROM {table} ORDER BY id")
    for r in data:
        r.pop("learner_token", None)
    buf = io.StringIO()
    if data:
        w = csv.DictWriter(buf, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="sensai_pilot_{table}.csv"'},
    )


# --------------------------------------------------------------------------
# Observer
# --------------------------------------------------------------------------

@app.get("/api/observer/sessions", dependencies=[Depends(require_researcher)])
def observer_sessions() -> list[dict]:
    with db.connect() as con:
        return _session_summaries(con, "WHERE status IN ('ready', 'active')")


class Observation(BaseModel):
    session_id: int
    observer: str = Field(min_length=1, max_length=40)
    code: str
    client_ms: int | None = None


@app.post("/api/observer/observations", dependencies=[Depends(require_researcher)])
def record_observation(body: Observation) -> dict:
    if body.code not in OBSERVER_CODES:
        raise HTTPException(400, f"Unknown code {body.code!r}.")
    with db.connect() as con:
        if not db.one(con, "SELECT id FROM sessions WHERE id = ?", body.session_id):
            raise HTTPException(404, "No such session.")
        con.execute(
            "INSERT INTO observations (session_id, observer, code, client_ms, server_ms) VALUES (?, ?, ?, ?, ?)",
            (body.session_id, body.observer.strip(), body.code, body.client_ms, db.now_ms()),
        )
        return {"ok": True}


# --------------------------------------------------------------------------
# Learner
# --------------------------------------------------------------------------

class Join(BaseModel):
    code: str = Field(min_length=2, max_length=10)


@app.post("/api/learner/join")
def join(body: Join) -> dict:
    code = body.code.strip().upper()
    with db.connect() as con:
        s = db.one(con, "SELECT * FROM sessions WHERE participant_code = ?", code)
        if not s:
            raise HTTPException(404, "That code doesn't match a session. Check it with the researcher.")
        if s["status"] not in ("ready", "active"):
            raise HTTPException(409, "This session has already finished.")
        token = secrets.token_urlsafe(24)
        resumed = s["status"] == "active"
        con.execute(
            "UPDATE sessions SET learner_token = ?, status = 'active', started_ms = COALESCE(started_ms, ?) WHERE id = ?",
            (token, db.now_ms(), s["id"]),
        )
        if resumed:
            db.log_event(con, s["id"], "session_resumed", s["current_index"], None)
        return {
            "participant_code": code,
            "token": token,
            "current_index": s["current_index"],
            "n_questions": N_QUESTIONS,
            "resumed": resumed,
        }


class LearnerAuth(BaseModel):
    code: str
    token: str


@app.post("/api/learner/question")
def current_question(body: LearnerAuth) -> dict:
    with db.connect() as con:
        s = learner_session(con, body.code, body.token)
        if s["current_index"] >= N_QUESTIONS:
            return {"done": True}
        return {"done": False, "question": FORM[s["current_index"]].public()}


class LearnerEvent(LearnerAuth):
    type: str
    question_index: int | None = None
    client_ms: int | None = None
    payload: dict = Field(default_factory=dict)


@app.post("/api/learner/event")
def learner_event(body: LearnerEvent) -> dict:
    if body.type not in LEARNER_EVENT_TYPES:
        raise HTTPException(400, f"Unknown event type {body.type!r}.")
    with db.connect() as con:
        s = learner_session(con, body.code, body.token)
        db.log_event(con, s["id"], body.type, body.question_index, body.client_ms, body.payload)
        return {"ok": True}


class HintRequest(LearnerAuth):
    question_index: int
    client_ms: int | None = None


@app.post("/api/learner/hint")
def hint(body: HintRequest) -> dict:
    with db.connect() as con:
        s = learner_session(con, body.code, body.token)
        if body.question_index != s["current_index"]:
            raise HTTPException(409, "That question is not the current one.")
        db.log_event(con, s["id"], "hint_requested", body.question_index, body.client_ms)
        return {"hint": FORM[body.question_index].hint}


class AnswerIn(LearnerAuth):
    question_index: int
    answer: str = Field(max_length=40)
    response_time_ms: int = Field(ge=0)
    first_input_ms: int | None = Field(default=None, ge=0)
    shown_client_ms: int | None = None
    answered_client_ms: int | None = None


@app.post("/api/learner/answer")
def answer(body: AnswerIn) -> dict:
    with db.connect() as con:
        s = learner_session(con, body.code, body.token)
        if body.question_index != s["current_index"]:
            raise HTTPException(409, "That question is not the current one.")
        q = FORM[body.question_index]
        correct = q.is_correct(body.answer)
        hints_used = con.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ? AND question_index = ? AND type = 'hint_requested'",
            (s["id"], q.index),
        ).fetchone()[0]

        con.execute(
            "INSERT INTO answers (session_id, question_index, kind, difficulty, given, correct, response_time_ms, "
            "first_input_ms, hints_used, shown_client_ms, answered_client_ms, server_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (s["id"], q.index, q.kind, q.difficulty, body.answer.strip(), int(correct), body.response_time_ms,
             body.first_input_ms, hints_used, body.shown_client_ms, body.answered_client_ms, db.now_ms()),
        )

        # Shadow mode: work out what the DQN would do, store it, change nothing.
        recent = db.rows(
            con, "SELECT correct FROM answers WHERE session_id = ? ORDER BY question_index DESC LIMIT ?",
            s["id"], RECENT_WINDOW,
        )
        at_level = con.execute(
            "SELECT COUNT(*) FROM answers WHERE session_id = ? AND difficulty = ?", (s["id"], q.difficulty)
        ).fetchone()[0]
        telemetry = AnswerTelemetry(
            response_time_s=body.response_time_ms / 1000,
            first_input_s=None if body.first_input_ms is None else body.first_input_ms / 1000,
            correct=correct,
            hints_used=hints_used,
            difficulty=q.difficulty,
            recent_errors=[not r["correct"] for r in reversed(recent)],
            questions_at_level=at_level,
        )
        state = provisional_state(telemetry)
        obs = build_observation(state, telemetry)
        decision = shadow_decision(obs, state, q.difficulty)
        con.execute(
            "INSERT INTO decisions (session_id, question_index, state, state_source, observation, model, "
            "raw_action_index, raw_action_name, action_index, action_name, guardrail, q_values, server_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (s["id"], q.index, state, "provisional_rules_v1", json.dumps([round(float(v), 4) for v in obs]),
             decision["model"], decision["raw_action_index"], decision["raw_action_name"],
             decision["action_index"], decision["action_name"], decision["guardrail"],
             json.dumps(decision["q_values"]), db.now_ms()),
        )

        next_index = q.index + 1
        finished = next_index >= N_QUESTIONS
        con.execute(
            "UPDATE sessions SET current_index = ? WHERE id = ?", (next_index, s["id"]))
        return {
            "correct": correct,
            "correct_answer": q.answer,
            "checkin_due": q.index in CHECKIN_AFTER,
            "finished": finished,
            "next_index": next_index,
        }


class CheckinIn(LearnerAuth):
    after_question: int
    choice: str
    client_ms: int | None = None


@app.post("/api/learner/checkin")
def checkin(body: CheckinIn) -> dict:
    if body.choice not in CHECKIN_CHOICES:
        raise HTTPException(400, f"Unknown choice {body.choice!r}.")
    with db.connect() as con:
        s = learner_session(con, body.code, body.token)
        con.execute(
            "INSERT INTO checkins (session_id, after_question, choice, client_ms, server_ms) VALUES (?, ?, ?, ?, ?)",
            (s["id"], body.after_question, body.choice, body.client_ms, db.now_ms()),
        )
        return {"ok": True}


@app.post("/api/learner/finish")
def finish(body: LearnerAuth) -> dict:
    with db.connect() as con:
        s = learner_session(con, body.code, body.token)
        if s["current_index"] < N_QUESTIONS:
            raise HTTPException(409, "There are still questions left.")
        con.execute(
            "UPDATE sessions SET status = 'finished', finished_ms = ?, learner_token = NULL WHERE id = ?",
            (db.now_ms(), s["id"]),
        )
        return {"ok": True}


# --------------------------------------------------------------------------
# Web app (built React files), with client-side routing
# --------------------------------------------------------------------------

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = (WEB_DIST / path).resolve()
        if path and target.is_relative_to(WEB_DIST.resolve()) and target.is_file():
            return FileResponse(target)
        return FileResponse(WEB_DIST / "index.html")
