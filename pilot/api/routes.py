from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from pilot.api.schemas import (
    AnswerIn, CheckinIn, HintRequest, Join, LearnerAuth, LearnerEvent, NewSession, Observation,
)
from pilot.api.security import KeyGuard
from pilot.core.states import OBSERVER_CODES, BehaviouralState
from pilot.core.tasks import CHECKIN_AFTER, SESSION_LENGTH
from pilot.services.learning import Answer, LearningService
from pilot.services.observation import ObservationService
from pilot.services.sessions import SessionService
from pilot.storage import now_ms


def shared_router() -> APIRouter:
    r = APIRouter(prefix="/api")

    @r.get("/time")
    def server_time() -> dict:
        """Devices use this to estimate their clock offset, so events from
        learner laptops and observers' phones line up in time."""
        return {"server_ms": now_ms()}

    @r.get("/meta")
    def meta() -> dict:
        return {"n_questions": SESSION_LENGTH, "checkin_after": sorted(CHECKIN_AFTER),
                "states": {s.value: s.label for s in BehaviouralState},
                "observer_codes": sorted(OBSERVER_CODES)}

    return r


def researcher_router(sessions: SessionService, keys: KeyGuard) -> APIRouter:
    r = APIRouter(prefix="/api/researcher", dependencies=[Depends(keys.researcher)])

    @r.post("/sessions")
    def create_session(body: NewSession) -> dict:
        return sessions.create(**body.model_dump())

    @r.get("/sessions")
    def list_sessions() -> list[dict]:
        return sessions.summaries()

    @r.get("/sessions/{session_id}")
    def session_detail(session_id: int) -> dict:
        return sessions.detail(session_id)

    @r.post("/sessions/{session_id}/close")
    def close_session(session_id: int) -> dict:
        sessions.close(session_id)
        return {"ok": True}

    @r.get("/export/{table}.csv")
    def export_table(table: str):
        body = sessions.export_csv(table)
        return StreamingResponse(iter([body]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="sensai_pilot_{table}.csv"'})

    return r


def observer_router(observation: ObservationService, keys: KeyGuard) -> APIRouter:
    r = APIRouter(prefix="/api/observer", dependencies=[Depends(keys.observer)])

    @r.get("/sessions")
    def open_sessions() -> list[dict]:
        return observation.open_sessions()

    @r.post("/observations")
    def record(body: Observation) -> dict:
        observation.record(body.session_id, body.observer, body.code, body.client_ms)
        return {"ok": True}

    return r


def learner_router(learning: LearningService) -> APIRouter:
    r = APIRouter(prefix="/api/learner")

    @r.post("/join")
    def join(body: Join) -> dict:
        return learning.join(body.code)

    @r.post("/question")
    def current_question(body: LearnerAuth) -> dict:
        return learning.current(body.code, body.token)

    @r.post("/event")
    def event(body: LearnerEvent) -> dict:
        learning.log_event(body.code, body.token, body.type, body.question_index, body.client_ms, body.payload)
        return {"ok": True}

    @r.post("/hint")
    def hint(body: HintRequest) -> dict:
        return {"hint": learning.hint(body.code, body.token, body.question_index, body.client_ms)}

    @r.post("/answer")
    def answer(body: AnswerIn) -> dict:
        return learning.answer(body.code, body.token, Answer(
            question_index=body.question_index, answer=body.answer, response_time_ms=body.response_time_ms,
            first_input_ms=body.first_input_ms, shown_client_ms=body.shown_client_ms,
            answered_client_ms=body.answered_client_ms))

    @r.post("/checkin")
    def checkin(body: CheckinIn) -> dict:
        learning.checkin(body.code, body.token, body.after_question, body.choice, body.client_ms)
        return {"ok": True}

    @r.post("/finish")
    def finish(body: LearnerAuth) -> dict:
        learning.finish(body.code, body.token)
        return {"ok": True}

    return r
