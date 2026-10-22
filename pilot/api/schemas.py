from __future__ import annotations

from pydantic import BaseModel, Field


class NewSession(BaseModel):
    guardian_consent: bool
    learner_assent: bool
    recorded_by: str = Field(min_length=1, max_length=60)
    notes: str = Field(default="", max_length=500)
    mode: str = "shadow"


class Observation(BaseModel):
    session_id: int
    observer: str = Field(min_length=1, max_length=40)
    code: str
    client_ms: int | None = None


class Join(BaseModel):
    code: str = Field(min_length=2, max_length=10)


class LearnerAuth(BaseModel):
    code: str
    token: str


class LearnerEvent(LearnerAuth):
    type: str
    question_index: int | None = None
    client_ms: int | None = None
    payload: dict = Field(default_factory=dict)


class HintRequest(LearnerAuth):
    question_index: int
    client_ms: int | None = None


class AnswerIn(LearnerAuth):
    question_index: int
    answer: str = Field(max_length=40)
    response_time_ms: int = Field(ge=0)
    first_input_ms: int | None = Field(default=None, ge=0)
    shown_client_ms: int | None = None
    answered_client_ms: int | None = None


class CheckinIn(LearnerAuth):
    after_question: int
    choice: str
    client_ms: int | None = None
