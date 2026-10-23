"""Everything a learner's device can do: join, get the current question,
report interaction events, open a hint, answer, check in, finish."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass

from pilot.core.states import CHECKIN_CHOICES, LEARNER_EVENT_TYPES
from pilot.core.tasks import CHECKIN_AFTER, SESSION_LENGTH, Question, QuestionBank
from pilot.core.telemetry import ObservationBuilder
from pilot.policy import Shield
from pilot.services.errors import Conflict, InvalidRequest, NotAllowed, NotFound
from pilot.services.modes import SessionMode
from pilot.storage import Database, Repositories


@dataclass(frozen=True)
class Answer:
    question_index: int
    answer: str
    response_time_ms: int
    first_input_ms: int | None
    shown_client_ms: int | None
    answered_client_ms: int | None


class LearningService:
    def __init__(self, db: Database, bank: QuestionBank, modes: dict[str, SessionMode], shield: Shield,
                 observations: ObservationBuilder | None = None) -> None:
        self.db = db
        self.bank = bank
        self.modes = modes
        self.shield = shield
        self.observations = observations or ObservationBuilder()

    # -- helpers -----------------------------------------------------------

    def _open_session(self, repos: Repositories, code: str, token: str) -> dict:
        s = repos.sessions.by_code(code)
        if not s or not s["learner_token"] or not secrets.compare_digest(s["learner_token"], token):
            raise NotAllowed("This session is not open on this device. Ask the researcher.")
        if s["status"] not in ("ready", "active"):
            raise Conflict("This session has ended.")
        return s

    def _current_question(self, s: dict) -> Question:
        if s["current_question_id"]:
            return self.bank.get(s["current_question_id"])
        return self.bank.pilot_form()[s["current_index"]]   # sessions created before adaptive mode existed

    @staticmethod
    def _summary(repos: Repositories, session_id: int) -> dict:
        """End-of-session summary shown to the learner in adaptive mode."""
        answers = repos.answers.for_session(session_id)
        return {"correct": sum(a["correct"] for a in answers), "total": len(answers),
                "highest_level": max((a["difficulty"] for a in answers), default=1)}

    # -- use cases ---------------------------------------------------------

    def join(self, code: str) -> dict:
        with self.db.connect() as con:
            repos = Repositories(con)
            s = repos.sessions.by_code(code.strip())
            if not s:
                raise NotFound("That code doesn't match a session. Check it with the researcher.")
            if s["status"] not in ("ready", "active"):
                raise Conflict("This session has already finished.")
            token = secrets.token_urlsafe(24)
            resumed = s["status"] == "active"
            repos.sessions.activate(s["id"], token, self.modes[s["mode"]].first_question().id)
            if resumed:
                repos.events.log(s["id"], "session_resumed", s["current_index"], None)
            return {"participant_code": s["participant_code"], "token": token, "mode": s["mode"],
                    "current_index": s["current_index"], "n_questions": SESSION_LENGTH, "resumed": resumed}

    def current(self, code: str, token: str) -> dict:
        with self.db.connect() as con:
            repos = Repositories(con)
            s = self._open_session(repos, code, token)
            if s["current_index"] >= SESSION_LENGTH:
                if s["mode"] == "adaptive":
                    return {"done": True, "summary": self._summary(repos, s["id"])}
                return {"done": True}
            out = {"done": False, "question": self._current_question(s).public(s["current_index"])}
            if s["pending_support"]:
                out["support"] = json.loads(s["pending_support"])
            return out

    def log_event(self, code: str, token: str, type_: str, question_index: int | None,
                  client_ms: int | None, payload: dict) -> None:
        if type_ not in LEARNER_EVENT_TYPES:
            raise InvalidRequest(f"Unknown event type {type_!r}.")
        with self.db.connect() as con:
            repos = Repositories(con)
            s = self._open_session(repos, code, token)
            repos.events.log(s["id"], type_, question_index, client_ms, payload)

    def hint(self, code: str, token: str, question_index: int, client_ms: int | None) -> str:
        with self.db.connect() as con:
            repos = Repositories(con)
            s = self._open_session(repos, code, token)
            if question_index != s["current_index"]:
                raise Conflict("That question is not the current one.")
            repos.events.log(s["id"], "hint_requested", question_index, client_ms)
            return self._current_question(s).hint

    def answer(self, code: str, token: str, a: Answer) -> dict:
        with self.db.connect() as con:
            repos = Repositories(con)
            s = self._open_session(repos, code, token)
            if a.question_index != s["current_index"]:
                raise Conflict("That question is not the current one.")
            question = self._current_question(s)
            correct = question.is_correct(a.answer)
            hints = repos.events.count(s["id"], question_index=a.question_index, type_="hint_requested")
            repos.answers.add(
                session_id=s["id"], question_index=a.question_index, question_id=question.id, kind=question.kind,
                difficulty=question.difficulty, given=a.answer.strip(), correct=correct,
                response_time_ms=a.response_time_ms, first_input_ms=a.first_input_ms, hints_used=hints,
                shown_client_ms=a.shown_client_ms, answered_client_ms=a.answered_client_ms)

            # Stage 1 -> Stage 2: label the learner, then ask the shielded policy.
            mode = self.modes[s["mode"]]
            history = repos.answers.history(s["id"])
            state = mode.labeler.label(history)
            observation = self.observations.build(state, history)
            decision = self.shield.decide(observation, state, question.difficulty)
            repos.decisions.add(session_id=s["id"], question_index=a.question_index, state=state.value,
                                state_source=mode.labeler.source,
                                observation=[round(float(v), 4) for v in observation], decision=decision)

            nxt = mode.continue_after(a.question_index, question, decision,
                                      repos.answers.seen_question_ids(s["id"]), s["id"])
            next_index = a.question_index + 1
            repos.sessions.advance(s["id"], next_index, nxt.next_question.id if nxt.next_question else None,
                                   nxt.support)
            return {"correct": correct, "correct_answer": question.answer,
                    "checkin_due": a.question_index in CHECKIN_AFTER,
                    "finished": next_index >= SESSION_LENGTH, "next_index": next_index}

    def checkin(self, code: str, token: str, after_question: int, choice: str, client_ms: int | None) -> None:
        if choice not in CHECKIN_CHOICES:
            raise InvalidRequest(f"Unknown choice {choice!r}.")
        with self.db.connect() as con:
            repos = Repositories(con)
            s = self._open_session(repos, code, token)
            repos.checkins.add(s["id"], after_question, choice, client_ms)

    def finish(self, code: str, token: str) -> None:
        with self.db.connect() as con:
            repos = Repositories(con)
            s = self._open_session(repos, code, token)
            if s["current_index"] < SESSION_LENGTH:
                raise Conflict("There are still questions left.")
            repos.sessions.finish(s["id"])
