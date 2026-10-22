"""How a session runs. Two strategies share one interface:

- `ShadowMode` (the pilot): every learner gets the fixed 15-question form;
  the DQN is asked what it would do and its answer is stored, never used.
- `AdaptiveMode` (the final MVP): the Stage 1 classifier labels the learner,
  the shielded DQN chooses an action, and the planner turns it into the next
  question and the support the learner sees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pilot.adaptation import CurriculumPlanner
from pilot.core.tasks import SESSION_LENGTH, Question, QuestionBank
from pilot.labelling import StateLabeler
from pilot.policy import Decision


@dataclass(frozen=True)
class Continuation:
    next_question: Question | None
    support: dict | None       # {"message": ..., "hint": ...} shown with the next question


class SessionMode(ABC):
    name: str

    def __init__(self, bank: QuestionBank, labeler: StateLabeler) -> None:
        self.bank = bank
        self.labeler = labeler

    @abstractmethod
    def first_question(self) -> Question: ...

    @abstractmethod
    def continue_after(self, position: int, last: Question, decision: Decision,
                       seen: set[str], session_id: int) -> Continuation:
        """What follows the answer at `position` (0-based)."""


class ShadowMode(SessionMode):
    name = "shadow"

    def first_question(self) -> Question:
        return self.bank.pilot_form()[0]

    def continue_after(self, position, last, decision, seen, session_id):
        form = self.bank.pilot_form()
        nxt = form[position + 1] if position + 1 < SESSION_LENGTH else None
        return Continuation(nxt, None)          # the decision is stored, never used


class AdaptiveMode(SessionMode):
    name = "adaptive"

    def __init__(self, bank: QuestionBank, labeler: StateLabeler, planner: CurriculumPlanner) -> None:
        super().__init__(bank, labeler)
        self.planner = planner

    def first_question(self) -> Question:
        return self.planner.first_question()

    def continue_after(self, position, last, decision, seen, session_id):
        if position + 1 >= SESSION_LENGTH:
            return Continuation(None, None)
        step = self.planner.next_step(decision.action, last, seen, seed=session_id * 100 + position)
        return Continuation(step.question, {"message": step.message, "hint": step.hint})
