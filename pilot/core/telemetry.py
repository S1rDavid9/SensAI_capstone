"""What the app knows about a learner after each answer, and how it becomes
the 9-number observation the DQN expects."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np

from pilot.core.states import BehaviouralState

RECENT_WINDOW = 3            # answers in the rolling error rate the DQN sees
RESPONSE_TIME_CAP_S = 40.0   # 40 s maps to 1.0
HINT_COUNT_CAP = 5.0         # same cap as the simulator
TIME_ON_TASK_CAP = 10.0      # same cap as the simulator, counted in questions


@dataclass(frozen=True)
class AnswerRecord:
    """One answered question, as stored."""
    question_index: int
    difficulty: int
    correct: bool
    response_time_s: float
    first_input_s: float | None
    hints_used: int


@dataclass(frozen=True)
class LearnerHistory:
    """A learner's answers so far, oldest first."""
    answers: tuple[AnswerRecord, ...]

    @property
    def latest(self) -> AnswerRecord:
        return self.answers[-1]

    def last(self, n: int) -> tuple[AnswerRecord, ...]:
        return self.answers[-n:]

    def error_rate(self, n: int) -> float:
        recent = self.last(n)
        return sum(not a.correct for a in recent) / len(recent)

    def median_response_time(self, n: int) -> float:
        return float(median(a.response_time_s for a in self.last(n)))

    def answers_at_level(self, difficulty: int) -> int:
        return sum(a.difficulty == difficulty for a in self.answers)


class ObservationBuilder:
    """Builds the DQN's input: the state's one-hot slot plus five telemetry
    features scaled to the simulator's [0, 1] ranges. A labeller decides the
    state; the other five numbers always come straight from the telemetry."""

    def build(self, state: BehaviouralState, history: LearnerHistory) -> np.ndarray:
        a = history.latest
        obs = np.zeros(9, dtype=np.float32)
        obs[state.dqn_slot] = 1.0
        obs[4] = min(1.0, a.response_time_s / RESPONSE_TIME_CAP_S)
        obs[5] = history.error_rate(RECENT_WINDOW)
        obs[6] = min(1.0, a.hints_used / HINT_COUNT_CAP)
        obs[7] = (a.difficulty - 1) / 4
        obs[8] = min(1.0, history.answers_at_level(a.difficulty) / TIME_ON_TASK_CAP)
        return obs
