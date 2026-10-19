"""Provisional behavioural state labels and the DQN's observation vector.

The Stage 1 classifier is trained on pilot data, so it cannot exist during
the pilot. To still log what the Stage 2 policy would do (shadow mode), each
answered question gets a provisional, rule-based behavioural state. The
thresholds below are placeholders, not findings. They are stored with every
decision (`state_source = "provisional_rules_v1"`) so the analysis can tell
these apart from Stage 1 predictions later.

The four behavioural states map onto the four state slots the DQN was
trained with. The simulator's internal names are kept only as slot labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Behavioural state -> index of the DQN's one-hot slot
# (simulator order: FRUSTRATED, BORED, CONFUSED, ENGAGED).
STATES = ("struggling", "rushing", "stuck", "on_task")
STATE_TO_SLOT = {"struggling": 0, "rushing": 1, "stuck": 2, "on_task": 3}

STATE_LABELS = {
    "on_task": "On task",
    "rushing": "Rushing",
    "struggling": "Struggling",
    "stuck": "Stuck",
}

# --- Provisional thresholds (placeholders, to be replaced by Stage 1) ---
RUSH_MAX_SECONDS = 4.0          # answered wrong faster than this -> rushing
STUCK_MIN_FIRST_INPUT_S = 20.0  # no typing for this long -> stuck
STRUGGLE_MIN_ERROR_RATE = 0.5   # recent error rate at or above this ...
STRUGGLE_MIN_SECONDS = 15.0     # ... and slow answers -> struggling
RECENT_WINDOW = 3               # questions in the rolling error rate

# --- Mapping real telemetry onto the simulator's [0, 1] feature scales ---
RESPONSE_TIME_CAP_S = 40.0  # 40 s maps to 1.0; typical on-task answers land mid-range
HINT_COUNT_CAP = 5.0        # same cap the simulator uses
TIME_ON_TASK_CAP = 10.0     # same cap the simulator uses, counted in questions


@dataclass
class AnswerTelemetry:
    response_time_s: float          # question shown -> answer submitted
    first_input_s: float | None     # question shown -> first keystroke
    correct: bool
    hints_used: int
    difficulty: int                 # 1..5
    recent_errors: list[bool]       # last RECENT_WINDOW answers, oldest first, incl. this one
    questions_at_level: int         # questions answered at this difficulty so far, incl. this one


def provisional_state(t: AnswerTelemetry) -> str:
    error_rate = sum(t.recent_errors) / max(1, len(t.recent_errors))
    if t.first_input_s is None or t.first_input_s >= STUCK_MIN_FIRST_INPUT_S:
        return "stuck"
    if not t.correct and t.response_time_s < RUSH_MAX_SECONDS:
        return "rushing"
    if error_rate >= STRUGGLE_MIN_ERROR_RATE and t.response_time_s >= STRUGGLE_MIN_SECONDS:
        return "struggling"
    return "on_task"


def build_observation(state: str, t: AnswerTelemetry) -> np.ndarray:
    """The 9-dim vector the DQN expects: one-hot state slot, then five
    telemetry features. Stage 1 will replace only the one-hot block; the
    other five come straight from the logged telemetry."""
    obs = np.zeros(9, dtype=np.float32)
    obs[STATE_TO_SLOT[state]] = 1.0
    obs[4] = min(1.0, t.response_time_s / RESPONSE_TIME_CAP_S)
    obs[5] = sum(t.recent_errors) / max(1, len(t.recent_errors))
    obs[6] = min(1.0, t.hints_used / HINT_COUNT_CAP)
    obs[7] = (t.difficulty - 1) / 4
    obs[8] = min(1.0, t.questions_at_level / TIME_ON_TASK_CAP)
    return obs
