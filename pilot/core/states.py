"""The behavioural states SensAI works with, and how they map onto the DQN.

States describe what a learner is observably doing, not how they feel. The
DQN was trained in a simulator whose four one-hot state slots are named
FRUSTRATED, BORED, CONFUSED and ENGAGED; each behavioural state is assigned
to one slot so the trained policy can be reused unchanged.
"""

from __future__ import annotations

from enum import Enum


class BehaviouralState(str, Enum):
    ON_TASK = "on_task"
    RUSHING = "rushing"          # under-challenged: fast, careless
    STRUGGLING = "struggling"    # over-challenged: slow, many errors
    STUCK = "stuck"              # not attempting

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def dqn_slot(self) -> int:
        """Index of the simulator's one-hot slot (FRUSTRATED, BORED, CONFUSED, ENGAGED)."""
        return _SLOTS[self]


_LABELS = {
    BehaviouralState.ON_TASK: "On task",
    BehaviouralState.RUSHING: "Rushing",
    BehaviouralState.STRUGGLING: "Struggling",
    BehaviouralState.STUCK: "Stuck",
}
_SLOTS = {
    BehaviouralState.STRUGGLING: 0,
    BehaviouralState.RUSHING: 1,
    BehaviouralState.STUCK: 2,
    BehaviouralState.ON_TASK: 3,
}

OBSERVER_CODES = frozenset({*(s.value for s in BehaviouralState), "off_task", "cant_see"})
CHECKIN_CHOICES = frozenset({"easy", "good", "hard", "lost"})
LEARNER_EVENT_TYPES = frozenset({
    "question_shown", "first_input", "input_changed", "idle",
    "focus_lost", "focus_returned", "session_resumed",
})
