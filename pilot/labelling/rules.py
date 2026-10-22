"""The provisional rules used during the pilot.

They existed only so the DQN could run in shadow mode before any labelled
data did. The thresholds are placeholders, not findings; the pilot showed
the "stuck" rule fires far too often on word problems, which take longer
than 20 seconds to read.
"""

from __future__ import annotations

from dataclasses import dataclass

from pilot.core.states import BehaviouralState
from pilot.core.telemetry import RECENT_WINDOW, LearnerHistory
from pilot.labelling.base import StateLabeler


@dataclass(frozen=True)
class RuleBasedLabeler(StateLabeler):
    rush_max_seconds: float = 4.0         # answered wrong faster than this -> rushing
    stuck_min_first_input_s: float = 20.0 # no typing for this long -> stuck
    struggle_min_error_rate: float = 0.5  # recent error rate at or above this ...
    struggle_min_seconds: float = 15.0    # ... and slow answers -> struggling
    source: str = "provisional_rules_v1"

    def label(self, history: LearnerHistory) -> BehaviouralState:
        a = history.latest
        if a.first_input_s is None or a.first_input_s >= self.stuck_min_first_input_s:
            return BehaviouralState.STUCK
        if not a.correct and a.response_time_s < self.rush_max_seconds:
            return BehaviouralState.RUSHING
        if (history.error_rate(RECENT_WINDOW) >= self.struggle_min_error_rate
                and a.response_time_s >= self.struggle_min_seconds):
            return BehaviouralState.STRUGGLING
        return BehaviouralState.ON_TASK
