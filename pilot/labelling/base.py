from __future__ import annotations

from abc import ABC, abstractmethod

from pilot.core.states import BehaviouralState
from pilot.core.telemetry import LearnerHistory


class StateLabeler(ABC):
    """Decides a learner's behavioural state after each answer."""

    #: Stored with every decision, so analysis can tell labellers apart.
    source: str

    @abstractmethod
    def label(self, history: LearnerHistory) -> BehaviouralState:
        """`history` always contains at least one answer."""

    def warm_up(self) -> None:
        """Load anything slow before the first learner arrives (default: nothing)."""
