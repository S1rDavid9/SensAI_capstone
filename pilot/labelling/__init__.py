"""Turning a learner's recent telemetry into a behavioural state.

Two interchangeable labellers share the `StateLabeler` interface:
`RuleBasedLabeler` (the provisional rules used during the pilot, before any
data existed) and `Stage1Labeler` (the classifier trained on the pilot's
check-ins). Services depend only on the interface.
"""

from pilot.labelling.base import StateLabeler
from pilot.labelling.rules import RuleBasedLabeler
from pilot.labelling.stage1 import Stage1Labeler

__all__ = ["RuleBasedLabeler", "Stage1Labeler", "StateLabeler"]
