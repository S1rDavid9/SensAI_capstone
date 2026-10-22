"""Stage 2: choosing a scaffolding action.

`DQNPolicy` scores the eight actions with the trained Q-network. `Shield`
wraps any `ActionScorer` with guardrail rules ("shielding"): actions a rule
rules out are skipped and the best remaining action is chosen. Both the
policy's own choice and the shielded one are returned.
"""

from pilot.policy.dqn import ActionScorer, DQNPolicy
from pilot.policy.shield import (
    DEFAULT_RULES, Decision, DifficultyBoundsRule, GuardrailRule, Shield, StateRule,
)

__all__ = ["ActionScorer", "DQNPolicy", "Decision", "GuardrailRule", "StateRule",
           "DifficultyBoundsRule", "Shield", "DEFAULT_RULES"]
