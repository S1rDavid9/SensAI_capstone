"""Guardrails ("shielding"): rules that rule out actions no teacher would
choose, applied on top of a learned policy. Each rule is a small class, so
rules can be added or tested on their own."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from environment.custom_env import ACTION_NAMES
from pilot.core.states import BehaviouralState
from pilot.core.tasks import MAX_LEVEL, MIN_LEVEL
from pilot.policy.dqn import ActionScorer


class GuardrailRule(ABC):
    @abstractmethod
    def ruled_out(self, state: BehaviouralState, difficulty: int) -> dict[str, str]:
        """Actions this rule forbids here, each with the reason."""


@dataclass(frozen=True)
class StateRule(GuardrailRule):
    """Forbid some actions whenever the learner is in a given state."""
    state: BehaviouralState
    reasons: tuple[tuple[str, str], ...]   # (action, reason)

    def ruled_out(self, state, difficulty):
        return dict(self.reasons) if state == self.state else {}


@dataclass(frozen=True)
class DifficultyBoundsRule(GuardrailRule):
    """Don't lower difficulty at the easiest level or raise it at the hardest."""

    def ruled_out(self, state, difficulty):
        out = {}
        if difficulty <= MIN_LEVEL:
            out["decrease_difficulty"] = "already at the easiest level"
        if difficulty >= MAX_LEVEL:
            out["increase_difficulty"] = "already at the hardest level"
        return out


def _not_when(state: BehaviouralState) -> StateRule:
    word = state.label.lower()
    return StateRule(state, (
        ("advance_topic", f"don't move on while the learner is {word}"),
        ("increase_difficulty", f"don't make it harder while the learner is {word}"),
        ("introduce_challenge", f"don't add a challenge while the learner is {word}"),
    ))


DEFAULT_RULES: tuple[GuardrailRule, ...] = (
    StateRule(BehaviouralState.ON_TASK,
              (("decrease_difficulty", "don't make it easier for a learner who is coping"),)),
    _not_when(BehaviouralState.STUCK),
    _not_when(BehaviouralState.STRUGGLING),
    DifficultyBoundsRule(),
)


@dataclass(frozen=True)
class Decision:
    model: str
    raw_action: str            # the policy's own first choice
    action: str                # after guardrails
    guardrail: str | None      # why the raw choice was ruled out, if it was
    scores: tuple[float, ...]

    @property
    def raw_action_index(self) -> int:
        return ACTION_NAMES.index(self.raw_action)

    @property
    def action_index(self) -> int:
        return ACTION_NAMES.index(self.action)


class Shield:
    def __init__(self, scorer: ActionScorer, rules: tuple[GuardrailRule, ...] = DEFAULT_RULES) -> None:
        self.scorer = scorer
        self.rules = rules

    def ruled_out(self, state: BehaviouralState, difficulty: int) -> dict[str, str]:
        blocked: dict[str, str] = {}
        for rule in self.rules:
            for action, reason in rule.ruled_out(state, difficulty).items():
                blocked.setdefault(action, reason)
        return blocked

    def decide(self, observation: np.ndarray, state: BehaviouralState, difficulty: int) -> Decision:
        scores = self.scorer.scores(observation)
        raw = ACTION_NAMES[int(np.argmax(scores))]
        blocked = self.ruled_out(state, difficulty)
        allowed = [i for i, name in enumerate(ACTION_NAMES) if name not in blocked]
        final = ACTION_NAMES[max(allowed, key=lambda i: scores[i])]
        return Decision(
            model=self.scorer.model_id,
            raw_action=raw,
            action=final,
            guardrail=blocked.get(raw) if final != raw else None,
            scores=tuple(round(float(s), 4) for s in scores),
        )
