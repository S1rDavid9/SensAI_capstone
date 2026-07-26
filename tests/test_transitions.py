"""Tests for the engagement-state transition model itself.

These are deliberately white-box: they call `_transition_logits` and the
module-level `_softmax` helper directly, rather than only sampling stochastic
episodes, so that the theory-grounded *direction* of each effect (e.g.
"large positive gap should make FRUSTRATED the dominant outcome") is checked
exactly rather than inferred from noisy samples. This is what makes the
transition model defensible in a report: each claim in the design rationale
has a corresponding assertion here.
"""

from __future__ import annotations

import pytest

from environment.custom_env import (
    ACTION_ADD_ENCOURAGEMENT,
    ACTION_CHANGE_TASK_TYPE,
    ACTION_INTRODUCE_CHALLENGE,
    ACTION_OFFER_HINT,
    EngagementState,
    AdaptLearnEnv,
    _softmax,
)


@pytest.fixture
def env() -> AdaptLearnEnv:
    e = AdaptLearnEnv()
    e.reset(seed=0)
    return e


def probs_for(env: AdaptLearnEnv, gap: float, prev_state: EngagementState, action: int):
    logits = env._transition_logits(gap=gap, prev_state=prev_state, action=action)
    return _softmax(logits, temperature=env.tc.softmax_temperature)


def test_well_matched_gap_maximizes_engaged_probability(env: AdaptLearnEnv) -> None:
    """Flow Theory: challenge ~= skill should be the condition most likely to
    produce ENGAGED, more so than a large mismatch in either direction."""
    p_matched = probs_for(env, gap=0.0, prev_state=EngagementState.CONFUSED, action=None)
    p_too_hard = probs_for(env, gap=3.5, prev_state=EngagementState.CONFUSED, action=None)
    p_too_easy = probs_for(env, gap=-3.5, prev_state=EngagementState.CONFUSED, action=None)

    assert p_matched[EngagementState.ENGAGED] > p_too_hard[EngagementState.ENGAGED]
    assert p_matched[EngagementState.ENGAGED] > p_too_easy[EngagementState.ENGAGED]


def test_large_positive_gap_makes_frustrated_dominant(env: AdaptLearnEnv) -> None:
    """Difficulty far above ability -> FRUSTRATED should be the single most
    likely outcome (Flow Theory: challenge >> skill -> anxiety/frustration)."""
    probs = probs_for(env, gap=4.0, prev_state=EngagementState.ENGAGED, action=None)
    assert int(probs.argmax()) == int(EngagementState.FRUSTRATED)


def test_large_negative_gap_makes_bored_dominant(env: AdaptLearnEnv) -> None:
    """Difficulty far below ability -> BORED should dominate (Flow Theory:
    skill >> challenge -> boredom)."""
    probs = probs_for(env, gap=-4.0, prev_state=EngagementState.ENGAGED, action=None)
    assert int(probs.argmax()) == int(EngagementState.BORED)


def test_encouragement_reduces_frustration_relief_holding_gap_fixed(env: AdaptLearnEnv) -> None:
    """SDT / Growth Mindset: encouragement should lower frustration risk and
    raise engagement odds for the same objective difficulty gap."""
    gap = 3.0
    baseline = probs_for(env, gap=gap, prev_state=EngagementState.FRUSTRATED, action=None)
    with_encouragement = probs_for(
        env, gap=gap, prev_state=EngagementState.FRUSTRATED, action=ACTION_ADD_ENCOURAGEMENT
    )
    assert with_encouragement[EngagementState.FRUSTRATED] < baseline[EngagementState.FRUSTRATED]
    assert with_encouragement[EngagementState.ENGAGED] > baseline[EngagementState.ENGAGED]


def test_hint_reduces_confusion_probability(env: AdaptLearnEnv) -> None:
    gap = 1.0
    baseline = probs_for(env, gap=gap, prev_state=EngagementState.CONFUSED, action=None)
    with_hint = probs_for(env, gap=gap, prev_state=EngagementState.CONFUSED, action=ACTION_OFFER_HINT)
    assert with_hint[EngagementState.CONFUSED] < baseline[EngagementState.CONFUSED]


def test_challenge_action_reduces_boredom_and_raises_engagement(env: AdaptLearnEnv) -> None:
    gap = -3.0
    baseline = probs_for(env, gap=gap, prev_state=EngagementState.BORED, action=None)
    with_challenge = probs_for(
        env, gap=gap, prev_state=EngagementState.BORED, action=ACTION_INTRODUCE_CHALLENGE
    )
    assert with_challenge[EngagementState.BORED] < baseline[EngagementState.BORED]
    assert with_challenge[EngagementState.ENGAGED] > baseline[EngagementState.ENGAGED]


def test_format_change_raises_confusion_risk(env: AdaptLearnEnv) -> None:
    """An abrupt task-format change is modeled as momentarily disorienting."""
    gap = 0.0
    baseline = probs_for(env, gap=gap, prev_state=EngagementState.ENGAGED, action=None)
    with_change = probs_for(
        env, gap=gap, prev_state=EngagementState.ENGAGED, action=ACTION_CHANGE_TASK_TYPE
    )
    assert with_change[EngagementState.CONFUSED] > baseline[EngagementState.CONFUSED]


def test_persistence_bonus_favors_staying_in_current_state(env: AdaptLearnEnv) -> None:
    """Affective inertia: at a neutral gap, whichever state the learner is
    already in should be at least as likely to persist as any other specific
    state would be to newly become dominant."""
    gap = 0.0
    p_from_bored = probs_for(env, gap=gap, prev_state=EngagementState.BORED, action=None)
    p_from_confused = probs_for(env, gap=gap, prev_state=EngagementState.CONFUSED, action=None)

    # Same gap, different prior state -> the persistence term should shift
    # probability mass toward whichever state was current.
    assert p_from_bored[EngagementState.BORED] > p_from_confused[EngagementState.BORED]
    assert p_from_confused[EngagementState.CONFUSED] > p_from_bored[EngagementState.CONFUSED]


def test_transition_probabilities_always_sum_to_one(env: AdaptLearnEnv) -> None:
    for gap in (-4.0, -1.0, 0.0, 1.0, 4.0):
        for state in EngagementState:
            for action in (None, 0, 1, 2, 3, 4, 5, 6, 7):
                probs = probs_for(env, gap=gap, prev_state=state, action=action)
                assert probs.sum() == pytest.approx(1.0)
                assert (probs >= 0.0).all()
