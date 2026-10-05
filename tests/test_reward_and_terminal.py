"""Reward-value and terminal-condition tests.

Covers: exact reward values per transition type, global reward bounds under
random play, and precise triggering of both terminal conditions (dropout at
exactly the 3rd consecutive disengaged step; session success only when all
topics are reached with no negative excursion).
"""

from __future__ import annotations

import pytest

from environment.custom_env import (
    ACTION_ADVANCE_TOPIC,
    ACTION_CHANGE_TASK_TYPE,
    AdaptLearnEnv,
    EngagementState,
)


class _ForcedChoiceRNG:
    """Wraps a real np.random.Generator but overrides `.choice` to always
    return a caller-specified index. Used to make the otherwise-stochastic
    state transition deterministic, so terminal-condition boundaries (exactly
    step 3, not step 2 or 4) can be tested precisely instead of statistically.
    All other calls (uniform/integers/normal, used by reset() and the
    behavioural-signal simulator) are forwarded unchanged to the real RNG.
    """

    def __init__(self, real_rng, forced_index: int) -> None:
        self._real = real_rng
        self.forced_index = forced_index

    def choice(self, *args, **kwargs):
        return self.forced_index

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def env() -> AdaptLearnEnv:
    e = AdaptLearnEnv(max_episode_steps=50, n_topics=2)
    yield e
    e.close()


# --------------------------------------------------------------------------
# Exact reward values
# --------------------------------------------------------------------------


def test_reward_reach_vs_maintain_engaged(env: AdaptLearnEnv) -> None:
    assert env._compute_reward(EngagementState.BORED, EngagementState.ENGAGED) == pytest.approx(
        env.rc.engaged_reach_reward
    )
    assert env._compute_reward(
        EngagementState.ENGAGED, EngagementState.ENGAGED
    ) == pytest.approx(env.rc.engaged_maintain_reward)


def test_reward_frustrated_bored_confused_values(env: AdaptLearnEnv) -> None:
    assert env._compute_reward(
        EngagementState.ENGAGED, EngagementState.FRUSTRATED
    ) == pytest.approx(env.rc.frustrated_step_reward)
    assert env._compute_reward(EngagementState.ENGAGED, EngagementState.BORED) == pytest.approx(
        env.rc.bored_step_reward
    )
    assert env._compute_reward(
        EngagementState.ENGAGED, EngagementState.CONFUSED
    ) == pytest.approx(env.rc.confused_step_reward)


def test_frustrated_and_bored_penalized_every_step_not_only_on_entry(env: AdaptLearnEnv) -> None:
    """Design decision: staying in FRUSTRATED must keep costing -1 each step,
    not just on the first entry, else an agent could pay once and coast."""
    r_entry = env._compute_reward(EngagementState.BORED, EngagementState.FRUSTRATED)
    r_persist = env._compute_reward(EngagementState.FRUSTRATED, EngagementState.FRUSTRATED)
    assert r_entry == r_persist == pytest.approx(env.rc.frustrated_step_reward)


# --------------------------------------------------------------------------
# Global reward bounds under random play
# --------------------------------------------------------------------------


def test_reward_stays_within_theoretical_bounds() -> None:
    """min possible = frustrated_step_reward + dropout_terminal_penalty + step_cost
    max possible = engaged_reach_reward + topic_advance_bonus + session_complete_bonus + step_cost
    (the max is achievable on a single step: the step that completes the
    final topic can simultaneously be a "reach ENGAGED" transition, earn the
    capped per-topic bonus, and trigger session success)."""
    env = AdaptLearnEnv()
    lower = env.rc.frustrated_step_reward + env.rc.dropout_terminal_penalty + env.rc.step_cost
    upper = (
        env.rc.engaged_reach_reward
        + env.rc.topic_advance_bonus
        + env.rc.session_complete_bonus
        + env.rc.step_cost
    )

    for seed in range(30):
        env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert lower - 1e-9 <= reward <= upper + 1e-9
    env.close()


# --------------------------------------------------------------------------
# Terminal conditions: dropout
# --------------------------------------------------------------------------


def test_dropout_triggers_exactly_on_third_consecutive_disengaged_step(
    env: AdaptLearnEnv,
) -> None:
    env.reset(seed=0)
    env.np_random = _ForcedChoiceRNG(env.np_random, forced_index=int(EngagementState.FRUSTRATED))

    _, _, terminated_1, _, info_1 = env.step(4)  # add_encouragement; action content irrelevant here
    assert terminated_1 is False
    assert info_1["dropout"] is False

    _, _, terminated_2, _, info_2 = env.step(4)
    assert terminated_2 is False
    assert info_2["dropout"] is False

    _, reward_3, terminated_3, _, info_3 = env.step(4)
    assert terminated_3 is True
    assert info_3["dropout"] is True
    # The dropout penalty must be added on top of the ordinary FRUSTRATED
    # step reward (and the flat step cost) on this exact step.
    assert reward_3 == pytest.approx(
        env.rc.frustrated_step_reward + env.rc.dropout_terminal_penalty + env.rc.step_cost
    )


def test_disengagement_streak_resets_on_engaged_step(env: AdaptLearnEnv) -> None:
    env.reset(seed=0)
    env.np_random = _ForcedChoiceRNG(env.np_random, forced_index=int(EngagementState.FRUSTRATED))
    env.step(4)
    env.step(4)
    assert env.consecutive_disengaged == 2

    env.np_random.forced_index = int(EngagementState.ENGAGED)
    env.step(4)
    assert env.consecutive_disengaged == 0

    # Two more disengaged steps should NOT terminate -- the streak was reset.
    env.np_random.forced_index = int(EngagementState.BORED)
    _, _, terminated, _, info = env.step(4)
    assert terminated is False
    assert info["dropout"] is False


def test_confused_does_not_count_toward_dropout_streak(env: AdaptLearnEnv) -> None:
    """Design decision: CONFUSED is excluded from the disengagement streak."""
    env.reset(seed=0)
    env.np_random = _ForcedChoiceRNG(env.np_random, forced_index=int(EngagementState.FRUSTRATED))
    env.step(4)
    env.step(4)
    assert env.consecutive_disengaged == 2

    env.np_random.forced_index = int(EngagementState.CONFUSED)
    _, _, terminated, _, info = env.step(4)
    assert terminated is False
    assert env.consecutive_disengaged == 0


# --------------------------------------------------------------------------
# Terminal conditions: session success
# --------------------------------------------------------------------------


def test_session_success_requires_all_topics_and_no_negative_excursion(
    env: AdaptLearnEnv,
) -> None:
    env.reset(seed=0)
    # reset() randomizes the initial engagement state (by design -- see
    # AdaptLearnEnv.reset docstring), which would otherwise make this test
    # flaky: if seed=0 happens to start FRUSTRATED/BORED, session_had_negative
    # is correctly tainted before this test's forced transitions even begin.
    # Pin a clean starting condition so the test isolates what it's actually
    # checking (the effect of the two forced ENGAGED steps that follow).
    env.engagement_state = EngagementState.ENGAGED
    env.session_had_negative = False
    env.consecutive_disengaged = 0
    env.np_random = _ForcedChoiceRNG(env.np_random, forced_index=int(EngagementState.ENGAGED))

    # env fixture uses n_topics=2.
    _, _, terminated_1, _, info_1 = env.step(ACTION_ADVANCE_TOPIC)
    assert terminated_1 is False
    assert info_1["success"] is False

    _, reward_2, terminated_2, _, info_2 = env.step(ACTION_ADVANCE_TOPIC)
    assert terminated_2 is True
    assert info_2["success"] is True
    # Starting state was pinned to ENGAGED above, so both forced transitions
    # are ENGAGED -> ENGAGED ("maintain"), not a fresh "reach". Both steps are
    # ACTION_ADVANCE_TOPIC with topics_completed <= n_topics(2) and a
    # non-disengaged outcome, so each also earns topic_advance_bonus; this
    # second step additionally triggers session_complete_bonus, and every
    # step carries the flat step_cost.
    assert reward_2 == pytest.approx(
        env.rc.engaged_maintain_reward
        + env.rc.topic_advance_bonus
        + env.rc.step_cost
        + env.rc.session_complete_bonus
    )


def test_session_success_blocked_by_any_prior_negative_excursion(env: AdaptLearnEnv) -> None:
    env.reset(seed=0)
    env.np_random = _ForcedChoiceRNG(env.np_random, forced_index=int(EngagementState.FRUSTRATED))
    env.step(ACTION_ADVANCE_TOPIC)  # taints session_had_negative

    env.np_random.forced_index = int(EngagementState.ENGAGED)
    _, _, terminated, _, info = env.step(ACTION_ADVANCE_TOPIC)  # reaches n_topics=2

    assert info["success"] is False
    assert terminated is False  # neither dropout nor success triggered


# --------------------------------------------------------------------------
# Reward-hacking regression test
# --------------------------------------------------------------------------


def test_progression_beats_camping_in_expected_value() -> None:
    """Regression test for a reward-hacking behavior found via manual policy
    inspection of a trained DQN agent (see RewardConfig's docstring, decision
    10 in the module docstring): the agent learned to reach ENGAGED once and
    then "camp" there for the rest of the episode using harmless filler
    actions, because accumulated per-step maintenance reward outweighed the
    reward for genuinely advancing through the curriculum.

    This compares two *idealized, deterministic* full-episode trajectories
    using the actual configured reward constants -- not signs or bounds:

      * "camp": reach ENGAGED, then take a harmless filler action
        (change_task_type) for the rest of the episode, forced to never
        destabilize. This is camping's best case.
      * "progress": take advance_topic every step, forced to never
        destabilize, until the session completes. This is a clean
        (zero-setback) run through the curriculum -- not even an
        exceptionally lucky one, just an uneventful one.

    A correctly balanced reward config must make the clean progression
    trajectory out-earn the idealized camping trajectory. If a future
    change to RewardConfig breaks this, it has reintroduced the exploit.
    """

    def run_forced(action: int) -> float:
        env = AdaptLearnEnv()
        env.reset(seed=0)
        # Start from a clean, already-ENGAGED baseline so both trajectories
        # are compared on equal footing (isolates "what should the agent do
        # once already engaged", which is exactly what the exploit was
        # about -- not the separate question of how it first got there).
        env.engagement_state = EngagementState.ENGAGED
        env.session_had_negative = False
        env.consecutive_disengaged = 0
        env.np_random = _ForcedChoiceRNG(env.np_random, forced_index=int(EngagementState.ENGAGED))

        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        env.close()
        return total_reward

    camp_total = run_forced(ACTION_CHANGE_TASK_TYPE)
    progress_total = run_forced(ACTION_ADVANCE_TOPIC)

    assert progress_total > camp_total, (
        f"progression ({progress_total:.2f}) should out-earn idealized camping "
        f"({camp_total:.2f}); if this fails, engaged_maintain_reward / "
        f"topic_advance_bonus / session_complete_bonus / step_cost need rebalancing."
    )
