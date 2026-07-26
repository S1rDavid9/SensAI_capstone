"""Structural sanity tests: action/observation spaces stay valid under all use.

These tests exist to catch the class of bug that silently corrupts training
without ever crashing -- an observation that drifts outside [0, 1], a reward
of the wrong type, a one-hot block that doesn't sum to 1. SB3 does not
validate any of this for you at runtime.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.custom_env import N_ACTIONS, N_STATES, AdaptLearnEnv


@pytest.fixture
def env() -> AdaptLearnEnv:
    e = AdaptLearnEnv()
    yield e
    e.close()


def test_action_space_is_discrete_8(env: AdaptLearnEnv) -> None:
    assert env.action_space.n == N_ACTIONS == 8


def test_observation_space_shape_dtype(env: AdaptLearnEnv) -> None:
    assert env.observation_space.shape == (9,)
    assert env.observation_space.dtype == np.float32
    assert np.all(env.observation_space.low == 0.0)
    assert np.all(env.observation_space.high == 1.0)


def test_reset_observation_in_space(env: AdaptLearnEnv) -> None:
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        assert env.observation_space.contains(obs), f"seed={seed}: {obs}"
        assert obs.dtype == np.float32


def test_step_observations_always_in_space(env: AdaptLearnEnv) -> None:
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        if terminated or truncated:
            obs, _ = env.reset()


def test_one_hot_engagement_block_is_valid(env: AdaptLearnEnv) -> None:
    """obs[0:4] must be a one-hot vector matching env.engagement_state."""
    obs, _ = env.reset(seed=1)
    for _ in range(100):
        one_hot = obs[0:4]
        assert one_hot.sum() == pytest.approx(1.0)
        assert set(np.unique(one_hot)) <= {0.0, 1.0}
        active_index = int(np.argmax(one_hot))
        assert active_index == int(env.engagement_state)
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()


def test_normalized_features_stay_bounded(env: AdaptLearnEnv) -> None:
    """Counters that could in principle grow unboundedly (hint count, time on
    task) must still normalize into [0, 1] via the documented caps, even when
    an action is spammed far past the cap."""
    env.reset(seed=2)
    for _ in range(60):
        obs, _, terminated, truncated, _ = env.step(3)  # offer_hint, repeatedly
        assert 0.0 <= obs[6] <= 1.0
        assert 0.0 <= obs[8] <= 1.0
        if terminated or truncated:
            env.reset()


def test_invalid_action_raises(env: AdaptLearnEnv) -> None:
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step(N_ACTIONS)


def test_engagement_state_count_matches_observation_block() -> None:
    assert N_STATES == 4


def test_gymnasium_check_env_compliance() -> None:
    """Runs Gymnasium's official environment API checker as a regression test,
    not just a one-off manual sanity run."""
    check_env(AdaptLearnEnv(), skip_render_check=True)
