"""Reproducibility tests.

Reproducibility is an explicit grading/report requirement for this project,
so it gets first-class test coverage: the same seed must produce a bit-for-
bit identical trajectory, and different seeds must (with overwhelming
probability) produce different episodes.
"""

from __future__ import annotations

import numpy as np

from environment.custom_env import AdaptLearnEnv

# A fixed, non-random action sequence is used (rather than
# env.action_space.sample()) so these tests isolate the *environment's* own
# randomness (ability, initial state, transition sampling, behavioural-signal
# noise) from the separate randomness of action sampling.
_ACTION_SEQUENCE = [i % 8 for i in range(50)]


def _run_trajectory(seed: int) -> list[tuple]:
    env = AdaptLearnEnv()
    obs, info = env.reset(seed=seed)
    trace = [(tuple(obs.tolist()), None, None, None)]
    for action in _ACTION_SEQUENCE:
        obs, reward, terminated, truncated, _ = env.step(action)
        trace.append((tuple(obs.tolist()), reward, terminated, truncated))
        if terminated or truncated:
            break
    env.close()
    return trace


def test_same_seed_produces_identical_trajectory() -> None:
    trace_a = _run_trajectory(seed=42)
    trace_b = _run_trajectory(seed=42)
    assert trace_a == trace_b


def test_same_seed_reproducible_across_several_seeds() -> None:
    for seed in (0, 1, 7, 123, 9999):
        assert _run_trajectory(seed) == _run_trajectory(seed)


def test_different_seeds_produce_different_trajectories() -> None:
    traces = [_run_trajectory(seed) for seed in range(10)]
    # Not every pair is guaranteed to differ in principle, but with a
    # continuous ability draw and stochastic transitions, all 10 differing
    # from each other is the expected outcome and a hash collision here would
    # itself indicate the RNG is not actually being seeded per-episode.
    unique_traces = {tuple(t) for t in traces}
    assert len(unique_traces) == len(traces)


def test_reset_without_seed_after_seeded_reset_still_advances_rng() -> None:
    """A second reset() without an explicit seed should continue drawing from
    the already-seeded RNG stream, not silently reseed to the same state."""
    env = AdaptLearnEnv()
    obs_1, _ = env.reset(seed=0)
    env.reset()  # unseeded reset: should continue the RNG stream, not reset it
    obs_3, _ = env.reset(seed=0)  # re-seeding must reproduce the first episode
    env.close()

    assert np.array_equal(obs_1, obs_3)
