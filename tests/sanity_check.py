"""Quick, human-readable, non-visual sanity check for AdaptLearnEnv.

This is intentionally separate from the pytest suite (tests/test_env.py):
it exists to be run once, by eye, to build confidence the core RL loop
works *before* any time is spent on the PyOpenGL rendering layer. Run with:

    uv run python tests/sanity_check.py
"""

from __future__ import annotations

import numpy as np

from environment.custom_env import ACTION_NAMES, AdaptLearnEnv


def run_random_episode(env: AdaptLearnEnv, seed: int) -> None:
    obs, info = env.reset(seed=seed)
    print(f"\n=== Episode (seed={seed}) ===")
    print(f"initial ability={info['ability']:.2f} difficulty={info['difficulty']} "
          f"state={info['engagement_state']}")
    assert env.observation_space.contains(obs), "reset() observation out of bounds"

    total_reward = 0.0
    reward_min, reward_max = float("inf"), float("-inf")
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs), f"step {step} observation out of bounds: {obs}"
        assert isinstance(reward, float), "reward must be a python float"

        total_reward += reward
        reward_min = min(reward_min, reward)
        reward_max = max(reward_max, reward)
        step += 1

        print(
            f"  t={step:02d} action={ACTION_NAMES[action]:<20s} "
            f"state={info['engagement_state']:<10s} reward={reward:+.2f} "
            f"gap={info['gap']:+.2f} consec_disengaged={info['consecutive_disengaged']}"
        )

    outcome = "DROPOUT" if info["dropout"] else "SUCCESS" if info["success"] else "TRUNCATED (max steps)"
    print(f"--- episode end after {step} steps: {outcome} | "
          f"total_reward={total_reward:+.2f} | reward range=[{reward_min:+.2f}, {reward_max:+.2f}]")


def main() -> None:
    env = AdaptLearnEnv()

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    for seed in range(5):
        run_random_episode(env, seed)

    env.close()
    print("\nSanity check complete: environment ran end-to-end with no errors.")


if __name__ == "__main__":
    main()
