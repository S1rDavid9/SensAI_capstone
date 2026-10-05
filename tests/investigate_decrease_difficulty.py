"""Diagnostic script investigating why decrease_difficulty became the
dominant action in the post-reward-fix DQN policy (see the cross-tab
produced by tests/manual_policy_inspection.py). Not a fix, not a formal
test -- this distinguishes two hypotheses empirically:

  A) RATIONAL CORRECTION: advance_topic permanently raises stored
     `difficulty` (identical mechanism to increase_difficulty -- see
     custom_env.py's step(), branch for ACTION_ADVANCE_TOPIC). offer_hint
     and repeat_simplified only cushion the single step they're used on
     (an "effective_gap" adjustment local to that transition, never
     touching stored `difficulty` -- see _transition_logits). So
     decrease_difficulty is the *only* action that can walk difficulty
     back down persistently after a run of advances. If this hypothesis is
     right, decrease_difficulty calls should cluster shortly after
     advance_topic calls, and should occur when the true gap is positive
     (difficulty ahead of ability -- correction genuinely warranted).

  B) FILLER OF CONVENIENCE: like change_task_type/repeat_simplified before
     the reward fix, decrease_difficulty is simply low-risk while ENGAGED
     (the transition model's Gaussian match window is fairly wide -- see
     TransitionConfig.match_sigma) and pays the same flat maintenance
     reward as any other harmless action. If this hypothesis is right,
     decrease_difficulty calls should be roughly uncorrelated with recent
     advance_topic calls or with the true gap's sign.

The true difficulty-ability gap is NOT observable by the agent (see
custom_env.py's module docstring, design decision 1) -- it is read here
directly from environment internals purely for this after-the-fact
diagnosis, not fed to the policy.

Run with:
    uv run python tests/investigate_decrease_difficulty.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from environment.custom_env import (
    ACTION_ADVANCE_TOPIC,
    ACTION_DECREASE_DIFFICULTY,
    AdaptLearnEnv,
)

MODEL_PATH = Path("models/dqn/dqn_02/model.zip")
N_EPISODES = 20
LOOKBACK_WINDOW = 3  # "recent" advance_topic = within this many prior steps


def main() -> None:
    print(f"Loading model from {MODEL_PATH} ...")
    model = DQN.load(str(MODEL_PATH))
    env = AdaptLearnEnv()

    gaps_before_decrease: list[float] = []
    difficulty_before_decrease: list[int] = []
    preceded_by_recent_advance: list[bool] = []
    steps_since_last_advance: list[int | None] = []

    difficulty_trajectories: list[list[int]] = []
    action_trajectories: list[list[int]] = []

    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep)
        terminated = truncated = False
        action_history: list[int] = []
        difficulty_history: list[int] = [env.difficulty]

        while not (terminated or truncated):
            difficulty_before = env.difficulty
            ability = env.ability
            gap_before = difficulty_before - ability

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            if action == ACTION_DECREASE_DIFFICULTY:
                gaps_before_decrease.append(gap_before)
                difficulty_before_decrease.append(difficulty_before)

                # How many steps back was the most recent advance_topic?
                last_advance_idx = None
                for back in range(1, min(LOOKBACK_WINDOW, len(action_history)) + 1):
                    if action_history[-back] == ACTION_ADVANCE_TOPIC:
                        last_advance_idx = back
                        break
                preceded_by_recent_advance.append(last_advance_idx is not None)
                steps_since_last_advance.append(last_advance_idx)

            obs, reward, terminated, truncated, info = env.step(action)
            action_history.append(action)
            difficulty_history.append(env.difficulty)

        difficulty_trajectories.append(difficulty_history)
        action_trajectories.append(action_history)

    env.close()

    n = len(gaps_before_decrease)
    print(f"\nTotal decrease_difficulty calls observed: {n} (across {N_EPISODES} episodes)\n")

    if n == 0:
        print("No decrease_difficulty calls observed -- nothing to analyze.")
        return

    gaps = np.array(gaps_before_decrease)
    print("=" * 70)
    print("HYPOTHESIS A CHECK 1: was the true gap positive (correction warranted)")
    print("=" * 70)
    print(f"  gap > 0  (difficulty ahead of ability): {np.mean(gaps > 0) * 100:5.1f}%")
    print(f"  gap ~ 0  (-0.5 <= gap <= 0.5, already matched): {np.mean((gaps >= -0.5) & (gaps <= 0.5)) * 100:5.1f}%")
    print(f"  gap < -0.5 (difficulty already below ability): {np.mean(gaps < -0.5) * 100:5.1f}%")
    print(f"  mean gap before decrease_difficulty: {gaps.mean():+.2f}  (median: {np.median(gaps):+.2f})")

    print("\n" + "=" * 70)
    print("HYPOTHESIS A CHECK 2: did a recent advance_topic precede the decrease")
    print("=" * 70)
    preceded_pct = 100 * sum(preceded_by_recent_advance) / n
    print(f"  decrease_difficulty calls preceded by advance_topic within last "
          f"{LOOKBACK_WINDOW} steps: {preceded_pct:.1f}%")
    lookback_counts = Counter(s for s in steps_since_last_advance if s is not None)
    for k in sorted(lookback_counts):
        print(f"    exactly {k} step(s) after an advance_topic: {lookback_counts[k]} occurrences")
    not_preceded = n - sum(preceded_by_recent_advance)
    print(f"  NOT preceded by a recent advance_topic (potential pure filler use): "
          f"{not_preceded} occurrences ({100 * not_preceded / n:.1f}%)")

    print("\n" + "=" * 70)
    print("DIFFICULTY TRAJECTORY SHAPE (first 3 episodes, illustrative)")
    print("=" * 70)
    for ep in range(min(3, N_EPISODES)):
        traj = difficulty_trajectories[ep]
        print(f"  episode {ep}: {traj}")

    # How many times does difficulty change direction (down then up, or up
    # then down) across an episode -- a proxy for "active oscillation" vs
    # "monotonic drift".
    direction_changes = []
    for traj in difficulty_trajectories:
        diffs = np.diff(traj)
        diffs = diffs[diffs != 0]
        if len(diffs) < 2:
            direction_changes.append(0)
            continue
        signs = np.sign(diffs)
        changes = np.sum(signs[1:] != signs[:-1])
        direction_changes.append(int(changes))
    print(f"\n  mean direction changes in difficulty per episode: {np.mean(direction_changes):.1f} "
          f"(0 = monotonic drift, high = active oscillation)")


if __name__ == "__main__":
    main()
