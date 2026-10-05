"""Manual, human-in-the-loop sanity check of a trained AdaptLearn-v1 policy.

This is a diagnostic script, not an automated test -- there are no
pass/fail assertions. Its purpose is to make a trained agent's actual
behavior legible enough to judge, by eye, whether it has learned a
plausible state-contingent tutoring policy or found a way to farm reward
without behaving like a sensible tutor (e.g. spamming one action
regardless of the learner's state). Run by hand and read the output:

    uv run python tests/manual_policy_inspection.py
    uv run python tests/manual_policy_inspection.py --model models/dqn/dqn_02/model.zip --algo dqn
    uv run python tests/manual_policy_inspection.py --model models/pg/ppo/ppo_11/model.zip --algo ppo
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import ACTION_NAMES, N_ACTIONS, AdaptLearnEnv, EngagementState

DEFAULT_MODEL_PATH = Path("models/dqn/dqn_02/model.zip")
FIXED_SEEDS = [0, 1, 2, 3, 4]
N_UNSEEDED_EPISODES = 5

ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "a2c": A2C}


def _load_model(algo: str, model_path: Path):
    if algo in ALGO_CLASSES:
        return ALGO_CLASSES[algo].load(str(model_path))
    if algo == "reinforce":
        from training.pg_training import REINFORCE

        return REINFORCE.load(str(model_path))
    raise ValueError(f"Unknown --algo {algo!r}")

# Overall-action-share threshold above which we flag possible reward
# farming regardless of engagement state (per the brief's ">40-50%" guidance).
DOMINANCE_FLAG_THRESHOLD = 0.40


def run_episode(model, env: AdaptLearnEnv, seed: int | None, episode_label: str, transition_log: list[tuple[str, int]]):
    obs, info = env.reset(seed=seed)
    print(f"\n=== Episode {episode_label} (seed={seed}) ===")
    action_counts: Counter[int] = Counter()
    total_reward = 0.0
    step = 0
    terminated = truncated = False
    info = {}

    while not (terminated or truncated):
        before_state = env.engagement_state
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        after_state = env.engagement_state
        step += 1
        total_reward += reward
        action_counts[action] += 1
        transition_log.append((before_state.name, action))

        print(
            f"  t={step:02d}  before={before_state.name:<10s}  "
            f"action={ACTION_NAMES[action]:<20s}  after={after_state.name:<10s}  "
            f"reward={reward:+.2f}"
        )

    outcome = "DROPOUT" if info.get("dropout") else "SUCCESS" if info.get("success") else "TRUNCATED (max steps)"
    print(f"  --- episode end after {step} steps: {outcome} | total_reward={total_reward:+.2f}")
    print("  action frequency this episode:")
    for a in range(N_ACTIONS):
        count = action_counts.get(a, 0)
        pct = 100 * count / step if step else 0.0
        print(f"    {ACTION_NAMES[a]:<20s} {count:3d}  ({pct:5.1f}%)")

    return dict(total_reward=total_reward, steps=step, outcome=outcome, action_counts=action_counts)


def print_state_action_crosstab(transition_log: list[tuple[str, int]]) -> None:
    print("\n" + "=" * 70)
    print("CROSS-TAB: action chosen, broken down by engagement state BEFORE the action")
    print("=" * 70)

    by_state: dict[str, Counter[int]] = {s.name: Counter() for s in EngagementState}
    for state_name, action in transition_log:
        by_state[state_name][action] += 1

    for state in EngagementState:
        counter = by_state[state.name]
        total = sum(counter.values())
        print(f"\nWhen learner was {state.name} ({total} occurrences):")
        if total == 0:
            print("    (never observed before an action in this run)")
            continue
        ranked = counter.most_common()
        top = ranked[:3]
        top_count = sum(c for _, c in top)
        parts = [f"{ACTION_NAMES[a]} {100 * c / total:.0f}%" for a, c in top]
        other_pct = 100 * (total - top_count) / total
        if other_pct > 0.05:
            parts.append(f"other {other_pct:.0f}%")
        print("    " + ", ".join(parts))
        for a, c in ranked:
            print(f"      {ACTION_NAMES[a]:<20s} {c:3d}  ({100 * c / total:5.1f}%)")


def print_overall_action_frequency(transition_log: list[tuple[str, int]]) -> None:
    print("\n" + "=" * 70)
    print("OVERALL ACTION FREQUENCY (across all states, all episodes)")
    print("=" * 70)

    total = len(transition_log)
    counts = Counter(action for _, action in transition_log)
    flagged = []
    for a in range(N_ACTIONS):
        count = counts.get(a, 0)
        share = count / total if total else 0.0
        marker = ""
        if share >= DOMINANCE_FLAG_THRESHOLD:
            marker = "  <-- FLAG: used in >= {:.0f}% of all steps".format(DOMINANCE_FLAG_THRESHOLD * 100)
            flagged.append((ACTION_NAMES[a], share))
        print(f"  {ACTION_NAMES[a]:<20s} {count:4d}  ({100 * share:5.1f}%){marker}")

    print()
    if flagged:
        for name, share in flagged:
            print(
                f"FLAGGED: '{name}' accounts for {100 * share:.1f}% of all actions taken across "
                f"every episode/state, at or above the {DOMINANCE_FLAG_THRESHOLD * 100:.0f}% dominance threshold. "
                f"This is consistent with a reward shortcut rather than state-contingent behavior."
            )
    else:
        print(f"No single action reached the {DOMINANCE_FLAG_THRESHOLD * 100:.0f}% overall-dominance threshold.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually inspect a trained AdaptLearn-v1 policy's behavior.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default="dqn")
    args = parser.parse_args()

    print(f"Loading {args.algo.upper()} model from {args.model} ...")
    model = _load_model(args.algo, args.model)

    env = AdaptLearnEnv()
    transition_log: list[tuple[str, int]] = []
    episode_summaries = []

    for seed in FIXED_SEEDS:
        summary = run_episode(model, env, seed=seed, episode_label=f"fixed-seed-{seed}", transition_log=transition_log)
        episode_summaries.append(summary)

    for i in range(N_UNSEEDED_EPISODES):
        summary = run_episode(model, env, seed=None, episode_label=f"unseeded-{i}", transition_log=transition_log)
        episode_summaries.append(summary)

    env.close()

    print("\n" + "=" * 70)
    print("PER-EPISODE OUTCOME SUMMARY")
    print("=" * 70)
    outcome_counts = Counter(s["outcome"] for s in episode_summaries)
    for outcome, count in outcome_counts.items():
        print(f"  {outcome:<28s} {count} / {len(episode_summaries)} episodes")
    mean_reward = sum(s["total_reward"] for s in episode_summaries) / len(episode_summaries)
    print(f"  mean total reward across {len(episode_summaries)} episodes: {mean_reward:+.2f}")

    print_state_action_crosstab(transition_log)
    print_overall_action_frequency(transition_log)


if __name__ == "__main__":
    main()
