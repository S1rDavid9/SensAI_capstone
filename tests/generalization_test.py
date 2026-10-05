"""Generalization test for the four best trained AdaptLearn-v1 agents, for
the report's "Generalization" subsection.

Not part of the pytest suite (no assertions -- this produces report tables,
similarly to tests/manual_policy_inspection.py). Run by hand:

    uv run python tests/generalization_test.py

Two parts:

  Part 1 -- held-out seed evaluation. 20 episodes per algorithm on a block
  of seeds (5000-5019) confirmed unused anywhere else in this project:
    - DQN training sweep:            seeds 1000-1011 (BASE_SEED, training/dqn_training.py)
    - PPO/A2C/REINFORCE training:    seeds 2000-2011 (BASE_SEED, training/pg_training.py)
    - policy-inspection (all algos): fixed seeds 0-4  (tests/manual_policy_inspection.py)
  5000-5019 overlaps none of these, so this is a genuine unseen-sample test,
  not a re-evaluation on data any training or prior reporting already used.

  Part 2 -- edge-case stress test on four deliberately extreme starting
  conditions the environment CAN produce under its own random reset() but
  which are rare by chance. AdaptLearnEnv.reset() does not expose a way to
  pin the starting ability/difficulty/engagement_state (see its docstring:
  full per-episode randomization is a deliberate design choice, not
  something to weaken by adding an override hook to the real environment).
  Instead, `_PinnedInitEnv` below is a local, test-only subclass that
  forces those three values after the normal reset() call -- it does not
  touch environment/custom_env.py, and is never used for training.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import (
    ACTION_DECREASE_DIFFICULTY,
    ACTION_INCREASE_DIFFICULTY,
    ACTION_ADD_ENCOURAGEMENT,
    ACTION_ADVANCE_TOPIC,
    ACTION_INTRODUCE_CHALLENGE,
    ACTION_NAMES,
    ACTION_OFFER_HINT,
    ACTION_REPEAT_SIMPLIFIED,
    AdaptLearnEnv,
    EngagementState,
)

# --------------------------------------------------------------------------
# Model loading (mirrors tests/manual_policy_inspection.py)
# --------------------------------------------------------------------------

ALGO_CLASSES: dict[str, type] = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
BEST_MODEL_PATHS: dict[str, str] = {
    "dqn": "models/dqn/dqn_02/model.zip",
    "ppo": "models/pg/ppo/ppo_11/model.zip",
    "a2c": "models/pg/a2c/a2c_07/model.zip",
    "reinforce": "models/pg/reinforce/reinforce_00/model.zip",
}
BEST_RUN_JSON_PATHS: dict[str, str] = {
    "dqn": "logs/dqn/best_run.json",
    "ppo": "logs/pg/ppo/best_run.json",
    "a2c": "logs/pg/a2c/best_run.json",
    "reinforce": "logs/pg/reinforce/best_run.json",
}


def _load_model(algo: str, model_path: str):
    if algo in ALGO_CLASSES:
        return ALGO_CLASSES[algo].load(model_path)
    from training.pg_training import REINFORCE

    return REINFORCE.load(model_path)


def _training_final_mean_reward(algo: str) -> float:
    return json.loads(Path(BEST_RUN_JSON_PATHS[algo]).read_text())["final_mean_reward"]


# --------------------------------------------------------------------------
# Part 1: held-out seed evaluation
# --------------------------------------------------------------------------

HELD_OUT_SEEDS = list(range(5000, 5020))  # 20 seeds, confirmed unused -- see module docstring

# Previously-reported 10-episode policy-inspection results for these exact
# models (5 fixed seeds [0-4] + 5 unseeded episodes each), for direct
# comparison -- see tests/manual_policy_inspection.py output, captured in
# this session's prior turn.
PRIOR_INSPECTION_RESULTS: dict[str, dict[str, Any]] = {
    "dqn": dict(mean_reward=8.83, n_episodes=10, outcomes={"DROPOUT": 2, "SUCCESS": 2, "TRUNCATED": 6}),
    "ppo": dict(mean_reward=6.31, n_episodes=10, outcomes={"DROPOUT": 3, "SUCCESS": 1, "TRUNCATED": 6}),
    "a2c": dict(mean_reward=6.31, n_episodes=10, outcomes={"DROPOUT": 4, "SUCCESS": 1, "TRUNCATED": 5}),
    "reinforce": dict(mean_reward=6.26, n_episodes=10, outcomes={"DROPOUT": 1, "SUCCESS": 0, "TRUNCATED": 9}),
}


def _outcome_of(info: dict) -> str:
    if info.get("dropout"):
        return "DROPOUT"
    if info.get("success"):
        return "SUCCESS"
    return "TRUNCATED"


def run_episode(model, env: AdaptLearnEnv, seed: int | None) -> tuple[float, int, str, list[tuple[str, int]]]:
    """Runs one deterministic episode, returning (total_reward, steps,
    outcome, [(state_before, action), ...]) -- the transition log is used
    by Part 2's "sensible first action" check."""
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    total_reward = 0.0
    steps = 0
    transitions: list[tuple[str, int]] = []
    while not (terminated or truncated):
        before_state = env.engagement_state.name
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        transitions.append((before_state, action))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward, steps, _outcome_of(info), transitions


def part1_held_out_eval() -> dict[str, dict[str, Any]]:
    print("=" * 78)
    print("PART 1: HELD-OUT SEED EVALUATION (seeds 5000-5019, 20 episodes/algorithm)")
    print("=" * 78)

    env = AdaptLearnEnv()
    results: dict[str, dict[str, Any]] = {}

    for algo, model_path in BEST_MODEL_PATHS.items():
        model = _load_model(algo, model_path)
        rewards = []
        outcomes: Counter[str] = Counter()
        for seed in HELD_OUT_SEEDS:
            total_reward, steps, outcome, _ = run_episode(model, env, seed)
            rewards.append(total_reward)
            outcomes[outcome] += 1

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        results[algo] = dict(
            mean_reward=mean_reward,
            std_reward=std_reward,
            outcomes=dict(outcomes),
            rewards=rewards,
        )

        training_reward = _training_final_mean_reward(algo)
        prior = PRIOR_INSPECTION_RESULTS[algo]
        print(f"\n{algo.upper()} (best run, held-out n=20):")
        print(f"  held-out mean reward:        {mean_reward:+.2f} (std {std_reward:.2f})")
        print(f"  held-out outcomes:            {dict(outcomes)}")
        print(f"  training-time final_mean_reward: {training_reward:+.2f}")
        print(f"  prior 10-ep policy-inspection mean: {prior['mean_reward']:+.2f}  outcomes={prior['outcomes']}")

    env.close()
    return results


# --------------------------------------------------------------------------
# Part 2: edge-case stress test
# --------------------------------------------------------------------------


class _PinnedInitEnv(AdaptLearnEnv):
    """Test-only subclass: pins ability/difficulty/engagement_state on
    reset() instead of AdaptLearnEnv's normal full randomization. See this
    module's docstring for why the real environment is deliberately left
    untouched -- this class exists solely for this stress test and is
    never used for training or in any other test."""

    def __init__(self, *, ability: float, difficulty: int, engagement_state: EngagementState, **kwargs) -> None:
        self._pinned_ability = ability
        self._pinned_difficulty = difficulty
        self._pinned_state = engagement_state
        super().__init__(**kwargs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        self.ability = self._pinned_ability
        self.difficulty = self._pinned_difficulty
        self.engagement_state = self._pinned_state
        gap = self.difficulty - self.ability
        self.response_time, self.error_rate = self._simulate_behavioral_signals(gap, self.engagement_state)
        self.session_had_negative = self.engagement_state in (EngagementState.FRUSTRATED, EngagementState.BORED)

        obs = self._build_observation()
        info = self._build_info()
        return obs, info


# (ability, difficulty, engagement_state, sensible first-action set).
# (a)/(b) pin engagement_state=ENGAGED so the extreme *gap* is the only
# thing being tested -- everything currently reads as fine on the visible
# one-hot state, so a genuinely gap-aware policy has to pick up on it via
# response_time/error_rate rather than an explicit FRUSTRATED/BORED label.
# (c)/(d) instead test reaction to an explicit state label, with a neutral
# gap (ability=difficulty=3), so they don't overlap with (a)/(b).
EDGE_CASES: dict[str, dict[str, Any]] = {
    "max_gap_engaged": dict(
        ability=1.0,
        difficulty=5,
        engagement_state=EngagementState.ENGAGED,
        description="Hardest possible mismatch (ability~1, difficulty=5), reads as ENGAGED",
        sensible_actions={ACTION_DECREASE_DIFFICULTY, ACTION_OFFER_HINT, ACTION_REPEAT_SIMPLIFIED},
        sensible_label="decrease_difficulty / offer_hint / repeat_simplified",
    ),
    "min_gap_engaged": dict(
        ability=5.0,
        difficulty=1,
        engagement_state=EngagementState.ENGAGED,
        description="Easiest possible mismatch (ability~5, difficulty=1), reads as ENGAGED",
        sensible_actions={ACTION_INCREASE_DIFFICULTY, ACTION_INTRODUCE_CHALLENGE, ACTION_ADVANCE_TOPIC},
        sensible_label="increase_difficulty / introduce_challenge / advance_topic",
    ),
    "start_frustrated": dict(
        ability=3.0,
        difficulty=3,
        engagement_state=EngagementState.FRUSTRATED,
        description="Starts directly in FRUSTRATED, neutral gap (ability=difficulty=3)",
        sensible_actions={ACTION_ADD_ENCOURAGEMENT, ACTION_DECREASE_DIFFICULTY},
        sensible_label="add_encouragement / decrease_difficulty",
    ),
    "start_confused": dict(
        ability=3.0,
        difficulty=3,
        engagement_state=EngagementState.CONFUSED,
        description="Starts directly in CONFUSED, neutral gap (ability=difficulty=3)",
        sensible_actions={ACTION_OFFER_HINT, ACTION_REPEAT_SIMPLIFIED},
        sensible_label="offer_hint / repeat_simplified (change_task_type excluded -- it mechanically "
        "*increases* CONFUSED risk in this model, see RewardConfig's docstring)",
    ),
}

N_REPS_PER_CONDITION = 5
# Same held-out range as Part 1, offset so no (algo, condition, seed) triple
# repeats a seed already used for Part 1 -- not load-bearing for validity
# (episodes are independent regardless), just keeps every run in this
# script on genuinely fresh seeds.
CONDITION_SEEDS = list(range(5100, 5100 + N_REPS_PER_CONDITION))


def part2_edge_case_stress_test() -> dict[str, dict[str, Any]]:
    print("\n" + "=" * 78)
    print("PART 2: EDGE-CASE STRESS TEST (4 conditions x 5 reps/algorithm)")
    print("=" * 78)

    results: dict[str, dict[str, Any]] = {algo: {} for algo in BEST_MODEL_PATHS}

    for condition_name, cfg in EDGE_CASES.items():
        print(f"\n--- Condition: {condition_name} -- {cfg['description']} ---")
        env = _PinnedInitEnv(
            ability=cfg["ability"], difficulty=cfg["difficulty"], engagement_state=cfg["engagement_state"]
        )

        for algo, model_path in BEST_MODEL_PATHS.items():
            model = _load_model(algo, model_path)
            rewards, outcomes, first_actions, first_action_sensible = [], Counter(), [], []

            for seed in CONDITION_SEEDS:
                total_reward, steps, outcome, transitions = run_episode(model, env, seed)
                rewards.append(total_reward)
                outcomes[outcome] += 1
                first_action = transitions[0][1] if transitions else None
                first_actions.append(ACTION_NAMES[first_action] if first_action is not None else "n/a")
                first_action_sensible.append(first_action in cfg["sensible_actions"] if first_action is not None else False)

            n_sensible = sum(first_action_sensible)
            results[algo][condition_name] = dict(
                mean_reward=float(np.mean(rewards)),
                std_reward=float(np.std(rewards)),
                rewards=rewards,
                outcomes=dict(outcomes),
                first_actions=first_actions,
                n_sensible_first_action=n_sensible,
                n_reps=len(CONDITION_SEEDS),
            )
            print(
                f"  {algo.upper():<10s} mean_reward={np.mean(rewards):+6.2f}  outcomes={dict(outcomes)}  "
                f"first_action_sensible={n_sensible}/{len(CONDITION_SEEDS)}  "
                f"first_actions={first_actions}"
            )

        env.close()

    return results


OUT_PATH = Path("logs/generalization/results.json")


def main() -> None:
    part1_results = part1_held_out_eval()
    part2_results = part2_edge_case_stress_test()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"part1_held_out_eval": part1_results, "part2_edge_case_stress_test": part2_results}, f, indent=2)

    print("\n" + "=" * 78)
    print(f"Done. Raw results written to {OUT_PATH}. See report's Generalization subsection for interpretation.")
    print("=" * 78)

    return part1_results, part2_results


if __name__ == "__main__":
    main()
