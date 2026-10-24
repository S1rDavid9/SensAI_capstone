"""Does the DQN do better than simple teacher rules?

Compares five policies on the same 200 held-out simulated learners (seeds
5000-5199, never used in training), in simulator v2 and v1:

  random             picks any action (the floor)
  state rules        a fixed rule per state, from common teaching practice
  staircase          errors only: many wrong -> easier, nearly all right -> harder
  DQN                the trained policy on its own
  DQN + guardrails   the deployed policy (shielded)

The rule policies were written from teaching practice before being run and
were not tuned on the simulator. They see exactly what the DQN sees: the
9-number observation. Because every policy meets the same learners, the
differences are paired; significance is tested with the Wilcoxon
signed-rank test on per-learner episode reward.

    uv run python -m analysis.baseline_comparison

Results: analysis/results/baseline_comparison.json and .md
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

from environment.custom_env import ACTION_NAMES, AdaptLearnEnv
from pilot.core.states import BehaviouralState
from pilot.policy import DQNPolicy, Shield

SEEDS = range(5000, 5200)
OUT = Path(__file__).parent / "results"
SLOT_TO_STATE = {0: BehaviouralState.STRUGGLING, 1: BehaviouralState.RUSHING,
                 2: BehaviouralState.STUCK, 3: BehaviouralState.ON_TASK}
A = {name: i for i, name in enumerate(ACTION_NAMES)}


def state_of(obs: np.ndarray) -> BehaviouralState:
    return SLOT_TO_STATE[int(np.argmax(obs[:4]))]


def difficulty_of(obs: np.ndarray) -> int:
    return int(round(obs[7] * 4)) + 1


class Policy(ABC):
    name: str

    @abstractmethod
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int: ...


class RandomPolicy(Policy):
    name = "Random"

    def act(self, obs, rng):
        return int(rng.integers(len(ACTION_NAMES)))


class StateRulePolicy(Policy):
    """One fixed response per state, as a teacher might write them down."""
    name = "State rules"
    RULES = {
        BehaviouralState.STRUGGLING: "repeat_simplified",
        BehaviouralState.STUCK: "offer_hint",
        BehaviouralState.RUSHING: "introduce_challenge",
        BehaviouralState.ON_TASK: "advance_topic",
    }

    def act(self, obs, rng):
        return A[self.RULES[state_of(obs)]]


class StaircasePolicy(Policy):
    """Errors only, like an adaptive test: no use of the state at all."""
    name = "Staircase (errors only)"

    def act(self, obs, rng):
        error_rate = float(obs[5])
        if error_rate >= 0.4:
            return A["decrease_difficulty"]
        if error_rate <= 0.15:
            return A["increase_difficulty"]
        return A["advance_topic"]


class DQNOnly(Policy):
    name = "DQN"

    def __init__(self, policy: DQNPolicy) -> None:
        self.policy = policy

    def act(self, obs, rng):
        return int(np.argmax(self.policy.scores(obs)))


class ShieldedDQN(Policy):
    name = "DQN + guardrails"

    def __init__(self, policy: DQNPolicy) -> None:
        self.shield = Shield(policy)

    def act(self, obs, rng):
        return self.shield.decide(obs, state_of(obs), difficulty_of(obs)).action_index


def run(policy: Policy, version: str) -> dict:
    env = AdaptLearnEnv(version=version)
    rewards, outcomes, engaged_share, topics = [], Counter(), [], []
    by_state = {s: Counter() for s in BehaviouralState}
    rng = np.random.default_rng(0)
    for seed in SEEDS:
        obs, info = env.reset(seed=seed)
        done, total, engaged, steps = False, 0.0, 0, 0
        while not done:
            action = policy.act(obs, rng)
            by_state[state_of(obs)][ACTION_NAMES[action]] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            engaged += info["engagement_state"] == "ENGAGED"
            steps += 1
            done = terminated or truncated
        rewards.append(total)
        engaged_share.append(engaged / steps)
        topics.append(info["topics_completed"])
        outcomes["dropout" if info["dropout"] else "success" if info["success"] else "time limit"] += 1
    r = np.array(rewards)
    return {
        "rewards": r.tolist(),
        "mean_reward": round(float(r.mean()), 2),
        "ci95": [round(float(x), 2) for x in stats.t.interval(0.95, len(r) - 1, loc=r.mean(), scale=stats.sem(r))],
        "dropout_rate": round(outcomes["dropout"] / len(r), 3),
        "success_rate": round(outcomes["success"] / len(r), 3),
        "engaged_share": round(float(np.mean(engaged_share)), 3),
        "topics_completed": round(float(np.mean(topics)), 1),
        "top_action_by_state": {s.value: c.most_common(1)[0][0] if c else None for s, c in by_state.items()},
    }


def main() -> None:
    dqn = DQNPolicy("v2")
    dqn.warm_up()
    policies = [RandomPolicy(), StaircasePolicy(), StateRulePolicy(), DQNOnly(dqn), ShieldedDQN(dqn)]
    report = {"seeds": f"{SEEDS.start}-{SEEDS.stop - 1}", "learners": len(SEEDS), "simulators": {}}
    lines = ["# DQN vs simple teacher rules", "",
             f"{len(SEEDS)} held-out simulated learners per policy (seeds {SEEDS.start}-{SEEDS.stop - 1}); "
             "the same learners for every policy. Generated by `analysis/baseline_comparison.py`.", ""]
    for version in ("v2", "v1"):
        results = {p.name: run(p, version) for p in policies}
        ref = np.array(results["DQN + guardrails"]["rewards"])
        for name, r in results.items():
            if name != "DQN + guardrails":
                diff = ref - np.array(r["rewards"])
                r["vs_dqn_guardrails"] = {"mean_difference": round(float(diff.mean()), 2),
                                          "wilcoxon_p": float(f"{stats.wilcoxon(diff).pvalue:.2g}")}
        report["simulators"][version] = {k: {kk: vv for kk, vv in v.items() if kk != "rewards"} for k, v in results.items()}
        lines += [f"## Simulator {version}", "",
                  "| Policy | Mean reward (95% CI) | Dropout | Success | Time engaged | Topics | DQN + guardrails better by (Wilcoxon p) |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            cmp = r.get("vs_dqn_guardrails")
            cmp_text = f"{cmp['mean_difference']:+.2f} (p = {cmp['wilcoxon_p']:.2g})" if cmp else "—"
            lines.append(f"| {name} | {r['mean_reward']:.2f} ({r['ci95'][0]:.2f} to {r['ci95'][1]:.2f}) | "
                         f"{r['dropout_rate']:.0%} | {r['success_rate']:.0%} | {r['engaged_share']:.0%} | "
                         f"{r['topics_completed']:.1f} | {cmp_text} |")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline_comparison.json").write_text(json.dumps(report, indent=2))
    (OUT / "baseline_comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
