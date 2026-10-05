"""Visualizes the generalization test results (tests/generalization_test.py)
as grouped bar charts -- previously reported as tables only.

Two panels in one saved figure:
  Top:    training-time final_mean_reward vs. held-out-seed mean reward
          (n=20, seeds 5000-5019), per algorithm. Error bars = std of the
          last 20 training episodes (training-time) / std across the 20
          held-out episodes (held-out) -- both computed the same way
          final_mean_reward's own window already is.
  Bottom: the four edge-case conditions (5 reps each), per algorithm.
          Error bars = std across the 5 reps.

Run with:
    uv run python -m training.generalization_plot
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = Path("logs/generalization/generalization_comparison.png")
GEN_RESULTS_PATH = Path("logs/generalization/results.json")

ALGOS = ["dqn", "ppo", "a2c", "reinforce"]
ALGO_LABELS = {"dqn": "DQN", "ppo": "PPO", "a2c": "A2C", "reinforce": "REINFORCE"}
ALGO_COLORS = {"dqn": "tab:blue", "ppo": "tab:orange", "a2c": "tab:green", "reinforce": "tab:red"}

BEST_RUN_JSON_PATHS = {
    "dqn": "logs/dqn/best_run.json",
    "ppo": "logs/pg/ppo/best_run.json",
    "a2c": "logs/pg/a2c/best_run.json",
    "reinforce": "logs/pg/reinforce/best_run.json",
}
RUN_RESULT_JSON_PATHS = {
    "dqn": "logs/dqn/dqn_02/run_result.json",
    "ppo": "logs/pg/ppo/ppo_11/run_result.json",
    "a2c": "logs/pg/a2c/a2c_07/run_result.json",
    "reinforce": "logs/pg/reinforce/reinforce_00/run_result.json",
}

EDGE_CONDITIONS = ["max_gap_engaged", "min_gap_engaged", "start_frustrated", "start_confused"]
EDGE_LABELS = {
    "max_gap_engaged": "max gap\n(ENGAGED)",
    "min_gap_engaged": "min gap\n(ENGAGED)",
    "start_frustrated": "start\nFRUSTRATED",
    "start_confused": "start\nCONFUSED",
}


def _training_time_mean_std(algo: str) -> tuple[float, float]:
    """final_mean_reward is already the mean of the last 20 episodes
    (training/__init__.py); std of that same window for a comparable
    error bar."""
    data = json.loads(Path(RUN_RESULT_JSON_PATHS[algo]).read_text())
    rewards = data["episode_rewards"]
    window = rewards[-min(20, len(rewards)):]
    return float(np.mean(window)), float(np.std(window))


def main() -> None:
    gen = json.loads(GEN_RESULTS_PATH.read_text())

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 11))

    # --- Top panel: training-time vs. held-out ---
    x = np.arange(len(ALGOS))
    width = 0.35
    train_means, train_stds, held_means, held_stds = [], [], [], []
    for algo in ALGOS:
        m, s = _training_time_mean_std(algo)
        train_means.append(m)
        train_stds.append(s)
        held_means.append(gen["part1_held_out_eval"][algo]["mean_reward"])
        held_stds.append(gen["part1_held_out_eval"][algo]["std_reward"])

    ax_top.bar(x - width / 2, train_means, width, yerr=train_stds, capsize=4,
               label="Training-time (last 20 episodes)", color="tab:gray", alpha=0.85)
    ax_top.bar(x + width / 2, held_means, width, yerr=held_stds, capsize=4,
               label="Held-out seeds (n=20, seeds 5000-5019)", color="tab:purple", alpha=0.85)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([ALGO_LABELS[a] for a in ALGOS])
    ax_top.set_ylabel("Mean episode reward")
    ax_top.set_title("Training-time performance vs. held-out-seed generalization")
    ax_top.axhline(0, color="black", linewidth=0.8)
    ax_top.legend()

    # --- Bottom panel: edge-case conditions ---
    n_algos = len(ALGOS)
    n_conditions = len(EDGE_CONDITIONS)
    group_width = 0.8
    bar_width = group_width / n_algos
    x_conditions = np.arange(n_conditions)

    for i, algo in enumerate(ALGOS):
        means, stds = [], []
        for cond in EDGE_CONDITIONS:
            r = gen["part2_edge_case_stress_test"][algo][cond]
            means.append(r["mean_reward"])
            stds.append(r["std_reward"])
        offset = (i - (n_algos - 1) / 2) * bar_width
        ax_bottom.bar(x_conditions + offset, means, bar_width, yerr=stds, capsize=3,
                      label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], alpha=0.85)

    ax_bottom.set_xticks(x_conditions)
    ax_bottom.set_xticklabels([EDGE_LABELS[c] for c in EDGE_CONDITIONS])
    ax_bottom.set_ylabel("Mean episode reward")
    ax_bottom.set_title("Edge-case stress test (5 reps/condition/algorithm)")
    ax_bottom.axhline(0, color="black", linewidth=0.8)
    ax_bottom.legend()

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
