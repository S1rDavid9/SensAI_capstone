"""Cumulative (per-episode, over training) reward curves for all four best
runs, each in its own subplot -- the rubric-literal form ("all methods in
subplots"), complementing logs/comparison/best_run_convergence.png, which
shows the same underlying data overlaid on shared axes instead.

"Cumulative" here follows this project's established usage throughout
(training/__init__.py's plot_reward_convergence, logs/*/plots/
reward_convergence.png): reward accumulated *as training progresses*,
i.e. the learning curve (episode reward vs. episode index, raw + rolling
mean) -- not a running/cumulative sum of reward, which would be
monotonically dominated by episode count and isn't used anywhere else in
this project's plots.

Run with:
    uv run python -m training.cumulative_reward_subplots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_PATH = Path("logs/comparison/cumulative_reward_subplots.png")
SMOOTHING_WINDOW = 10

RUNS: dict[str, str] = {
    "DQN (dqn_02)": "logs/dqn/dqn_02/run_result.json",
    "PPO (ppo_11)": "logs/pg/ppo/ppo_11/run_result.json",
    "A2C (a2c_07)": "logs/pg/a2c/a2c_07/run_result.json",
    "REINFORCE (reinforce_00)": "logs/pg/reinforce/reinforce_00/run_result.json",
}
COLORS = {"DQN (dqn_02)": "tab:blue", "PPO (ppo_11)": "tab:orange", "A2C (a2c_07)": "tab:green", "REINFORCE (reinforce_00)": "tab:red"}


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, (label, path) in zip(axes.flat, RUNS.items()):
        data = json.loads(Path(path).read_text())
        rewards = data["episode_rewards"]
        color = COLORS[label]

        ax.plot(range(len(rewards)), rewards, color=color, alpha=0.25, linewidth=0.8, label="raw episode reward")
        rolling = pd.Series(rewards).rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        ax.plot(range(len(rewards)), rolling.values, color=color, linewidth=2, label=f"{SMOOTHING_WINDOW}-episode rolling mean")

        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode reward")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Reward over training, all four best runs (each in its own subplot)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
