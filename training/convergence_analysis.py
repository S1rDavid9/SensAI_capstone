"""Episodes-to-converge plot for each algorithm's best run.

Method (same one used earlier this session when this was reported as a
table only): 20-episode rolling mean of episode reward (min_periods=1,
matching training/__init__.py's own window for final_mean_reward/
best_mean_reward); "converged episode" = the first episode index at which
the rolling mean first reaches >=90% of its own peak value over the whole
run. This script turns that into an actual plot (previously table-only),
one subplot per algorithm in a single combined figure, with the peak,
90% threshold, and converged-episode point all marked directly on the
curve.

Run with:
    uv run python -m training.convergence_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("logs/comparison")
WINDOW = 20
THRESHOLD_FRACTION = 0.90

RUNS: dict[str, str] = {
    "DQN (dqn_02)": "logs/dqn/dqn_02/run_result.json",
    "PPO (ppo_11)": "logs/pg/ppo/ppo_11/run_result.json",
    "A2C (a2c_07)": "logs/pg/a2c/a2c_07/run_result.json",
    "REINFORCE (reinforce_00)": "logs/pg/reinforce/reinforce_00/run_result.json",
}
COLORS = {"DQN (dqn_02)": "tab:blue", "PPO (ppo_11)": "tab:orange", "A2C (a2c_07)": "tab:green", "REINFORCE (reinforce_00)": "tab:red"}


def analyze_one(label: str, path: str) -> dict:
    data = json.loads(Path(path).read_text())
    rewards = data["episode_rewards"]
    rolling = pd.Series(rewards).rolling(WINDOW, min_periods=1).mean()
    peak_value = float(rolling.max())
    peak_idx = int(rolling.idxmax())
    threshold = THRESHOLD_FRACTION * peak_value
    crossing = rolling[rolling >= threshold]
    converged_idx = int(crossing.index[0])
    converged_value = float(rolling.iloc[converged_idx])
    final_value = float(rolling.iloc[-1])
    return dict(
        label=label, n_episodes=len(rewards), rolling=rolling,
        peak_value=peak_value, peak_idx=peak_idx, threshold=threshold,
        converged_idx=converged_idx, converged_value=converged_value, final_value=final_value,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [analyze_one(label, path) for label, path in RUNS.items()]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, r in zip(axes.flat, results):
        color = COLORS[r["label"]]
        ax.plot(r["rolling"].index, r["rolling"].values, color=color, linewidth=1.5, label=f"{WINDOW}-episode rolling mean")
        ax.axhline(r["peak_value"], color="gray", linestyle=":", linewidth=1, label=f"peak ({r['peak_value']:.2f})")
        ax.axhline(r["threshold"], color="gray", linestyle="--", linewidth=1, label=f"90% of peak ({r['threshold']:.2f})")
        ax.axvline(r["converged_idx"], color="black", linestyle="--", linewidth=1)
        ax.plot(r["converged_idx"], r["converged_value"], "o", color="black", markersize=7, zorder=5,
                 label=f"converged @ episode {r['converged_idx']}")
        ax.set_title(r["label"])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rolling mean reward")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Episodes-to-converge: 20-episode rolling mean reward, with first episode reaching >=90% of peak marked", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / "convergence_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    csv_rows = [
        dict(algo=r["label"], n_episodes=r["n_episodes"], peak_value=round(r["peak_value"], 4),
             peak_episode=r["peak_idx"], threshold_90pct=round(r["threshold"], 4),
             converged_episode=r["converged_idx"], converged_value=round(r["converged_value"], 4),
             final_value=round(r["final_value"], 4))
        for r in results
    ]
    csv_path = OUT_DIR / "convergence_analysis.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    print(pd.DataFrame(csv_rows).to_string(index=False))


if __name__ == "__main__":
    main()
