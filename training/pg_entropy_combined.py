"""Combined view of the three policy-gradient-family entropy curves (PPO,
A2C, REINFORCE) in one figure -- complementing the three separate,
per-algorithm PNGs already in logs/stability/plots/.

Subplots, not shared axes, deliberately: PPO/A2C log train/entropy_loss
(SB3's sign convention: entropy_loss = -mean(entropy), so more negative =
higher actual entropy), while REINFORCE's custom entropy_history is raw,
positive entropy (nats) -- plotting these on one shared y-axis would be
misleading. Each subplot's ylabel states exactly which convention it uses.

Run with:
    uv run python -m training.pg_entropy_combined
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_PATH = Path("logs/stability/plots/pg_entropy_combined.png")

PANELS = [
    dict(
        label="PPO (ppo_11)",
        path="logs/stability/ppo_entropy.csv",
        color="tab:orange",
        ylabel="train/entropy_loss\n(= -mean entropy; more negative = higher entropy)",
    ),
    dict(
        label="A2C (a2c_07)",
        path="logs/stability/a2c_entropy.csv",
        color="tab:green",
        ylabel="train/entropy_loss\n(= -mean entropy; more negative = higher entropy)",
    ),
    dict(
        label="REINFORCE (reinforce_00)",
        path="logs/stability/reinforce_entropy.csv",
        color="tab:red",
        ylabel="policy entropy (nats)\n(raw, not negated)",
    ),
]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, panel in zip(axes, PANELS):
        df = pd.read_csv(panel["path"])
        x_col = "step" if "step" in df.columns else "batch_index"
        y_col = "value" if "value" in df.columns else "entropy"
        ax.plot(df[x_col], df[y_col], color=panel["color"])
        ax.set_title(panel["label"])
        ax.set_xlabel("Training step" if x_col == "step" else "Batch index")
        ax.set_ylabel(panel["ylabel"], fontsize=9)

    fig.suptitle("Policy-gradient family: entropy over training (PPO, A2C, REINFORCE)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
