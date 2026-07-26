"""Cross-algorithm comparison for AdaptLearn-v1: DQN vs. PPO vs. A2C vs.
REINFORCE, built from each algorithm's already-completed 12-run sweep.

This is deliberately a post-hoc analysis script, not a training script -- it
reads the artifacts `training/dqn_training.py` and `training/pg_training.py`
already wrote (`runs_summary.csv`, `best_run.json`, and each run's
`run_result.json`, which carries the full per-episode reward curve) rather
than retraining anything. All four sweeps used the same 20,000-timestep
budget per run (see DEFAULT_TOTAL_TIMESTEPS in both training modules), which
is what makes a direct comparison meaningful rather than an apples-to-oranges
one; this script asserts that rather than silently assuming it.

Run with:
    uv run python -m training.compare_algorithms
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, matching training/__init__.py's convention
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (display label, summary-csv dir, per-run log dir -- same in every case here
# since runs_summary.csv/best_run.json live alongside the per-run subdirs)
ALGO_LOG_DIRS: dict[str, Path] = {
    "DQN": Path("logs/dqn"),
    "PPO": Path("logs/pg/ppo"),
    "A2C": Path("logs/pg/a2c"),
    "REINFORCE": Path("logs/pg/reinforce"),
}

OUT_DIR = Path("logs/comparison")
ALGO_COLORS: dict[str, str] = {
    "DQN": "tab:blue",
    "PPO": "tab:orange",
    "A2C": "tab:green",
    "REINFORCE": "tab:red",
}
SMOOTHING_WINDOW = 10


@dataclass
class AlgoSweep:
    algo: str
    summary: pd.DataFrame  # one row per run, from runs_summary.csv
    best_run_id: str
    best_episode_rewards: list[float]  # full curve for the best run
    total_timesteps: int


def _load_algo_sweep(algo: str, log_dir: Path) -> AlgoSweep:
    summary_path = log_dir / "runs_summary.csv"
    best_run_path = log_dir / "best_run.json"
    if not summary_path.exists() or not best_run_path.exists():
        raise FileNotFoundError(
            f"{algo}: missing {summary_path} or {best_run_path} -- run its sweep "
            f"(training/dqn_training.py or training/pg_training.py) before comparing."
        )

    summary = pd.read_csv(summary_path)
    best_run = json.loads(best_run_path.read_text())
    best_run_id = best_run["run_id"]

    run_result_path = log_dir / best_run_id / "run_result.json"
    run_result = json.loads(run_result_path.read_text())

    return AlgoSweep(
        algo=algo,
        summary=summary,
        best_run_id=best_run_id,
        best_episode_rewards=run_result["episode_rewards"],
        total_timesteps=run_result["total_timesteps"],
    )


def _assert_comparable_budgets(sweeps: list[AlgoSweep]) -> None:
    """The comparison below only means something if every algorithm was
    given the same training budget -- fail loudly rather than silently plot
    a mismatched comparison if that ever stops being true."""
    budgets = {s.algo: s.total_timesteps for s in sweeps}
    unique = set(budgets.values())
    if len(unique) != 1:
        raise ValueError(f"Algorithms were trained with different timestep budgets, not directly comparable: {budgets}")


def plot_best_run_convergence(sweeps: list[AlgoSweep], out_path: Path) -> None:
    """Each algorithm's single best run, smoothed reward curve overlaid --
    the direct "how fast/how well did each algorithm's best configuration
    learn" comparison."""
    plt.figure(figsize=(9.5, 6))
    for s in sweeps:
        if not s.best_episode_rewards:
            continue
        series = pd.Series(s.best_episode_rewards).rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        plt.plot(
            series.values,
            color=ALGO_COLORS[s.algo],
            linewidth=2,
            label=f"{s.algo} (best run {s.best_run_id})",
        )
    plt.xlabel("Episode")
    plt.ylabel(f"Episode reward ({SMOOTHING_WINDOW}-episode rolling mean)")
    plt.title("Best run per algorithm: reward convergence")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sweep_distribution(sweeps: list[AlgoSweep], out_path: Path) -> None:
    """Box plot of final_mean_reward across each algorithm's full 12-run
    sweep -- shows hyperparameter sensitivity/robustness, not just the best
    case each algorithm happened to land on."""
    plt.figure(figsize=(8, 5.5))
    data = [s.summary["final_mean_reward"].dropna().values for s in sweeps]
    labels = [s.algo for s in sweeps]
    bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, s in zip(bp["boxes"], sweeps):
        patch.set_facecolor(ALGO_COLORS[s.algo])
        patch.set_alpha(0.5)
    plt.ylabel("final_mean_reward (per run, 12 runs/algorithm)")
    plt.title("Sweep-wide reward distribution across hyperparameter grid")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_best_run_bar(sweeps: list[AlgoSweep], out_path: Path) -> None:
    """At-a-glance bar chart of each algorithm's best final_mean_reward."""
    plt.figure(figsize=(7.5, 5.5))
    algos = [s.algo for s in sweeps]
    best_vals = [s.summary["final_mean_reward"].max() for s in sweeps]
    colors = [ALGO_COLORS[a] for a in algos]
    bars = plt.bar(algos, best_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, best_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Best final_mean_reward across the 12-run sweep")
    plt.title("Best-configuration reward by algorithm")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_time(sweeps: list[AlgoSweep], out_path: Path) -> None:
    """Mean wall-clock training time per run, by algorithm -- same timestep
    budget for all four (asserted above), so differences here reflect
    per-step compute cost (e.g. PPO/A2C's extra rollout/GAE bookkeeping vs.
    the from-scratch REINFORCE loop), not a difference in how much learning
    signal each algorithm got."""
    plt.figure(figsize=(7.5, 5.5))
    algos = [s.algo for s in sweeps]
    means = [s.summary["training_time_seconds"].mean() for s in sweeps]
    colors = [ALGO_COLORS[a] for a in algos]
    bars = plt.bar(algos, means, color=colors, alpha=0.85)
    for bar, val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Mean wall-clock training time per run (s)")
    plt.title("Training cost by algorithm (same 20,000-timestep budget)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_comparison_table(sweeps: list[AlgoSweep], out_path: Path) -> pd.DataFrame:
    rows = []
    for s in sweeps:
        fmr = s.summary["final_mean_reward"].dropna()
        rows.append(
            {
                "algo": s.algo,
                "best_run_id": s.best_run_id,
                "best_final_mean_reward": round(fmr.max(), 4),
                "sweep_mean_final_mean_reward": round(fmr.mean(), 4),
                "sweep_std_final_mean_reward": round(fmr.std(), 4),
                "sweep_min_final_mean_reward": round(fmr.min(), 4),
                "total_timesteps_per_run": s.total_timesteps,
                "mean_training_time_seconds": round(s.summary["training_time_seconds"].mean(), 2),
            }
        )
    df = pd.DataFrame(rows).sort_values("best_final_mean_reward", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def write_comparison_summary_md(df: pd.DataFrame, out_path: Path) -> None:
    """A short, factual markdown summary -- numbers and their direct
    reading only, not report-level interpretation (that belongs in the
    report itself, not a generated artifact that will be re-run and
    overwritten)."""
    ranked = df.sort_values("best_final_mean_reward", ascending=False)
    lines = [
        "# Cross-algorithm comparison summary",
        "",
        "Generated by `training/compare_algorithms.py` from each algorithm's "
        "12-run hyperparameter sweep (same 20,000-timestep budget per run).",
        "",
        "## Ranking by best run's final_mean_reward",
        "",
        "| Rank | Algorithm | Best run | Best final_mean_reward | Sweep mean +/- std | Mean training time |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(ranked.itertuples(), start=1):
        lines.append(
            f"| {i} | {row.algo} | {row.best_run_id} | {row.best_final_mean_reward:.2f} | "
            f"{row.sweep_mean_final_mean_reward:.2f} +/- {row.sweep_std_final_mean_reward:.2f} | "
            f"{row.mean_training_time_seconds:.1f}s |"
        )
    lines += [
        "",
        "## Plots",
        "",
        "- `best_run_convergence.png` -- reward curve of each algorithm's best run, overlaid",
        "- `sweep_reward_distribution.png` -- box plot of final_mean_reward across all 12 runs/algorithm",
        "- `best_run_bar.png` -- best final_mean_reward per algorithm, bar chart",
        "- `training_time.png` -- mean wall-clock training time per run, by algorithm",
        "",
        "Interpretation and discussion belong in the report, not here -- this file is "
        "regenerated by re-running the script and will be overwritten.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    sweeps = [_load_algo_sweep(algo, log_dir) for algo, log_dir in ALGO_LOG_DIRS.items()]
    _assert_comparable_budgets(sweeps)

    plot_best_run_convergence(sweeps, OUT_DIR / "best_run_convergence.png")
    plot_sweep_distribution(sweeps, OUT_DIR / "sweep_reward_distribution.png")
    plot_best_run_bar(sweeps, OUT_DIR / "best_run_bar.png")
    plot_training_time(sweeps, OUT_DIR / "training_time.png")

    table = write_comparison_table(sweeps, OUT_DIR / "comparison_table.csv")
    write_comparison_summary_md(table, OUT_DIR / "comparison_summary.md")

    print(f"Wrote comparison plots + table + summary to {OUT_DIR}/")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
