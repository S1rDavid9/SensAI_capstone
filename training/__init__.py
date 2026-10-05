"""Shared training-pipeline utilities for the AdaptLearn-v1 algorithm comparison.

This module holds the logic that is identical regardless of which RL
algorithm is being trained: running one Stable-Baselines3-style training run
with full logging, writing a per-sweep summary CSV, and producing the three
comparison plots the assignment brief asks for (reward convergence, variance
across runs, training stability). `training/dqn_training.py` and
`training/pg_training.py` both import from here rather than duplicating this
logic, since the required project layout has no separate module for it.

Terminology: a "run" is one full training of one algorithm under one fixed
hyperparameter configuration and seed. A "sweep" is the set of runs for one
algorithm across its hyperparameter grid (>= 10 runs, per the brief).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: training runs should never require a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import AdaptLearnEnv


@dataclass
class RunResult:
    """Everything logged for one training run, sufficient on its own to
    reproduce the run (algo + hyperparams + seed) and to analyze it
    (reward/length curves, timing) without re-running training."""

    run_id: str
    algo: str
    seed: int
    hyperparams: dict[str, Any]
    total_timesteps: int
    training_time_seconds: float
    episode_rewards: list[float]
    episode_lengths: list[int]
    final_mean_reward: float
    best_mean_reward: float
    model_path: str


def make_training_env() -> Monitor:
    """A fresh, Monitor-wrapped AdaptLearnEnv for one training run.

    Monitor records per-episode (reward, length, wall-clock time) to a CSV
    as training proceeds -- this is the source of the reward curves used
    for every plot below, and is the standard SB3 mechanism for this,
    rather than a hand-rolled callback.
    """
    return Monitor(AdaptLearnEnv())


def run_single_training(
    *,
    algo_class: type,
    policy: str,
    run_id: str,
    algo_name: str,
    seed: int,
    hyperparams: dict[str, Any],
    total_timesteps: int,
    models_dir: Path,
    logs_dir: Path,
    sb3_kwargs: dict[str, Any] | None = None,
    tensorboard_log: str | None = None,
) -> RunResult:
    """Train one model for one (algorithm, hyperparameter, seed) run, saving
    the model and a full log of the run to disk, and returning a RunResult.

    `hyperparams` is the human-readable configuration (as it should appear
    in the report/reproducibility log); `sb3_kwargs` is the possibly-
    different set of keyword arguments actually passed to the SB3
    constructor (e.g. `net_arch` gets nested inside `policy_kwargs`).
    Keeping these separate means the logged hyperparameters stay flat and
    readable even when the SB3 API requires nesting.

    `tensorboard_log` is optional and defaults to None (no behavior change
    for existing callers that don't pass it -- the SB3 constructor call
    below is then byte-for-byte identical to before this parameter existed).
    When given a path, SB3's own scalar logging (e.g. train/loss,
    train/entropy_loss) is enabled for this run -- see
    tests/stability_analysis.py, the only current caller that passes this.
    """
    run_model_dir = models_dir / run_id
    run_log_dir = logs_dir / run_id
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    monitor_path = run_log_dir / "monitor.csv"
    env = Monitor(AdaptLearnEnv(), filename=str(monitor_path))

    extra_kwargs = {} if tensorboard_log is None else {"tensorboard_log": tensorboard_log}
    model = algo_class(policy, env, seed=seed, verbose=0, **extra_kwargs, **(sb3_kwargs or hyperparams))

    # tb_log_name is only meaningful (and only SB3-algo-compatible -- REINFORCE's
    # .learn() takes no such argument) when tensorboard_log was actually
    # requested; omitted entirely otherwise, so this is a no-op for every
    # existing caller (none of which pass tensorboard_log).
    learn_kwargs = {} if tensorboard_log is None else {"tb_log_name": run_id}
    start = time.time()
    model.learn(total_timesteps=total_timesteps, **learn_kwargs)
    training_time_seconds = time.time() - start

    model_path = run_model_dir / "model.zip"
    model.save(str(model_path))
    env.close()

    episode_rewards, episode_lengths = _read_monitor_csv(monitor_path)

    window = min(20, len(episode_rewards)) or 1
    final_mean_reward = float(np.mean(episode_rewards[-window:])) if episode_rewards else float("nan")
    if len(episode_rewards) >= window:
        best_mean_reward = float(pd.Series(episode_rewards).rolling(window).mean().max())
    else:
        best_mean_reward = final_mean_reward

    result = RunResult(
        run_id=run_id,
        algo=algo_name,
        seed=seed,
        hyperparams=hyperparams,
        total_timesteps=total_timesteps,
        training_time_seconds=training_time_seconds,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        final_mean_reward=final_mean_reward,
        best_mean_reward=best_mean_reward,
        model_path=str(model_path),
    )

    with open(run_log_dir / "run_result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


def _read_monitor_csv(monitor_path: Path) -> tuple[list[float], list[int]]:
    """SB3's Monitor CSV has a one-line JSON comment header, then columns
    r (episode reward), l (episode length), t (wall-clock time)."""
    if not monitor_path.exists():
        return [], []
    df = pd.read_csv(monitor_path, skiprows=1)
    if df.empty:
        return [], []
    return df["r"].tolist(), df["l"].astype(int).tolist()


def write_summary_csv(results: list[RunResult], out_path: Path) -> pd.DataFrame:
    """One row per run: identifying info, timing, outcome, and every
    hyperparameter (flattened, prefixed `hp_`) -- the single artifact a
    report needs to cite for "every run was logged, not just the best"."""
    rows = []
    for r in results:
        row = {
            "run_id": r.run_id,
            "algo": r.algo,
            "seed": r.seed,
            "total_timesteps": r.total_timesteps,
            "training_time_seconds": round(r.training_time_seconds, 2),
            "n_episodes": len(r.episode_rewards),
            "final_mean_reward": round(r.final_mean_reward, 4) if r.episode_rewards else None,
            "best_mean_reward": round(r.best_mean_reward, 4) if r.episode_rewards else None,
            "model_path": r.model_path,
        }
        row.update({f"hp_{k}": v for k, v in r.hyperparams.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def write_best_run_marker(results: list[RunResult], out_path: Path) -> RunResult:
    """Records which run had the best `final_mean_reward`, so `main.py` can
    automatically load it for the live demo without re-scanning every run."""
    best = max(results, key=lambda r: r.final_mean_reward)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "run_id": best.run_id,
                "algo": best.algo,
                "seed": best.seed,
                "model_path": best.model_path,
                "final_mean_reward": best.final_mean_reward,
                "hyperparams": best.hyperparams,
            },
            f,
            indent=2,
        )
    return best


# --------------------------------------------------------------------------
# Comparison plots
# --------------------------------------------------------------------------


def plot_reward_convergence(
    results: list[RunResult], out_path: Path, algo_label: str, smoothing_window: int = 10
) -> None:
    """Every run's smoothed reward curve (faint), overlaid with the mean
    curve across all runs (bold) -- shows both individual convergence
    behavior and the sweep's overall trend in one figure."""
    results = [r for r in results if r.episode_rewards]
    if not results:
        return

    plt.figure(figsize=(9, 5.5))
    for r in results:
        series = pd.Series(r.episode_rewards).rolling(smoothing_window, min_periods=1).mean()
        plt.plot(series.values, color="tab:blue", alpha=0.25, linewidth=1)

    min_len = min(len(r.episode_rewards) for r in results)
    if min_len > 0:
        stacked = np.array([r.episode_rewards[:min_len] for r in results])
        mean_curve = pd.Series(stacked.mean(axis=0)).rolling(smoothing_window, min_periods=1).mean()
        plt.plot(
            mean_curve.values,
            color="tab:blue",
            linewidth=2.5,
            label=f"{algo_label} mean across {len(results)} runs",
        )

    plt.xlabel("Episode")
    plt.ylabel(f"Episode reward ({smoothing_window}-episode rolling mean)")
    plt.title(f"{algo_label}: reward convergence across hyperparameter sweep")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_variance_across_runs(results: list[RunResult], out_path: Path, algo_label: str) -> None:
    """Mean reward curve across runs (aligned by episode index, truncated to
    the shortest run) with a +/-1 std-dev band -- shows how much hyperparameter
    choice affects outcome, independent of any single run's trend."""
    results = [r for r in results if r.episode_rewards]
    if not results:
        return

    min_len = min(len(r.episode_rewards) for r in results)
    if min_len == 0:
        return
    stacked = np.array([r.episode_rewards[:min_len] for r in results])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    x = np.arange(min_len)

    plt.figure(figsize=(9, 5.5))
    plt.plot(x, mean, color="tab:orange", linewidth=2)
    plt.fill_between(x, mean - std, mean + std, color="tab:orange", alpha=0.25, label="+/-1 std across runs")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title(f"{algo_label}: variance across {len(results)} runs")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_stability(
    results: list[RunResult], out_path: Path, algo_label: str, tail_window: int = 20
) -> None:
    """Per-run standard deviation of reward over the final `tail_window`
    episodes, sorted ascending -- a direct measure of how stable each
    hyperparameter configuration's *converged* behavior is (a run can reach
    a high mean reward while still oscillating wildly step to step; this
    plot is what distinguishes that from a run that has actually settled)."""
    results = [r for r in results if r.episode_rewards]
    if not results:
        return

    run_ids = [r.run_id for r in results]
    tail_std = [float(np.std(r.episode_rewards[-tail_window:])) for r in results]
    order = np.argsort(tail_std)
    sorted_ids = [run_ids[i] for i in order]
    sorted_std = [tail_std[i] for i in order]

    plt.figure(figsize=(10, 5.5))
    plt.bar(range(len(sorted_ids)), sorted_std, color="tab:green")
    plt.xticks(range(len(sorted_ids)), sorted_ids, rotation=60, ha="right", fontsize=7)
    plt.ylabel(f"Std dev of reward, last {tail_window} episodes\n(lower = more stable)")
    plt.title(f"{algo_label}: training stability across hyperparameter sweep")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_sweep_plots(results: list[RunResult], plots_dir: Path, algo_label: str) -> None:
    """Convenience wrapper generating all three required comparison plots
    for one algorithm's sweep."""
    plot_reward_convergence(results, plots_dir / "reward_convergence.png", algo_label)
    plot_variance_across_runs(results, plots_dir / "variance_across_runs.png", algo_label)
    plot_training_stability(results, plots_dir / "training_stability.png", algo_label)
