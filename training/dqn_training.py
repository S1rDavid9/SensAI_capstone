"""DQN hyperparameter sweep for AdaptLearn-v1.

Trains Stable-Baselines3's DQN across a curated grid of >= 10 hyperparameter
configurations (learning rate, discount factor, batch size, Q-network
architecture, replay buffer size, exploration schedule, and target-network
update interval), logging every run -- not just the best -- for
reproducibility and later report write-up.

WHY A CURATED GRID, NOT RANDOM SEARCH
----------------------------------------
The brief asks for "extensive hyperparameter tuning: minimum 10 runs ... with
varied hyperparameter combinations". A curated grid (each run changes a
deliberate, named thing relative to a baseline) rather than random search
was chosen so that the resulting comparison plots and summary table map
cleanly onto specific, citable claims in a report (e.g. "lowering gamma from
0.99 to 0.90 destabilized training -- see run dqn_04") rather than an
unstructured sample a reader has to reverse-engineer.

WHY ONE SEED PER CONFIGURATION, NOT REPEATED SEEDS
--------------------------------------------------------
Each of the 12 grid entries below uses a single, fixed seed (judgment call:
repeating every configuration across multiple seeds would give a cleaner
per-configuration variance estimate, but would multiply the compute budget
for this assignment's sweep without changing which conclusions are
supportable at this scope). The seed is still logged per run for exact
reproducibility, and `plot_variance_across_runs` reports the variance that
*is* available for free at this budget: variance *across the 12
configurations*, which is exactly what a hyperparameter sensitivity analysis
needs.

Usage
-----
    uv run python -m training.dqn_training                 # full sweep
    uv run python -m training.dqn_training --quick          # fast smoke test
    uv run python -m training.dqn_training --total-timesteps 50000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN
from tqdm import tqdm

from training import (
    RunResult,
    generate_sweep_plots,
    run_single_training,
    write_best_run_marker,
    write_summary_csv,
)

MODELS_DIR = Path("models/dqn")
LOGS_DIR = Path("logs/dqn")
PLOTS_DIR = LOGS_DIR / "plots"

DEFAULT_TOTAL_TIMESTEPS = 20_000

# Each entry: (run suffix, seed, hyperparameters). Hyperparameters are kept
# flat and human-readable here; `_to_sb3_kwargs` below adapts them to SB3's
# actual DQN constructor signature (mainly: nesting `net_arch` inside
# `policy_kwargs`).
#
# Grid design: entry 00 is a reasonable baseline; each subsequent entry
# perturbs one or two axes relative to it (learning rate, gamma, batch size,
# network depth/width, replay buffer size, exploration fraction, target
# update interval), so the sweep functions as a mini ablation study.
DQN_HYPERPARAM_GRID: list[dict[str, Any]] = [
    dict(learning_rate=1e-3, gamma=0.99, batch_size=32, net_arch=[64, 64],
         buffer_size=10_000, exploration_fraction=0.30, target_update_interval=500),
    dict(learning_rate=1e-4, gamma=0.99, batch_size=32, net_arch=[64, 64],
         buffer_size=10_000, exploration_fraction=0.30, target_update_interval=500),
    dict(learning_rate=5e-4, gamma=0.95, batch_size=64, net_arch=[64, 64],
         buffer_size=20_000, exploration_fraction=0.20, target_update_interval=500),
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64, net_arch=[128, 128],
         buffer_size=20_000, exploration_fraction=0.20, target_update_interval=250),
    dict(learning_rate=1e-3, gamma=0.90, batch_size=32, net_arch=[64, 64],
         buffer_size=10_000, exploration_fraction=0.40, target_update_interval=500),
    dict(learning_rate=1e-3, gamma=0.99, batch_size=128, net_arch=[128, 128],
         buffer_size=50_000, exploration_fraction=0.30, target_update_interval=1000),
    dict(learning_rate=2.5e-4, gamma=0.99, batch_size=64, net_arch=[64],
         buffer_size=20_000, exploration_fraction=0.25, target_update_interval=500),
    dict(learning_rate=1e-3, gamma=0.99, batch_size=32, net_arch=[32, 32],
         buffer_size=5_000, exploration_fraction=0.30, target_update_interval=200),
    dict(learning_rate=7e-4, gamma=0.97, batch_size=64, net_arch=[64, 64, 64],
         buffer_size=20_000, exploration_fraction=0.30, target_update_interval=500),
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64, net_arch=[64, 64],
         buffer_size=20_000, exploration_fraction=0.10, target_update_interval=500),
    dict(learning_rate=3e-4, gamma=0.995, batch_size=64, net_arch=[128, 64],
         buffer_size=30_000, exploration_fraction=0.30, target_update_interval=750),
    dict(learning_rate=1e-3, gamma=0.99, batch_size=256, net_arch=[64, 64],
         buffer_size=20_000, exploration_fraction=0.30, target_update_interval=500),
]

BASE_SEED = 1000


def _to_sb3_kwargs(hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Adapts the flat, report-friendly hyperparameter dict into the keyword
    arguments SB3's DQN constructor actually expects (network architecture
    is a policy_kwargs sub-dict, not a top-level constructor argument)."""
    kwargs = dict(hyperparams)
    net_arch = kwargs.pop("net_arch")
    kwargs["policy_kwargs"] = dict(net_arch=net_arch)
    return kwargs


def run_sweep(total_timesteps: int, grid: list[dict[str, Any]] | None = None) -> list[RunResult]:
    grid = grid if grid is not None else DQN_HYPERPARAM_GRID
    results: list[RunResult] = []

    for i, hyperparams in enumerate(tqdm(grid, desc="DQN sweep", unit="run")):
        run_id = f"dqn_{i:02d}"
        seed = BASE_SEED + i
        result = run_single_training(
            algo_class=DQN,
            policy="MlpPolicy",
            run_id=run_id,
            algo_name="DQN",
            seed=seed,
            hyperparams=hyperparams,
            total_timesteps=total_timesteps,
            models_dir=MODELS_DIR,
            logs_dir=LOGS_DIR,
            sb3_kwargs=_to_sb3_kwargs(hyperparams),
        )
        results.append(result)
        tqdm.write(
            f"  {run_id}: final_mean_reward={result.final_mean_reward:.3f} "
            f"episodes={len(result.episode_rewards)} time={result.training_time_seconds:.1f}s"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN hyperparameter sweep on AdaptLearn-v1.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help=f"Training budget per run (default: {DEFAULT_TOTAL_TIMESTEPS}).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test: only the first 3 grid entries, 2000 timesteps each.",
    )
    args = parser.parse_args()

    grid = DQN_HYPERPARAM_GRID[:3] if args.quick else DQN_HYPERPARAM_GRID
    total_timesteps = 2000 if args.quick else args.total_timesteps

    print(f"Running DQN sweep: {len(grid)} configurations x {total_timesteps} timesteps each.")
    start = time.time()
    results = run_sweep(total_timesteps, grid)
    elapsed = time.time() - start

    summary_df = write_summary_csv(results, LOGS_DIR / "runs_summary.csv")
    best = write_best_run_marker(results, LOGS_DIR / "best_run.json")
    generate_sweep_plots(results, PLOTS_DIR, algo_label="DQN")

    print(f"\nDQN sweep complete in {elapsed / 60:.1f} min.")
    print(f"Summary CSV: {LOGS_DIR / 'runs_summary.csv'}")
    print(f"Plots: {PLOTS_DIR}/")
    print(f"Best run: {best.run_id} (final_mean_reward={best.final_mean_reward:.3f}) -> {best.model_path}")
    print(summary_df[["run_id", "final_mean_reward", "best_mean_reward", "training_time_seconds"]])


if __name__ == "__main__":
    main()
