"""Training-stability metrics (DQN Q-loss, PPO/A2C/REINFORCE policy entropy)
for the four current best runs -- dqn_02, ppo_11, a2c_07, reinforce_00.

None of the 48 original sweep runs had this logging enabled (see the report
data extraction, Section 5/8 -- no tensorboard_log was ever passed, and
REINFORCE's per-batch entropy was computed but never saved). This script
reruns exactly those four runs -- same hyperparameters, same seed, same
total_timesteps=20000 already on record in each algorithm's best_run.json --
adding ONLY the logging itself. It is not a re-tune: `training.run_single_
training`'s new `tensorboard_log` parameter (see training/__init__.py) and
REINFORCE's new `entropy_history` list (see training/pg_training.py) are
both purely additive/backward-compatible, so the actual training math for
these four runs is unaffected -- the reproduced final_mean_reward values
should land at or very near the recorded ones (verified below, not assumed).

All new output goes under logs/stability/ and models/stability/ -- the
existing logs/dqn/, logs/pg/, models/dqn/, models/pg/ directories (the other
44 runs) are never opened for writing by this script.

Run with:
    uv run python tests/stability_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from environment.custom_env import AdaptLearnEnv
from training import _read_monitor_csv, run_single_training
from training.pg_training import REINFORCE

STABILITY_DIR = Path("logs/stability")
TB_DIR = STABILITY_DIR / "tensorboard"
PLOTS_DIR = STABILITY_DIR / "plots"
RUNS_DIR = STABILITY_DIR / "runs"
MODELS_DIR = Path("models/stability")  # isolated from models/dqn, models/pg

TOTAL_TIMESTEPS = 20000

# Hyperparameters/seeds copied verbatim from logs/{dqn,pg/ppo,pg/a2c}/best_run.json
SB3_RUNS: dict[str, dict[str, Any]] = {
    "dqn": dict(
        algo_class=DQN,
        run_id="dqn_02",
        seed=1002,
        hyperparams=dict(
            learning_rate=0.0005,
            gamma=0.95,
            batch_size=64,
            net_arch=[64, 64],
            buffer_size=20000,
            exploration_fraction=0.2,
            target_update_interval=500,
        ),
        recorded_final_mean_reward=9.6585,
        scalar_tag="train/loss",
        metric_name="loss",
    ),
    "ppo": dict(
        algo_class=PPO,
        run_id="ppo_11",
        seed=2011,
        hyperparams=dict(
            learning_rate=0.001,
            gamma=0.995,
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            ent_coef=0.0,
            net_arch=[128, 64],
        ),
        recorded_final_mean_reward=7.6745,
        scalar_tag="train/entropy_loss",
        metric_name="entropy",
    ),
    "a2c": dict(
        algo_class=A2C,
        run_id="a2c_07",
        seed=2007,
        hyperparams=dict(
            learning_rate=0.0007,
            gamma=0.99,
            n_steps=5,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            net_arch=[64, 64],
        ),
        recorded_final_mean_reward=5.3250,
        scalar_tag="train/entropy_loss",
        metric_name="entropy",
    ),
}

REINFORCE_RUN = dict(
    run_id="reinforce_00",
    seed=2000,
    hyperparams=dict(
        learning_rate=0.001,
        gamma=0.99,
        net_arch=[64, 64],
        batch_episodes=4,
        entropy_coef=0.0,
        use_baseline=True,
    ),
    recorded_final_mean_reward=3.0775,
)


def _to_sb3_kwargs(hyperparams: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(hyperparams)
    net_arch = kwargs.pop("net_arch")
    kwargs["policy_kwargs"] = dict(net_arch=net_arch)
    return kwargs


def run_sb3_stability(algo_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    print(f"\n=== {algo_key.upper()} stability rerun ({cfg['run_id']}, seed={cfg['seed']}) ===")
    tb_log = str(TB_DIR / algo_key)

    result = run_single_training(
        algo_class=cfg["algo_class"],
        policy="MlpPolicy",
        run_id=cfg["run_id"],
        algo_name=algo_key.upper(),
        seed=cfg["seed"],
        hyperparams=cfg["hyperparams"],
        total_timesteps=TOTAL_TIMESTEPS,
        models_dir=MODELS_DIR / algo_key,
        logs_dir=RUNS_DIR / algo_key,
        sb3_kwargs=_to_sb3_kwargs(cfg["hyperparams"]),
        tensorboard_log=tb_log,
    )

    print(
        f"  reproduced final_mean_reward = {result.final_mean_reward:.4f}"
        f"  (recorded: {cfg['recorded_final_mean_reward']:.4f})"
    )

    run_dirs = list((TB_DIR / algo_key).glob(f"{cfg['run_id']}_*"))
    if not run_dirs:
        raise RuntimeError(f"No tensorboard run directory found under {TB_DIR / algo_key}")
    run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError(f"No tensorboard event file found under {run_dir}")

    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()
    tag = cfg["scalar_tag"]
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        raise RuntimeError(f"Tag {tag!r} not found for {algo_key}; available scalar tags: {available}")
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    metric = cfg["metric_name"]
    csv_path = STABILITY_DIR / f"{algo_key}_{metric}.csv"
    pd.DataFrame({"step": steps, "value": values}).to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5.5))
    plt.plot(steps, values, color="tab:blue" if algo_key == "dqn" else "tab:orange")
    plt.xlabel("Training step")
    plt.ylabel(tag)
    plt.title(f"{algo_key.upper()} ({cfg['run_id']}): {tag} over training")
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"{algo_key}_{metric}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  wrote {csv_path} ({len(steps)} points) and {plot_path}")

    return dict(
        reproduced_final_mean_reward=result.final_mean_reward,
        recorded_final_mean_reward=cfg["recorded_final_mean_reward"],
        n_points=len(steps),
        tag=tag,
    )


def run_reinforce_stability() -> dict[str, Any]:
    cfg = REINFORCE_RUN
    print(f"\n=== REINFORCE stability rerun ({cfg['run_id']}, seed={cfg['seed']}) ===")

    run_dir = RUNS_DIR / "reinforce" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = run_dir / "monitor.csv"
    env = Monitor(AdaptLearnEnv(), filename=str(monitor_path))

    model = REINFORCE("MlpPolicy", env, seed=cfg["seed"], verbose=0, **cfg["hyperparams"])
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    env.close()

    model_dir = MODELS_DIR / "reinforce" / cfg["run_id"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "model.zip"))

    episode_rewards, _ = _read_monitor_csv(monitor_path)
    window = min(20, len(episode_rewards)) or 1
    final_mean_reward = float(np.mean(episode_rewards[-window:])) if episode_rewards else float("nan")
    print(
        f"  reproduced final_mean_reward = {final_mean_reward:.4f}"
        f"  (recorded: {cfg['recorded_final_mean_reward']:.4f})"
    )

    entropy_history = model.entropy_history
    csv_path = STABILITY_DIR / "reinforce_entropy.csv"
    pd.DataFrame({"batch_index": range(len(entropy_history)), "entropy": entropy_history}).to_csv(
        csv_path, index=False
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(range(len(entropy_history)), entropy_history, color="tab:red")
    plt.xlabel(f"Batch index (each = {cfg['hyperparams']['batch_episodes']} episodes)")
    plt.ylabel("Policy entropy (mean, nats)")
    plt.title(f"REINFORCE ({cfg['run_id']}): policy entropy over training")
    plt.tight_layout()
    plot_path = PLOTS_DIR / "reinforce_entropy.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  wrote {csv_path} ({len(entropy_history)} points) and {plot_path}")

    return dict(
        reproduced_final_mean_reward=final_mean_reward,
        recorded_final_mean_reward=cfg["recorded_final_mean_reward"],
        n_points=len(entropy_history),
        tag="entropy (batch mean, nats)",
    )


def main() -> None:
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for algo_key, cfg in SB3_RUNS.items():
        summary[algo_key] = run_sb3_stability(algo_key, cfg)

    summary["reinforce"] = run_reinforce_stability()

    with open(STABILITY_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("SUMMARY (reproduced vs. recorded final_mean_reward)")
    print("=" * 78)
    for algo_key, s in summary.items():
        diff = s["reproduced_final_mean_reward"] - s["recorded_final_mean_reward"]
        print(
            f"  {algo_key.upper():<10s} reproduced={s['reproduced_final_mean_reward']:+.4f}  "
            f"recorded={s['recorded_final_mean_reward']:+.4f}  diff={diff:+.4f}  "
            f"n_points={s['n_points']}  metric={s['tag']}"
        )
    print(f"\nWrote {STABILITY_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
