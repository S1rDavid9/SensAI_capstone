"""AdaptLearn-v1 project entry point.

A single CLI tying together the pieces built across this project: the
per-algorithm training sweeps, the cross-algorithm comparison, and a live
rendered demo of a trained agent (the piece `environment/rendering.py`'s
module docstring refers to as "main.py runs a trained agent" for
screen-recording). Every subcommand is a thin wrapper over the module that
already does the real work -- this file adds no new training, rendering, or
environment logic of its own.

Usage:
    uv run python main.py train --algo all
    uv run python main.py train --algo dqn --quick        # fast smoke test
    uv run python main.py compare
    uv run python main.py demo                             # best model overall, live window
    uv run python main.py demo --algo ppo                  # best PPO run specifically
    uv run python main.py demo --manual                    # human_manual pacing (spacebar-advance), for recording
    uv run python main.py demo --seed 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import ACTION_NAMES, AdaptLearnEnv

ALGO_LOG_DIRS: dict[str, Path] = {
    "dqn": Path("logs/dqn"),
    "ppo": Path("logs/pg/ppo"),
    "a2c": Path("logs/pg/a2c"),
    "reinforce": Path("logs/pg/reinforce"),
}
ALGO_CLASSES: dict[str, type] = {"dqn": DQN, "ppo": PPO, "a2c": A2C}


def _load_best_run(algo: str) -> dict:
    best_run_path = ALGO_LOG_DIRS[algo] / "best_run.json"
    if not best_run_path.exists():
        raise FileNotFoundError(
            f"No completed sweep found for {algo!r} ({best_run_path} missing). "
            f"Run `uv run python main.py train --algo {algo}` first."
        )
    return json.loads(best_run_path.read_text())


def _load_model(algo: str, model_path: str):
    if algo in ALGO_CLASSES:
        return ALGO_CLASSES[algo].load(model_path)
    from training.pg_training import REINFORCE  # local import: keeps torch-only REINFORCE off the default import path

    return REINFORCE.load(model_path)


def _best_overall_algo() -> str:
    """Whichever algorithm's best run has the highest final_mean_reward,
    reading each algorithm's best_run.json -- the same numbers
    training/compare_algorithms.py ranks by."""
    best_by_algo = {}
    for algo in ALGO_LOG_DIRS:
        try:
            best_by_algo[algo] = _load_best_run(algo)["final_mean_reward"]
        except FileNotFoundError:
            continue
    if not best_by_algo:
        raise FileNotFoundError(
            "No completed sweeps found for any algorithm. Run `uv run python main.py train --algo all` first."
        )
    return max(best_by_algo, key=best_by_algo.get)


def cmd_train(args: argparse.Namespace) -> None:
    """Delegates to training/dqn_training.py and/or training/pg_training.py
    as subprocesses (rather than importing and calling their main()
    in-process) so their own argparse handling stays the single source of
    truth for flags/defaults, and sweep progress still streams live."""
    extra = ["--total-timesteps", str(args.total_timesteps)] if args.total_timesteps else []
    if args.quick:
        extra.append("--quick")

    if args.algo in ("dqn", "all"):
        subprocess.run([sys.executable, "-m", "training.dqn_training", *extra], check=True)
    if args.algo == "all":
        subprocess.run([sys.executable, "-m", "training.pg_training", "--algo", "all", *extra], check=True)
    elif args.algo in ("ppo", "a2c", "reinforce"):
        subprocess.run([sys.executable, "-m", "training.pg_training", "--algo", args.algo, *extra], check=True)


def cmd_compare(args: argparse.Namespace) -> None:
    from training.compare_algorithms import main as compare_main

    compare_main()


def cmd_demo(args: argparse.Namespace) -> None:
    algo = args.algo or _best_overall_algo()
    run = _load_best_run(algo)
    print(f"Demo: {algo.upper()} best run {run['run_id']} (final_mean_reward={run['final_mean_reward']:.2f})")

    model = _load_model(algo, run["model_path"])
    render_mode = "human_manual" if args.manual else "human"
    print(f"render_mode={render_mode!r} -- {'press spacebar to advance each step' if args.manual else 'auto-paced'}")

    env = AdaptLearnEnv(render_mode=render_mode)
    obs, info = env.reset(seed=args.seed)
    print(f"seed={args.seed}  initial state={info['engagement_state']}  difficulty={info['difficulty']}")

    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)  # renders internally
        step += 1
        print(
            f"t={step:02d}  action={ACTION_NAMES[action]:<20s}  "
            f"state={info['engagement_state']:<10s}  reward={reward:+.2f}"
        )

    outcome = "DROPOUT" if info["dropout"] else "SUCCESS" if info["success"] else "TRUNCATED (max steps)"
    print(f"\nEpisode end after {step} steps: {outcome}")
    env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AdaptLearn-v1: train, compare, and demo adaptive-tutoring RL agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run a hyperparameter sweep for one algorithm (or all four).")
    p_train.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce", "all"], default="all")
    p_train.add_argument("--total-timesteps", type=int, default=None, help="Override the default 20,000-timestep budget per run.")
    p_train.add_argument("--quick", action="store_true", help="Fast smoke test: a few grid entries, 2000 timesteps each.")
    p_train.set_defaults(func=cmd_train)

    p_compare = sub.add_parser("compare", help="Regenerate the cross-algorithm comparison plots/table (see logs/comparison/).")
    p_compare.set_defaults(func=cmd_compare)

    p_demo = sub.add_parser("demo", help="Render a live episode of a trained agent.")
    p_demo.add_argument(
        "--algo",
        choices=["dqn", "ppo", "a2c", "reinforce"],
        default=None,
        help="Which algorithm's best run to demo (default: best overall, across all four).",
    )
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument(
        "--manual",
        action="store_true",
        help="Use render_mode='human_manual' (spacebar-advance pacing, for recording) instead of the fixed-timing 'human' mode.",
    )
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
