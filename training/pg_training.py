"""Policy-gradient family hyperparameter sweeps for AdaptLearn-v1: REINFORCE
(custom implementation), PPO, and A2C.

Mirrors training/dqn_training.py's design exactly (curated 12-run grids, one
seed per configuration, every run logged, shared plotting/summary utilities
from training/__init__.py) -- see that module's docstring for the reasoning
behind those choices, which applies identically here.

WHY A CUSTOM REINFORCE IMPLEMENTATION
------------------------------------------
REINFORCE (Williams, 1992) is not implemented in Stable-Baselines3, which
only ships value-based and actor-critic algorithms. The brief explicitly
anticipates this ("custom implementation if not natively in SB3 ... use a
standard accepted implementation"). The implementation below is the
textbook version: Monte Carlo policy gradient over full episode returns,
with a per-batch return-normalization baseline (subtract the batch mean,
divide by the batch std) to reduce gradient variance -- this is the
"REINFORCE with baseline" variant covered in Sutton & Barto (2nd ed., ch.
13.4), not a bespoke design. The `REINFORCE` class below deliberately
mirrors Stable-Baselines3's constructor/`.learn()`/`.save()`/`.predict()`
interface so it is a drop-in for `training.run_single_training`, the exact
same driver used for DQN/PPO/A2C -- no separate training loop needed for it.

Usage
-----
    uv run python -m training.pg_training                    # all three algorithms
    uv run python -m training.pg_training --algo reinforce   # just one
    uv run python -m training.pg_training --quick            # fast smoke test
    uv run python -m training.pg_training --total-timesteps 50000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import A2C, PPO
from torch.distributions import Categorical
from tqdm import tqdm

from training import (
    RunResult,
    generate_sweep_plots,
    run_single_training,
    write_best_run_marker,
    write_summary_csv,
)

MODELS_DIR = Path("models/pg")
LOGS_DIR = Path("logs/pg")

DEFAULT_TOTAL_TIMESTEPS = 20_000
BASE_SEED = 2000  # distinct range from DQN's 1000s, so seeds never collide across algorithms


# ============================================================================
# REINFORCE (custom -- see module docstring)
# ============================================================================


class _REINFORCEPolicyNet(nn.Module):
    """A small MLP mapping observations to action logits. Architecture is
    configurable (net_arch) exactly like SB3's policy_kwargs, for a fair,
    like-for-like hyperparameter sweep against DQN/PPO/A2C."""

    def __init__(self, obs_dim: int, n_actions: int, net_arch: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in net_arch:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class REINFORCE:
    """Monte Carlo policy gradient (REINFORCE with a batch-normalized return
    baseline). Constructor/`.learn()`/`.save()`/`.predict()` deliberately
    mirror Stable-Baselines3's on-policy algorithms so this is a drop-in for
    `training.run_single_training` -- see module docstring.

    Design notes (judgment calls not fully specified by "implement REINFORCE"):
      * `batch_episodes` full episodes are collected before each gradient
        step (REINFORCE is Monte Carlo -- it needs complete episodes to
        compute true returns, unlike DQN/PPO/A2C which can bootstrap from
        partial rollouts). `total_timesteps` is therefore a soft budget: the
        current batch always finishes before the loop checks it again, so a
        run may overshoot by up to one batch's worth of steps. This is
        standard for episode-based policy-gradient training and is noted
        here rather than truncating mid-episode, which would corrupt the
        Monte Carlo return calculation for that episode.
      * The baseline is the batch's own reward-to-go mean/std (per-batch
        normalization), not a learned value function -- that would make this
        an actor-critic method (closer to A2C) rather than REINFORCE proper.
    """

    def __init__(
        self,
        policy: str,
        env,
        *,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        net_arch: list[int] | tuple[int, ...] = (64, 64),
        batch_episodes: int = 4,
        entropy_coef: float = 0.0,
        use_baseline: bool = True,
        seed: int | None = None,
        verbose: int = 0,
    ) -> None:
        del policy, verbose  # accepted only for interface-compatibility with SB3's constructor
        self.env = env
        self.gamma = gamma
        self.batch_episodes = batch_episodes
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline

        if seed is not None:
            torch.manual_seed(seed)

        self._obs_dim = int(np.prod(env.observation_space.shape))
        self._n_actions = int(env.action_space.n)
        self._net_arch = list(net_arch)
        self.policy_net = _REINFORCEPolicyNet(self._obs_dim, self._n_actions, self._net_arch)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def predict(self, obs, deterministic: bool = False):
        """Matches SB3's `model.predict(obs, deterministic=...) -> (action, state)`
        signature, so this is a drop-in for the same policy-inspection/demo
        scripts used for the SB3-trained agents (e.g. tests/manual_watch_episode.py)."""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(obs_t)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(Categorical(logits=logits).sample().item())
        return action, None

    def learn(self, total_timesteps: int) -> "REINFORCE":
        steps_done = 0
        while steps_done < total_timesteps:
            batch_log_probs: list[torch.Tensor] = []
            batch_returns: list[float] = []
            batch_entropies: list[torch.Tensor] = []

            for _ in range(self.batch_episodes):
                obs, _ = self.env.reset()
                log_probs, rewards, entropies = [], [], []
                terminated = truncated = False

                while not (terminated or truncated):
                    obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
                    logits = self.policy_net(obs_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
                    log_probs.append(dist.log_prob(action))
                    entropies.append(dist.entropy())
                    rewards.append(reward)

                steps_done += len(rewards)

                # Discounted return-to-go from each timestep to episode end.
                returns: list[float] = []
                g = 0.0
                for r in reversed(rewards):
                    g = r + self.gamma * g
                    returns.insert(0, g)

                batch_log_probs.extend(log_probs)
                batch_returns.extend(returns)
                batch_entropies.extend(entropies)

            returns_t = torch.tensor(batch_returns, dtype=torch.float32)
            if self.use_baseline and returns_t.numel() > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std(unbiased=False) + 1e-8)
            log_probs_t = torch.cat(batch_log_probs)
            entropy_t = torch.cat(batch_entropies).mean()

            # Policy gradient ascent on E[log pi(a|s) * G] == minimize its negation;
            # a small entropy bonus (when entropy_coef > 0) discourages premature
            # collapse to a deterministic policy, the same role it plays in PPO/A2C.
            loss = -(log_probs_t * returns_t).mean() - self.entropy_coef * entropy_t

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self

    def save(self, path) -> None:
        torch.save(
            {
                "state_dict": self.policy_net.state_dict(),
                "obs_dim": self._obs_dim,
                "n_actions": self._n_actions,
                "net_arch": self._net_arch,
            },
            path,
        )

    @classmethod
    def load(cls, path, env=None) -> "REINFORCE":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls.__new__(cls)
        obj.env = env
        obj.gamma = None
        obj.batch_episodes = None
        obj.entropy_coef = None
        obj.use_baseline = None
        obj._obs_dim = checkpoint["obs_dim"]
        obj._n_actions = checkpoint["n_actions"]
        obj._net_arch = checkpoint["net_arch"]
        obj.policy_net = _REINFORCEPolicyNet(obj._obs_dim, obj._n_actions, obj._net_arch)
        obj.policy_net.load_state_dict(checkpoint["state_dict"])
        obj.optimizer = None
        return obj


# Grid design mirrors dqn_training.py: entry 00 is a baseline, each following
# entry perturbs one axis (learning rate, gamma, network size, episodes per
# update -- REINFORCE's analogue of "batch size", entropy regularization),
# and the final entry ablates the return-normalization baseline itself
# (use_baseline=False) -- directly testing, for the report, whether the
# baseline is actually earning its keep on this environment.
REINFORCE_HYPERPARAM_GRID: list[dict[str, Any]] = [
    dict(learning_rate=1e-3, gamma=0.99, net_arch=[64, 64], batch_episodes=4, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=1e-4, gamma=0.99, net_arch=[64, 64], batch_episodes=4, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=5e-3, gamma=0.99, net_arch=[64, 64], batch_episodes=4, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=1e-3, gamma=0.95, net_arch=[64, 64], batch_episodes=8, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=1e-3, gamma=0.90, net_arch=[64, 64], batch_episodes=4, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=1e-3, gamma=0.99, net_arch=[128, 128], batch_episodes=16, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=2.5e-4, gamma=0.99, net_arch=[64], batch_episodes=8, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=1e-3, gamma=0.99, net_arch=[32, 32], batch_episodes=4, entropy_coef=0.01, use_baseline=True),
    dict(learning_rate=7e-4, gamma=0.97, net_arch=[64, 64, 64], batch_episodes=8, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=1e-3, gamma=0.99, net_arch=[64, 64], batch_episodes=8, entropy_coef=0.02, use_baseline=True),
    dict(learning_rate=3e-4, gamma=0.995, net_arch=[128, 64], batch_episodes=8, entropy_coef=0.0, use_baseline=True),
    dict(learning_rate=1e-3, gamma=0.99, net_arch=[64, 64], batch_episodes=4, entropy_coef=0.0, use_baseline=False),
]


# ============================================================================
# PPO / A2C (native SB3)
# ============================================================================

PPO_HYPERPARAM_GRID: list[dict[str, Any]] = [
    dict(learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.0, net_arch=[64, 64]),
    dict(learning_rate=1e-4, gamma=0.99, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.0, net_arch=[64, 64]),
    dict(learning_rate=1e-3, gamma=0.99, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.0, net_arch=[64, 64]),
    dict(learning_rate=3e-4, gamma=0.95, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.0, net_arch=[64, 64]),
    dict(learning_rate=3e-4, gamma=0.90, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.0, net_arch=[64, 64]),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=512, batch_size=64, n_epochs=10, ent_coef=0.0, net_arch=[64, 64]),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=32, n_epochs=10, ent_coef=0.0, net_arch=[64, 64]),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, n_epochs=4, ent_coef=0.0, net_arch=[64, 64]),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.0, net_arch=[128, 128]),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.0, net_arch=[32, 32]),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.02, net_arch=[64, 64]),
    dict(learning_rate=1e-3, gamma=0.995, n_steps=512, batch_size=128, n_epochs=10, ent_coef=0.0, net_arch=[128, 64]),
]

A2C_HYPERPARAM_GRID: list[dict[str, Any]] = [
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, net_arch=[64, 64]),
    dict(learning_rate=1e-4, gamma=0.99, n_steps=5, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, net_arch=[64, 64]),
    dict(learning_rate=2e-3, gamma=0.99, n_steps=5, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, net_arch=[64, 64]),
    dict(learning_rate=7e-4, gamma=0.95, n_steps=5, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, net_arch=[64, 64]),
    dict(learning_rate=7e-4, gamma=0.90, n_steps=5, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, net_arch=[64, 64]),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=20, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, net_arch=[64, 64]),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, gae_lambda=0.9, ent_coef=0.0, vf_coef=0.5, net_arch=[64, 64]),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, gae_lambda=1.0, ent_coef=0.01, vf_coef=0.5, net_arch=[64, 64]),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.25, net_arch=[64, 64]),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, net_arch=[128, 128]),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, net_arch=[32, 32]),
    dict(learning_rate=1e-3, gamma=0.995, n_steps=10, gae_lambda=0.95, ent_coef=0.02, vf_coef=0.5, net_arch=[128, 64]),
]


def _to_sb3_kwargs(hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Nests `net_arch` inside `policy_kwargs`, matching dqn_training.py's
    adapter -- SB3's on-policy algorithms take network architecture the
    same way DQN does."""
    kwargs = dict(hyperparams)
    net_arch = kwargs.pop("net_arch")
    kwargs["policy_kwargs"] = dict(net_arch=net_arch)
    return kwargs


ALGORITHMS: dict[str, dict[str, Any]] = {
    "reinforce": dict(algo_class=REINFORCE, grid=REINFORCE_HYPERPARAM_GRID, to_sb3_kwargs=None),
    "ppo": dict(algo_class=PPO, grid=PPO_HYPERPARAM_GRID, to_sb3_kwargs=_to_sb3_kwargs),
    "a2c": dict(algo_class=A2C, grid=A2C_HYPERPARAM_GRID, to_sb3_kwargs=_to_sb3_kwargs),
}


def run_sweep(algo_name: str, total_timesteps: int, grid: list[dict[str, Any]] | None = None) -> list[RunResult]:
    spec = ALGORITHMS[algo_name]
    grid = grid if grid is not None else spec["grid"]
    models_dir = MODELS_DIR / algo_name
    logs_dir = LOGS_DIR / algo_name
    results: list[RunResult] = []

    for i, hyperparams in enumerate(tqdm(grid, desc=f"{algo_name.upper()} sweep", unit="run")):
        run_id = f"{algo_name}_{i:02d}"
        seed = BASE_SEED + i
        sb3_kwargs = spec["to_sb3_kwargs"](hyperparams) if spec["to_sb3_kwargs"] else None
        result = run_single_training(
            algo_class=spec["algo_class"],
            policy="MlpPolicy",
            run_id=run_id,
            algo_name=algo_name.upper(),
            seed=seed,
            hyperparams=hyperparams,
            total_timesteps=total_timesteps,
            models_dir=models_dir,
            logs_dir=logs_dir,
            sb3_kwargs=sb3_kwargs,
        )
        results.append(result)
        tqdm.write(
            f"  {run_id}: final_mean_reward={result.final_mean_reward:.3f} "
            f"episodes={len(result.episode_rewards)} time={result.training_time_seconds:.1f}s"
        )

    return results


def run_algorithm(algo_name: str, total_timesteps: int, quick: bool) -> list[RunResult]:
    spec = ALGORITHMS[algo_name]
    grid = spec["grid"][:3] if quick else spec["grid"]
    budget = 2000 if quick else total_timesteps

    print(f"\nRunning {algo_name.upper()} sweep: {len(grid)} configurations x {budget} timesteps each.")
    start = time.time()
    results = run_sweep(algo_name, budget, grid)
    elapsed = time.time() - start

    logs_dir = LOGS_DIR / algo_name
    plots_dir = logs_dir / "plots"
    summary_df = write_summary_csv(results, logs_dir / "runs_summary.csv")
    best = write_best_run_marker(results, logs_dir / "best_run.json")
    generate_sweep_plots(results, plots_dir, algo_label=algo_name.upper())

    print(f"{algo_name.upper()} sweep complete in {elapsed / 60:.1f} min.")
    print(f"Summary CSV: {logs_dir / 'runs_summary.csv'}")
    print(f"Plots: {plots_dir}/")
    print(f"Best run: {best.run_id} (final_mean_reward={best.final_mean_reward:.3f}) -> {best.model_path}")
    print(summary_df[["run_id", "final_mean_reward", "best_mean_reward", "training_time_seconds"]])

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train policy-gradient hyperparameter sweeps on AdaptLearn-v1.")
    parser.add_argument(
        "--algo",
        choices=["reinforce", "ppo", "a2c", "all"],
        default="all",
        help="Which algorithm to sweep (default: all three).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help=f"Training budget per run (default: {DEFAULT_TOTAL_TIMESTEPS}).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test: only the first 3 grid entries per algorithm, 2000 timesteps each.",
    )
    args = parser.parse_args()

    algos = ["reinforce", "ppo", "a2c"] if args.algo == "all" else [args.algo]
    for algo_name in algos:
        run_algorithm(algo_name, args.total_timesteps, args.quick)


if __name__ == "__main__":
    main()
