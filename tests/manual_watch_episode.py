"""Quick, unscientific visual gut-check: watch one live episode of the
currently-saved best DQN agent, rendered in real time.

Not a validation pass. `render_mode="human"` already makes
AdaptLearnEnv.step()/reset() call render() internally (see
environment/custom_env.py), and the renderer itself now self-paces each
step in real time (action cue -> mood tween -> brief pause, ~1.5-2.5s
total -- see rendering.py's STEP_TOTAL_SECONDS), so this script does not
call render() again itself; it just adds a short extra pause on top so the
printed action/state line stays readable alongside the animation.

Run with:
    uv run python tests/manual_watch_episode.py
    uv run python tests/manual_watch_episode.py --seed 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import DQN

from environment.custom_env import ACTION_NAMES, AdaptLearnEnv

DEFAULT_MODEL_PATH = Path("models/dqn/dqn_02/model.zip")
READ_PAUSE_SECONDS = 0.2  # small extra pause on top of the renderer's own ~1.5-2.5s per-step choreography


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch one live episode of a trained AdaptLearn-v1 agent.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = DQN.load(str(args.model))

    env = AdaptLearnEnv(render_mode="human")
    obs, info = env.reset(seed=args.seed)
    print(f"seed={args.seed}  initial state={info['engagement_state']}  difficulty={info['difficulty']}")

    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)  # renders internally (render_mode="human")
        step += 1
        print(
            f"t={step:02d}  action={ACTION_NAMES[action]:<20s}  "
            f"state={info['engagement_state']:<10s}  reward={reward:+.2f}"
        )
        time.sleep(READ_PAUSE_SECONDS)

    outcome = "DROPOUT" if info["dropout"] else "SUCCESS" if info["success"] else "TRUNCATED (max steps)"
    print(f"\nEpisode end after {step} steps: {outcome}")
    env.close()


if __name__ == "__main__":
    main()
