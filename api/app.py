"""SensAI Stage 2 policy (AdaptLearn-v1 environment) as a JSON API -- MVP demonstration.

Shows that the environment's state/actions/rewards are trivially
JSON-serializable, so a frontend (a web dashboard, or the existing
PyOpenGL classroom renderer reimplemented in a browser) could drive a
live visualization by polling this API instead of running the Gymnasium
loop directly in Python. This is a minimal demonstration, not production
infrastructure -- no auth, no database, no deployment config, a single
global in-memory session. See api/README.md for how to run it and full
example requests.

Run:
    uv run uvicorn api.app:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import ACTION_NAMES, AdaptLearnEnv

app = FastAPI(
    title="SensAI API",
    description="SensAI Stage 2 MVP: the simulation-trained DQN scaffolding policy and its AdaptLearn-v1 environment, exposed as a JSON API.",
)

# Mirrors main.py's model-loading convention (kept self-contained here
# rather than importing main.py, matching how every other analysis
# script in this project -- compare_algorithms.py, generalization_test.py,
# etc. -- defines its own small copy of this rather than sharing one).
ALGO_LOG_DIRS: dict[str, Path] = {
    "dqn": Path("logs/dqn"),
    "ppo": Path("logs/pg/ppo"),
    "a2c": Path("logs/pg/a2c"),
    "reinforce": Path("logs/pg/reinforce"),
}
ALGO_CLASSES: dict[str, type] = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
DEFAULT_ALGO = "dqn"

_model_cache: dict[str, Any] = {}


def _load_best_run(algo: str) -> dict:
    best_run_path = ALGO_LOG_DIRS[algo] / "best_run.json"
    if not best_run_path.exists():
        raise HTTPException(404, f"No completed sweep found for {algo!r} ({best_run_path} missing).")
    return json.loads(best_run_path.read_text())


def _get_model(algo: str):
    """Loads and caches the trained model for `algo` -- repeated /reset
    calls for the same algorithm never touch disk again after the first."""
    if algo not in ALGO_LOG_DIRS:
        raise HTTPException(400, f"Unknown algo {algo!r}; choose one of {list(ALGO_LOG_DIRS)}.")
    if algo not in _model_cache:
        model_path = _load_best_run(algo)["model_path"]
        if algo in ALGO_CLASSES:
            _model_cache[algo] = ALGO_CLASSES[algo].load(model_path)
        else:
            from training.pg_training import REINFORCE  # local import, matches main.py's convention

            _model_cache[algo] = REINFORCE.load(model_path)
    return _model_cache[algo]


class _Session:
    """Single global in-memory session -- no multi-user handling, by
    design (see module docstring: this is a serialization demo, not a
    deployable service)."""

    env: AdaptLearnEnv | None = None
    model: Any = None
    algo: str | None = None
    history: list[dict] = []
    last_obs: Any = None
    done: bool = False


session = _Session()


class ResetRequest(BaseModel):
    algo: str | None = None
    seed: int | None = None


class StepRequest(BaseModel):
    action: int | None = None


def _reset_response(obs, info) -> dict:
    return {
        "observation": obs.tolist(),
        "engagement_state": info["engagement_state"],
        "difficulty": info["difficulty"],
        "step": info["steps"],
    }


def _step_response(action: int, obs, reward: float, terminated: bool, truncated: bool, info: dict) -> dict:
    return {
        "action_taken": {"index": action, "name": ACTION_NAMES[action]},
        "observation": obs.tolist(),
        "engagement_state": info["engagement_state"],
        "difficulty": info["difficulty"],
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()) -> dict:
    algo = body.algo or DEFAULT_ALGO
    model = _get_model(algo)

    env = AdaptLearnEnv()
    obs, info = env.reset(seed=body.seed)

    session.env = env
    session.model = model
    session.algo = algo
    session.history = []
    session.last_obs = obs
    session.done = False

    return _reset_response(obs, info)


@app.post("/step")
def step(body: StepRequest = StepRequest()) -> dict:
    if session.env is None:
        raise HTTPException(400, "No active episode -- call /reset first.")
    if session.done:
        raise HTTPException(400, "Episode has ended -- call /reset to start a new one.")

    if body.action is not None:
        if not session.env.action_space.contains(body.action):
            raise HTTPException(400, f"Invalid action {body.action}; expected 0..7.")
        action = body.action
    else:
        action, _ = session.model.predict(session.last_obs, deterministic=True)
        action = int(action)

    obs, reward, terminated, truncated, info = session.env.step(action)
    session.last_obs = obs
    session.done = terminated or truncated
    result = _step_response(action, obs, reward, terminated, truncated, info)
    session.history.append(result)
    return result


@app.get("/episode-summary")
def episode_summary() -> list[dict]:
    return session.history
