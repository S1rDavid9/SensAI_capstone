"""The Stage 2 DQN in shadow mode: asked what it would do, never obeyed.

Uses the best DQN trained on simulator v2 by default (see design decision 12
in environment/custom_env.py). Set SENSAI_POLICY=v1 to use the Phase 1
model instead. Every stored decision records which model made it.

Guardrails: a few actions are ruled out in situations where no teacher would
choose them, or where they would do nothing. When the DQN's first choice is
ruled out, the decision becomes its highest-valued allowed action. Both the
DQN's own choice and the final one are stored, so the analysis can report
the DQN on its own and with guardrails.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from environment.custom_env import ACTION_NAMES

ROOT = Path(__file__).resolve().parent.parent
BEST_RUN = {
    "v1": ROOT / "logs/dqn/best_run.json",
    "v2": ROOT / "logs/v2/dqn/best_run.json",
}

# Behavioural state -> actions ruled out, with the reason stored alongside.
_NOT_WHEN_LOST = {
    "advance_topic": "don't move on while the learner is {state}",
    "increase_difficulty": "don't make it harder while the learner is {state}",
    "introduce_challenge": "don't add a challenge while the learner is {state}",
}
GUARDRAILS: dict[str, dict[str, str]] = {
    "on_task": {"decrease_difficulty": "don't make it easier for a learner who is coping"},
    "stuck": {a: r.format(state="stuck") for a, r in _NOT_WHEN_LOST.items()},
    "struggling": {a: r.format(state="struggling") for a, r in _NOT_WHEN_LOST.items()},
}
MIN_DIFFICULTY, MAX_DIFFICULTY = 1, 5


@lru_cache(maxsize=1)
def _model():
    from stable_baselines3 import DQN

    version = os.environ.get("SENSAI_POLICY", "v2")
    best = json.loads(BEST_RUN[version].read_text())
    return DQN.load(ROOT / best["model_path"], device="cpu"), f"{version}/{best['run_id']}"


def ruled_out(state: str, difficulty: int) -> dict[str, str]:
    blocked = dict(GUARDRAILS.get(state, {}))
    if difficulty <= MIN_DIFFICULTY:
        blocked["decrease_difficulty"] = "already at the easiest level"
    if difficulty >= MAX_DIFFICULTY:
        blocked["increase_difficulty"] = "already at the hardest level"
    return blocked


def shadow_decision(obs: np.ndarray, state: str, difficulty: int) -> dict:
    model, model_id = _model()
    with torch.no_grad():
        q = model.q_net(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
    raw = int(np.argmax(q))
    blocked = ruled_out(state, difficulty)
    allowed = [i for i, name in enumerate(ACTION_NAMES) if name not in blocked]
    final = max(allowed, key=lambda i: q[i])
    return {
        "model": model_id,
        "raw_action_index": raw,
        "raw_action_name": ACTION_NAMES[raw],
        "action_index": final,
        "action_name": ACTION_NAMES[final],
        "guardrail": blocked.get(ACTION_NAMES[raw]) if final != raw else None,
        "q_values": [round(float(v), 4) for v in q],
    }
