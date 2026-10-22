from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from environment.custom_env import ACTION_NAMES

ROOT = Path(__file__).resolve().parents[2]
BEST_RUN = {
    "v1": ROOT / "logs/dqn/best_run.json",      # Phase 1 simulator
    "v2": ROOT / "logs/v2/dqn/best_run.json",   # simulator v2 (design decision 12)
}


class ActionScorer(ABC):
    """Anything that scores the eight scaffolding actions for an observation."""

    model_id: str

    @abstractmethod
    def scores(self, observation: np.ndarray) -> np.ndarray:
        """One score per action in `ACTION_NAMES` order; higher is better."""


class DQNPolicy(ActionScorer):
    def __init__(self, version: str = "v2") -> None:
        if version not in BEST_RUN:
            raise ValueError(f"Unknown policy version {version!r}")
        self.version = version

    @cached_property
    def _loaded(self) -> tuple[object, str]:
        from stable_baselines3 import DQN

        best = json.loads(BEST_RUN[self.version].read_text())
        return DQN.load(ROOT / best["model_path"], device="cpu"), f"{self.version}/{best['run_id']}"

    @property
    def model_id(self) -> str:
        return self._loaded[1]

    def warm_up(self) -> None:
        """Load the network now rather than on the first learner's answer."""
        _ = self._loaded

    def scores(self, observation: np.ndarray) -> np.ndarray:
        model, _ = self._loaded
        with torch.no_grad():
            q = model.q_net(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
        return q.squeeze(0).numpy()


assert len(ACTION_NAMES) == 8
