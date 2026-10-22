"""The Stage 1 classifier trained on the pilot (analysis/stage1.py).

It predicts one of three classes from the learner's last five answers; each
class maps to a behavioural state. "Needs support" maps to struggling: the
pilot had too few "hard" and "lost" check-ins to separate struggling from
stuck.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import joblib
import numpy as np

from pilot.core.states import BehaviouralState
from pilot.core.telemetry import LearnerHistory
from pilot.labelling.base import StateLabeler

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "stage1" / "stage1_logreg.joblib"
WINDOW = 5
CLASS_TO_STATE = {
    "under_challenged": BehaviouralState.RUSHING,
    "on_task": BehaviouralState.ON_TASK,
    "needs_support": BehaviouralState.STRUGGLING,
}


class Stage1Labeler(StateLabeler):
    source = "stage1_logreg_v1"

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        self.model_path = model_path

    @cached_property
    def _bundle(self) -> dict:
        return joblib.load(self.model_path)

    def features(self, history: LearnerHistory) -> dict[str, float]:
        return {
            "error_rate": history.error_rate(WINDOW),
            "median_rt": history.median_response_time(WINDOW),
        }

    def probabilities(self, history: LearnerHistory) -> dict[str, float]:
        feats = self.features(history)
        x = np.array([[feats[f] for f in self._bundle["features"]]])
        model = self._bundle["model"]
        p = model.predict_proba(x)[0]
        return {self._bundle["classes"][int(c)]: float(v) for c, v in zip(model.classes_, p)}

    def label(self, history: LearnerHistory) -> BehaviouralState:
        probs = self.probabilities(history)
        return CLASS_TO_STATE[max(probs, key=probs.get)]
