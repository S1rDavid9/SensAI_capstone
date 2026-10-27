"""Stage 1: predict a learner's behavioural state from telemetry.

Labels are the learners' own check-ins (three classes: under-challenged,
on task, needs support). Every model is evaluated with leave-one-learner-out
cross-validation: each learner's blocks are predicted by a model that never
saw that learner. Results (aggregates only) go to analysis/results/, and the
chosen model is saved to models/stage1/.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from analysis.dataset import CLASSES, FEATURES, blocks

# The deployed model: the two-signal model. In leave-one-learner-out testing it
# matches error rate alone (0.67 vs 0.66 balanced accuracy) and beats every
# larger feature set; see "feature_subsets" in results/stage1.json.
DEPLOYED_FEATURES = ["error_rate", "median_rt"]
SUBSETS = [["error_rate"], ["error_rate", "median_rt"], ["error_rate", "median_first_input"],
           ["error_rate", "pauses_per_q"], ["error_rate", "hints_per_q"], ["error_rate", "mean_difficulty"], FEATURES]

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "analysis" / "results"
MODEL_DIR = ROOT / "models" / "stage1"


def make_models():
    return {
        "Majority class (baseline)": None,
        "Error rate only (logistic)": ("error_only", lambda: make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))),
        "Logistic regression": ("all", lambda: make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))),
        "Shallow XGBoost (depth 2)": ("all", lambda: XGBClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.1, subsample=0.9,
            objective="multi:softprob", eval_metric="mlogloss", random_state=0)),
    }


def evaluate(df: pd.DataFrame) -> tuple[dict, dict]:
    y = df.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df.learner.to_numpy()
    logo = LeaveOneGroupOut()
    results, predictions = {}, {}
    for name, spec in make_models().items():
        pred = np.zeros_like(y)
        proba = np.zeros((len(y), len(CLASSES)))
        for train, test in logo.split(df, y, groups):
            if spec is None:
                majority = np.bincount(y[train], minlength=len(CLASSES)).argmax()
                pred[test] = majority
                proba[test, majority] = 1.0
                continue
            cols, factory = spec
            X = df[["error_rate"] if cols == "error_only" else FEATURES].to_numpy()
            model = factory()
            if isinstance(model, XGBClassifier):
                model.fit(X[train], y[train], sample_weight=compute_sample_weight("balanced", y[train]))
            else:
                model.fit(X[train], y[train])
            p = np.zeros((len(test), len(CLASSES)))
            p[:, model.classes_] = model.predict_proba(X[test])
            proba[test] = p
            pred[test] = p.argmax(1)
        needs = (y == CLASSES.index("needs_support")).astype(int)
        results[name] = {
            "balanced_accuracy": round(balanced_accuracy_score(y, pred), 3),
            "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 3),
            "recall_by_class": dict(zip(CLASSES, np.round(recall_score(y, pred, average=None, labels=range(3), zero_division=0), 3).tolist())),
            "needs_support_auc": round(roc_auc_score(needs, proba[:, CLASSES.index("needs_support")]), 3) if spec else 0.5,
            "confusion_matrix (rows=true, cols=predicted)": confusion_matrix(y, pred, labels=range(3)).tolist(),
        }
        predictions[name] = pred
    return results, predictions


def logreg_cv(df: pd.DataFrame, cols: list[str]) -> dict:
    y = df.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X = df[cols].to_numpy()
    pred, needs = np.zeros_like(y), np.zeros(len(y))
    for train, test in LeaveOneGroupOut().split(X, y, df.learner):
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[train], y[train])
        p = np.zeros((len(test), len(CLASSES)))
        p[:, m.classes_] = m.predict_proba(X[test])
        pred[test], needs[test] = p.argmax(1), p[:, CLASSES.index("needs_support")]
    return {"balanced_accuracy": round(balanced_accuracy_score(y, pred), 3),
            "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 3),
            "needs_support_auc": round(roc_auc_score((y == CLASSES.index("needs_support")).astype(int), needs), 3)}


def main() -> None:
    df = blocks()
    results, _ = evaluate(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    subsets = {" + ".join(c) if c != FEATURES else "all 7 features": logreg_cv(df, c) for c in SUBSETS}

    y = df.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    full = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    full.fit(df[FEATURES].to_numpy(), y)
    coefs = pd.DataFrame(full[-1].coef_, index=CLASSES, columns=FEATURES).round(2)

    final = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    final.fit(df[DEPLOYED_FEATURES].to_numpy(), y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": DEPLOYED_FEATURES, "classes": CLASSES,
                 "trained_on": "SensAI pilot, Excella School, 2026-10-21, 92 check-in blocks from 31 learners",
                 "evaluation": subsets[" + ".join(DEPLOYED_FEATURES)]},
                MODEL_DIR / "stage1_logreg.joblib")

    summary = {
        "data": {"blocks": len(df), "learners": int(df.learner.nunique()),
                 "label_counts": df.label.value_counts().reindex(CLASSES).to_dict(),
                 "checkin_counts": df.checkin.value_counts().to_dict()},
        "evaluation": "leave-one-learner-out cross-validation",
        "models": results,
        "feature_subsets (logistic regression, leave-one-learner-out)": subsets,
        "deployed_model": {"type": "Logistic regression, standardised, balanced class weights",
                           "features": DEPLOYED_FEATURES, **subsets[" + ".join(DEPLOYED_FEATURES)]},
        "all_feature_coefficients (standardised; positive = more likely)": coefs.to_dict(orient="index"),
    }
    (RESULTS / "stage1.json").write_text(json.dumps(summary, indent=2))
    for name, r in results.items():
        print(f"{name:30s} bal.acc {r['balanced_accuracy']:.3f}  macroF1 {r['macro_f1']:.3f}  "
              f"needs-support AUC {r['needs_support_auc']:.3f}  recall {r['recall_by_class']}")
    print("\nConfusion (logistic regression), rows true, cols predicted", CLASSES)
    print(np.array(results["Logistic regression"]["confusion_matrix (rows=true, cols=predicted)"]))
    print("\nCoefficients:\n", coefs.to_string())


if __name__ == "__main__":
    main()
