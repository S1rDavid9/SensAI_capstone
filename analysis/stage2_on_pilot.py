"""Stage 1 -> Stage 2 on the real pilot data.

For every answered question, the Stage 1 classifier predicts the learner's
state from their last five answers (using a model that never saw that
learner), and the v2 DQN (with guardrails) chooses an action from that state
and the real telemetry. This is what the full system would have done during
the pilot. Aggregates go to analysis/results/stage2_pilot.json; the
plausibility rating sheet, which contains learner-level rows, goes to
pilot/data/ (never committed).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.dataset import CLASSES, blocks, load_tables, per_question, window_features
from analysis.stage1 import DEPLOYED_FEATURES
from pilot.labeler import RECENT_WINDOW, AnswerTelemetry, build_observation
from pilot.policy import shadow_decision

ROOT = Path(__file__).parent.parent
CLASS_TO_STATE = {"under_challenged": "rushing", "on_task": "on_task", "needs_support": "struggling"}
WINDOW = 5


def run() -> pd.DataFrame:
    t = load_tables()
    q = per_question(t)
    codes = t["sessions"].set_index("id").participant_code
    q["learner"] = q.session_id.map(codes)
    train = blocks(t)
    y = train.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    rows = []
    for learner, lq in q.sort_values("question_index").groupby("learner"):
        mask = (train.learner != learner).to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        model.fit(train.loc[mask, DEPLOYED_FEATURES].to_numpy(), y[mask])
        lq = lq.reset_index(drop=True)
        for i, a in lq.iterrows():
            window = lq.iloc[max(0, i - WINDOW + 1): i + 1]
            feats = window_features(window)
            p = np.zeros(len(CLASSES))
            p[model.classes_] = model.predict_proba(np.array([[feats[f] for f in DEPLOYED_FEATURES]]))[0]
            cls = CLASSES[int(p.argmax())]
            state = CLASS_TO_STATE[cls]
            recent = lq.iloc[max(0, i - RECENT_WINDOW + 1): i + 1]
            tel = AnswerTelemetry(
                response_time_s=a.rt, first_input_s=None if pd.isna(a.first_input) else a.first_input,
                correct=bool(a.correct), hints_used=int(a.hints_used), difficulty=int(a.difficulty),
                recent_errors=[not c for c in recent.correct.astype(bool)],
                questions_at_level=int((lq.iloc[: i + 1].difficulty == a.difficulty).sum()))
            d = shadow_decision(build_observation(state, tel), state, int(a.difficulty))
            rows.append({"learner": learner, "question": int(a.question_index) + 1, "kind": a.kind,
                         "level": int(a.difficulty), "correct": bool(a.correct), "rt_s": round(a.rt, 1),
                         "first_input_s": None if pd.isna(a.first_input) else round(a.first_input, 1),
                         "hints": int(a.hints_used), "pauses": int(a.pauses),
                         "stage1_class": cls, "state": state, "p_needs_support": round(float(p[2]), 2),
                         "dqn_action": d["raw_action_name"], "final_action": d["action_name"],
                         "guardrail": d["guardrail"]})
    return pd.DataFrame(rows)


def rating_sheet(df: pd.DataFrame, n_per_state: int = 20, seed: int = 0) -> pd.DataFrame:
    """A blind sheet for the researcher and teacher: each sampled moment once
    with the DQN's own action and, where a guardrail changed it, once with the
    final action. Shuffled; which is which is kept in a separate key."""
    rng = np.random.default_rng(seed)
    picks = pd.concat([g.sample(min(n_per_state, len(g)), random_state=seed) for _, g in df.groupby("state")])
    items = []
    for _, r in picks.iterrows():
        items.append({**r.to_dict(), "action_shown": r.dqn_action, "version": "dqn"})
        if r.guardrail:
            items.append({**r.to_dict(), "action_shown": r.final_action, "version": "guarded"})
    sheet = pd.DataFrame(items).sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
    sheet.insert(0, "item", range(1, len(sheet) + 1))
    return sheet


def main() -> None:
    df = run()
    t = load_tables()
    provisional = t["decisions"]
    summary = {
        "answers": len(df), "learners": int(df.learner.nunique()),
        "predicted_states": df.state.value_counts().to_dict(),
        "provisional_rule_states_logged_during_pilot": provisional.state.value_counts().to_dict(),
        "dqn_action_by_state (share)": {s: (g.dqn_action.value_counts(normalize=True).round(3).to_dict())
                                        for s, g in df.groupby("state")},
        "final_action_by_state (share, after guardrails)": {s: (g.final_action.value_counts(normalize=True).round(3).to_dict())
                                                            for s, g in df.groupby("state")},
        "guardrail_rate": round(float(df.guardrail.notna().mean()), 3),
        "guardrail_reasons": Counter(df.guardrail.dropna()).most_common(),
    }
    (ROOT / "analysis" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis" / "results" / "stage2_pilot.json").write_text(json.dumps(summary, indent=2))
    sheet = rating_sheet(df)
    out = ROOT / "pilot" / "data"
    df.to_csv(out / "stage2_decisions_per_answer.csv", index=False)
    sheet.drop(columns=["dqn_action", "final_action", "guardrail", "version", "stage1_class", "p_needs_support"]).assign(
        rating_researcher="", rating_teacher="").to_csv(out / "plausibility_rating_sheet.csv", index=False)
    sheet[["item", "version", "dqn_action", "final_action", "guardrail"]].to_csv(out / "plausibility_rating_key.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if "share" not in k}, indent=1))
    for k in ("dqn_action_by_state (share)", "final_action_by_state (share, after guardrails)"):
        print("\n" + k)
        for s, d in summary[k].items():
            print(f"  {s:11s}", {a: v for a, v in d.items() if v >= 0.05})
    print(f"\nRating sheet: {len(sheet)} items from {sheet.groupby(['learner','question']).ngroups} moments")


if __name__ == "__main__":
    main()
