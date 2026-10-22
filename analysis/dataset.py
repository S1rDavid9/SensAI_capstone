"""Turns the pilot database into one row per check-in block.

Each learner answered "How is it going?" after questions 5, 10 and 15. Each
check-in labels the five questions just before it, and the features are
the telemetry from those five questions: exactly what the app could compute
live, at the moment a check-in would have been shown.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(os.environ.get("PILOT_DB", Path(__file__).parent.parent / "pilot" / "data" / "pilot.sqlite3"))

# Check-in answer -> behavioural state -> the DQN's state slot.
CHECKIN_TO_STATE = {"easy": "rushing", "good": "on_task", "hard": "struggling", "lost": "stuck"}
# Three classes used for training: too few "hard" (4) and "lost" (7) answers to separate them.
CHECKIN_TO_CLASS = {"easy": "under_challenged", "good": "on_task", "hard": "needs_support", "lost": "needs_support"}
CLASSES = ["under_challenged", "on_task", "needs_support"]

FEATURES = [
    "error_rate",          # share of the 5 answers that were wrong
    "median_rt",           # seconds from question shown to answer
    "median_first_input",  # seconds before the first keystroke or choice
    "pauses_per_q",        # 20-second pauses per question
    "hints_per_q",         # hints opened per question
    "left_window",         # times the learner switched away
    "mean_difficulty",     # level of the questions (1-5), known to the app
]


def load_tables(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    tables = {t: pd.read_sql(f"SELECT * FROM {t}", con)
              for t in ("sessions", "answers", "events", "checkins", "observations", "decisions")}
    con.close()
    return tables


def per_question(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    a = t["answers"].copy()
    ev = t["events"]
    pauses = ev[ev.type == "idle"].groupby(["session_id", "question_index"]).size().rename("pauses")
    left = ev[ev.type == "focus_lost"].groupby(["session_id", "question_index"]).size().rename("left_window")
    a = a.join(pauses, on=["session_id", "question_index"]).join(left, on=["session_id", "question_index"])
    a[["pauses", "left_window"]] = a[["pauses", "left_window"]].fillna(0)
    a["rt"] = a.response_time_ms / 1000
    a["first_input"] = a.first_input_ms / 1000
    a["error"] = 1 - a.correct
    return a


def blocks(t: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    t = t or load_tables()
    q = per_question(t)
    codes = t["sessions"].set_index("id").participant_code
    checkins = t["checkins"].sort_values("id").drop_duplicates(["session_id", "after_question"])
    rows = []
    for _, c in checkins.iterrows():
        b = q[(q.session_id == c.session_id)
              & (q.question_index > c.after_question - 5) & (q.question_index <= c.after_question)]
        if b.empty:
            continue
        rows.append({
            "learner": codes[c.session_id],
            "block": int((c.after_question + 1) // 5),
            "checkin": c.choice,
            "label": CHECKIN_TO_CLASS[c.choice],
            "error_rate": b.error.mean(),
            "median_rt": b.rt.median(),
            "median_first_input": b.first_input.median(),
            "pauses_per_q": b.pauses.mean(),
            "hints_per_q": b.hints_used.mean(),
            "left_window": b.left_window.sum(),
            "mean_difficulty": b.difficulty.mean(),
            "n_questions": len(b),
        })
    df = pd.DataFrame(rows)
    df["median_first_input"] = df.median_first_input.fillna(df.median_rt)
    return df


def window_features(q: pd.DataFrame) -> dict[str, float]:
    """The same features for any run of answered questions (used live by the app)."""
    return {
        "error_rate": float(q.error.mean()),
        "median_rt": float(q.rt.median()),
        "median_first_input": float(q.first_input.fillna(q.rt).median()),
        "pauses_per_q": float(q.pauses.mean()),
        "hints_per_q": float(q.hints_used.mean()),
        "left_window": float(q.left_window.sum()),
        "mean_difficulty": float(q.difficulty.mean()),
    }


if __name__ == "__main__":
    df = blocks()
    print(df.label.value_counts().to_string())
    print(df[FEATURES].describe().round(2).to_string())
    print("learners:", df.learner.nunique(), "| rows:", len(df), "| any NaN:", bool(df[FEATURES].isna().any().any()))
    _ = np
