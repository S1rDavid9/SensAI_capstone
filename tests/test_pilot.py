"""Tests for the SensAI pilot and adaptive app.

Unit tests cover each building block on its own (question bank, labellers,
guardrails, curriculum planner, database upgrade). Integration tests drive
the whole API through FastAPI's test client, in both session modes.
"""

import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from environment.custom_env import ACTION_NAMES
from pilot.adaptation import EFFECTS, CurriculumPlanner
from pilot.api import create_app
from pilot.config import Settings
from pilot.core.states import BehaviouralState
from pilot.core.tasks import BANK, CHECKIN_AFTER, PILOT_FORM_IDS, SESSION_LENGTH, QuestionBank
from pilot.core.telemetry import AnswerRecord, LearnerHistory, ObservationBuilder
from pilot.labelling import RuleBasedLabeler, Stage1Labeler
from pilot.policy import ActionScorer, DifficultyBoundsRule, Shield, StateRule
from pilot.storage import Database

KEY = {"X-Researcher-Key": "sensai-dev"}
OBSERVER = {"X-Researcher-Key": "sensai-observe"}
S = BehaviouralState


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(db_path=tmp_path / "pilot.sqlite3"))) as c:
        yield c


def history(*answers) -> LearnerHistory:
    defaults = dict(question_index=0, difficulty=2, correct=True, response_time_s=10.0, first_input_s=2.0, hints_used=0)
    return LearnerHistory(tuple(AnswerRecord(**{**defaults, "question_index": i, **a}) for i, a in enumerate(answers)))


class FixedScorer(ActionScorer):
    """A stand-in policy that prefers actions in a given order."""
    model_id = "test"

    def __init__(self, *preferred: str) -> None:
        self.preferred = preferred

    def scores(self, observation):
        return np.array([len(self.preferred) - self.preferred.index(a) if a in self.preferred else -1.0
                         for a in ACTION_NAMES])


def new_session(client, mode="shadow") -> dict:
    s = client.post("/api/researcher/sessions", headers=KEY, json={
        "guardian_consent": True, "learner_assent": True, "recorded_by": "AN", "mode": mode}).json()
    token = client.post("/api/learner/join", json={"code": s["participant_code"]}).json()["token"]
    return {"id": s["id"], "auth": {"code": s["participant_code"], "token": token}}


def run_session(client, auth, answer_for=lambda q, i: "0") -> list[dict]:
    seen = []
    for i in range(SESSION_LENGTH):
        cur = client.post("/api/learner/question", json=auth).json()
        seen.append(cur)
        q = cur["question"]
        r = client.post("/api/learner/answer", json={**auth, "question_index": i, "answer": answer_for(q, i),
                                                     "response_time_ms": 8000, "first_input_ms": 1500}).json()
        if r["checkin_due"]:
            client.post("/api/learner/checkin", json={**auth, "after_question": i, "choice": "good"})
    return seen


# --------------------------------------------------------------------------
# Question bank
# --------------------------------------------------------------------------

def test_bank_answers_are_valid():
    for q in BANK:
        assert q.is_correct(q.answer), q.id
        if q.choices:
            assert q.answer in q.choices and len(set(q.choices)) == len(q.choices), q.id
        assert "answer" not in q.public(0)
    assert {q.domain for q in BANK} == {"arithmetic", "pattern"}
    assert all(len(QuestionBank().at_level(level)) >= 7 for level in range(1, 6))


def test_pilot_form_is_unchanged():
    """The pilot data (21 October 2026) refers to these exact positions."""
    form = QuestionBank().pilot_form()
    assert [q.id for q in form] == list(PILOT_FORM_IDS) and len(form) == SESSION_LENGTH
    assert [q.difficulty for q in form] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    assert [q.answer for q in form] == ["15", "🔵", "11", "19", "14", "I", "24", "■", "85", "25", "21", "850",
                                       "8", "13", "21"]


def test_number_answers_tolerate_spaces_and_commas():
    q = QuestionBank().get("4-word-a")
    assert q.is_correct(" 850 ") and not q.is_correct("85")


# --------------------------------------------------------------------------
# Labellers (two implementations of one interface)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer, expected", [
    (dict(), S.ON_TASK),
    (dict(first_input_s=None), S.STUCK),
    (dict(first_input_s=25.0), S.STUCK),
    (dict(correct=False, response_time_s=2.0), S.RUSHING),
])
def test_rule_based_labeler(answer, expected):
    assert RuleBasedLabeler().label(history(answer)) == expected


def test_rule_based_labeler_struggling_uses_recent_errors():
    h = history(dict(correct=False), dict(correct=False), dict(correct=False, response_time_s=20.0))
    assert RuleBasedLabeler().label(h) == S.STRUGGLING


def test_stage1_labeler_follows_error_rate():
    stage1 = Stage1Labeler()
    all_wrong = history(*[dict(correct=False, response_time_s=30.0)] * 5)
    all_right_fast = history(*[dict(correct=True, response_time_s=6.0)] * 5)
    assert stage1.label(all_wrong) == S.STRUGGLING
    assert stage1.label(all_right_fast) == S.RUSHING
    assert abs(sum(stage1.probabilities(all_wrong).values()) - 1) < 1e-6


def test_observation_matches_dqn_input():
    obs = ObservationBuilder().build(S.RUSHING, history(dict(difficulty=5, hints_used=10, response_time_s=80)))
    assert obs.shape == (9,) and list(obs[:4]) == [0, 1, 0, 0]
    assert obs.min() >= 0 and obs.max() <= 1 and obs[7] == 1.0


# --------------------------------------------------------------------------
# Guardrails (shielding)
# --------------------------------------------------------------------------

def test_shield_passes_allowed_choice_through():
    d = Shield(FixedScorer("advance_topic")).decide(np.zeros(9), S.ON_TASK, 3)
    assert d.raw_action == d.action == "advance_topic" and d.guardrail is None


def test_shield_replaces_ruled_out_choice_with_next_best():
    d = Shield(FixedScorer("decrease_difficulty", "offer_hint")).decide(np.zeros(9), S.ON_TASK, 3)
    assert d.raw_action == "decrease_difficulty" and d.action == "offer_hint"
    assert d.guardrail == "don't make it easier for a learner who is coping"


@pytest.mark.parametrize("state", [S.STUCK, S.STRUGGLING])
def test_shield_never_moves_on_or_raises_difficulty_when_learner_is_lost(state):
    scorer = FixedScorer("advance_topic", "increase_difficulty", "introduce_challenge", "add_encouragement")
    assert Shield(scorer).decide(np.zeros(9), state, 3).action == "add_encouragement"


def test_difficulty_bounds_rule():
    rule = DifficultyBoundsRule()
    assert "decrease_difficulty" in rule.ruled_out(S.RUSHING, 1)
    assert "increase_difficulty" in rule.ruled_out(S.RUSHING, 5)
    assert rule.ruled_out(S.RUSHING, 3) == {}


def test_custom_rule_can_be_added():
    no_hints = StateRule(S.RUSHING, (("offer_hint", "no hints when rushing"),))
    d = Shield(FixedScorer("offer_hint", "advance_topic"), rules=(no_hints,)).decide(np.zeros(9), S.RUSHING, 3)
    assert d.action == "advance_topic"


# --------------------------------------------------------------------------
# Curriculum planner
# --------------------------------------------------------------------------

def test_every_action_has_an_effect():
    assert set(EFFECTS) == set(ACTION_NAMES)


@pytest.mark.parametrize("action, level_change", [
    ("increase_difficulty", +1), ("advance_topic", +1), ("introduce_challenge", +1),
    ("decrease_difficulty", -1), ("repeat_simplified", -1),
    ("offer_hint", 0), ("add_encouragement", 0), ("change_task_type", 0),
])
def test_planner_moves_difficulty_as_the_action_says(action, level_change):
    bank = QuestionBank()
    last = bank.get("3-sum-a")
    step = CurriculumPlanner(bank).next_step(action, last, {last.id}, seed=1)
    assert step.question.difficulty == 3 + level_change
    assert step.question.id != last.id and step.message
    assert (step.hint is not None) == (action == "offer_hint")


def test_planner_message_matches_what_actually_happened():
    bank = QuestionBank()
    easiest = bank.get("1-sum-a")
    step = CurriculumPlanner(bank).next_step("decrease_difficulty", easiest, {easiest.id}, seed=0)
    assert step.question.difficulty == 1 and "easier" not in step.message


def test_planner_respects_question_type_rules():
    bank, planner = QuestionBank(), CurriculumPlanner(QuestionBank())
    last = bank.get("3-numpat-a")
    assert planner.next_step("repeat_simplified", last, {last.id}, seed=0).question.kind == last.kind
    assert planner.next_step("change_task_type", last, {last.id}, seed=0).question.kind != last.kind


def test_planner_falls_back_to_nearest_level_when_target_is_used_up():
    bank = QuestionBank()
    last = bank.get("5-sum-a")
    seen = {q.id for q in bank.at_level(5)}
    step = CurriculumPlanner(bank).next_step("increase_difficulty", last, seen, seed=0)
    assert step.question.difficulty == 4 and step.question.id not in seen


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def test_database_upgrade_keeps_pilot_data(tmp_path):
    """The real pilot database predates adaptive mode; upgrading must add the
    new columns without touching existing rows."""
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE sessions (id INTEGER PRIMARY KEY, participant_code TEXT NOT NULL UNIQUE,
            guardian_consent INTEGER NOT NULL, learner_assent INTEGER NOT NULL, consent_recorded_by TEXT NOT NULL,
            notes TEXT, status TEXT NOT NULL DEFAULT 'ready', learner_token TEXT,
            current_index INTEGER NOT NULL DEFAULT 0, created_ms INTEGER NOT NULL, started_ms INTEGER,
            finished_ms INTEGER);
        INSERT INTO sessions (participant_code, guardian_consent, learner_assent, consent_recorded_by, created_ms,
            status, current_index) VALUES ('P01', 1, 1, 'a', 1, 'finished', 15);
    """)
    con.commit()
    con.close()
    db = Database(path)
    db.initialise()
    db.initialise()   # idempotent
    with db.connect() as con:
        row = dict(con.execute("SELECT * FROM sessions").fetchone())
    assert row["participant_code"] == "P01" and row["mode"] == "shadow" and row["status"] == "finished"


# --------------------------------------------------------------------------
# API: researcher, observer, consent
# --------------------------------------------------------------------------

def test_researcher_endpoints_need_key(client):
    assert client.get("/api/researcher/sessions").status_code == 401
    assert client.get("/api/researcher/sessions", headers=KEY).status_code == 200


def test_session_needs_consent_and_assent(client):
    r = client.post("/api/researcher/sessions", headers=KEY,
                    json={"guardian_consent": True, "learner_assent": False, "recorded_by": "AN"})
    assert r.status_code == 400


def test_unknown_mode_is_rejected(client):
    r = client.post("/api/researcher/sessions", headers=KEY, json={
        "guardian_consent": True, "learner_assent": True, "recorded_by": "AN", "mode": "magic"})
    assert r.status_code == 400


def test_observer_key_records_observations_but_sees_no_data(client):
    s = new_session(client)
    listing = client.get("/api/observer/sessions", headers=OBSERVER)
    assert listing.status_code == 200
    assert set(listing.json()[0]) == {"id", "participant_code", "status", "current_index"}
    assert client.post("/api/observer/observations", headers=OBSERVER,
                       json={"session_id": s["id"], "observer": "teacher", "code": "on_task"}).status_code == 200
    assert client.post("/api/observer/observations", headers=OBSERVER,
                       json={"session_id": s["id"], "observer": "teacher", "code": "sleepy"}).status_code == 400
    for path in ("/api/researcher/sessions", f"/api/researcher/sessions/{s['id']}", "/api/researcher/export/answers.csv"):
        assert client.get(path, headers=OBSERVER).status_code == 401


# --------------------------------------------------------------------------
# API: learner, shadow mode (the pilot)
# --------------------------------------------------------------------------

def test_full_shadow_session(client):
    s = new_session(client)
    auth = s["auth"]
    assert s["auth"]["code"] == "P01"
    assert client.post("/api/learner/question", json={"code": "P01", "token": "wrong"}).status_code == 403
    assert client.post("/api/learner/hint", json={**auth, "question_index": 0}).json()["hint"]
    form = QuestionBank().pilot_form()
    seen = run_session(client, auth, lambda q, i: form[i].answer if i % 3 else "0")
    assert [x["question"]["id"] for x in seen] == list(PILOT_FORM_IDS)
    assert client.post("/api/learner/question", json=auth).json() == {"done": True}
    assert client.post("/api/learner/finish", json=auth).json() == {"ok": True}

    d = client.get(f"/api/researcher/sessions/{s['id']}", headers=KEY).json()
    assert len(d["answers"]) == len(d["decisions"]) == SESSION_LENGTH
    assert len(d["checkins"]) == len(CHECKIN_AFTER)
    assert d["answers"][0]["hints_used"] == 1 and d["answers"][1]["correct"] == 1 and d["answers"][0]["correct"] == 0
    assert {x["state_source"] for x in d["decisions"]} == {"provisional_rules_v1"}
    assert "learner_token" not in d["session"]
    assert client.post("/api/learner/join", json={"code": "P01"}).status_code == 409
    assert client.get("/api/researcher/export/decisions.csv", headers=KEY).text.startswith("id,session_id,question_index")


def test_shadow_mode_never_reaches_the_learner(client):
    """In the pilot the DQN's decisions are stored on the server only: no
    response a learner's device receives may contain any part of them."""
    s = new_session(client)
    auth = s["auth"]
    received = []
    for i in range(SESSION_LENGTH):
        received.append(client.post("/api/learner/question", json=auth).json())
        received.append(client.post("/api/learner/hint", json={**auth, "question_index": i}).json())
        received.append(client.post("/api/learner/answer", json={**auth, "question_index": i, "answer": "0",
                                                                 "response_time_ms": 1000, "first_input_ms": 200}).json())
    received.append(client.post("/api/learner/finish", json=auth).json())

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)
        else:
            yield obj

    seen = set(walk(received))
    assert not seen & {"action", "action_name", "action_index", "raw_action_name", "guardrail", "q_values",
                       "state", "decision", "support"}
    assert not seen & set(ACTION_NAMES)
    assert len(client.get(f"/api/researcher/sessions/{s['id']}", headers=KEY).json()["decisions"]) == SESSION_LENGTH


def test_wrong_question_index_and_bad_inputs_are_rejected(client):
    auth = new_session(client)["auth"]
    assert client.post("/api/learner/answer", json={**auth, "question_index": 3, "answer": "1",
                                                     "response_time_ms": 10}).status_code == 409
    assert client.post("/api/learner/answer", json={**auth, "question_index": 0, "answer": "1",
                                                     "response_time_ms": -5}).status_code == 422
    assert client.post("/api/learner/event", json={**auth, "type": "teleport"}).status_code == 400
    assert client.post("/api/learner/checkin", json={**auth, "after_question": 4, "choice": "meh"}).status_code == 400
    assert client.post("/api/learner/finish", json=auth).status_code == 409
    assert client.post("/api/learner/join", json={"code": "P99"}).status_code == 404


# --------------------------------------------------------------------------
# API: learner, adaptive mode (the final MVP)
# --------------------------------------------------------------------------

def test_adaptive_session_adapts_and_explains(client):
    s = new_session(client, mode="adaptive")
    bank = QuestionBank()
    # Answer everything correctly and fast: Stage 1 should see an under-challenged learner.
    seen = run_session(client, s["auth"], lambda q, i: bank.get(q["id"]).answer)
    ids = [x["question"]["id"] for x in seen]
    assert len(ids) == len(set(ids)) == SESSION_LENGTH                # never repeats a question
    assert all("support" in x and x["support"]["message"] for x in seen[1:])
    assert max(bank.get(i).difficulty for i in ids) > 1                # difficulty rose
    d = client.get(f"/api/researcher/sessions/{s['id']}", headers=KEY).json()
    assert {x["state_source"] for x in d["decisions"]} == {"stage1_logreg_v1"}
    assert [a["question_id"] for a in d["answers"]] == ids


def test_adaptive_session_eases_off_for_a_struggling_learner(client):
    s = new_session(client, mode="adaptive")
    seen = run_session(client, s["auth"], lambda q, i: "wrong")
    levels = [QuestionBank().get(x["question"]["id"]).difficulty for x in seen]
    assert max(levels) <= 2, levels
    actions = {x["action_name"] for x in client.get(f"/api/researcher/sessions/{s['id']}", headers=KEY).json()["decisions"]}
    assert not actions & {"advance_topic", "increase_difficulty", "introduce_challenge"}


def test_device_info_is_recorded(client):
    s = new_session(client)
    payload = {"user_agent": "Mozilla/5.0 (X11; CrOS x86_64)", "screen": "1366x768", "touch": False}
    assert client.post("/api/learner/event", json={**s["auth"], "type": "device_info", "payload": payload}).status_code == 200
    events = client.get(f"/api/researcher/sessions/{s['id']}", headers=KEY).json()["events"]
    assert any(e["type"] == "device_info" and "CrOS" in e["payload"] for e in events)


def test_database_can_be_initialised_by_several_processes_at_once(tmp_path):
    """Several uvicorn workers start together; none may fail."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    path = tmp_path / "workers.sqlite3"
    procs = [ctx.Process(target=_init_db, args=(str(path),)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    assert all(p.exitcode == 0 for p in procs)
    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def _init_db(path: str) -> None:
    Database(path).initialise()
