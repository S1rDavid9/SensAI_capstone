"""Tests for the pilot data-collection server."""

import pytest
from fastapi.testclient import TestClient

from pilot.labeler import AnswerTelemetry, build_observation, provisional_state
from pilot.tasks import FORM, N_QUESTIONS

KEY = {"X-Researcher-Key": "sensai-dev"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from pilot import db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "pilot.sqlite3")
    from pilot.server import app

    with TestClient(app) as c:
        yield c


def _telemetry(**kw):
    base = dict(response_time_s=10.0, first_input_s=2.0, correct=True, hints_used=0,
                difficulty=2, recent_errors=[False], questions_at_level=1)
    base.update(kw)
    return AnswerTelemetry(**base)


def test_form_is_fixed_and_rises_in_difficulty():
    assert N_QUESTIONS == 15
    assert [q.difficulty for q in FORM] == sorted(q.difficulty for q in FORM)
    assert len({q.prompt for q in FORM}) == N_QUESTIONS
    assert all("answer" not in q.public() for q in FORM)
    assert {q.domain for q in FORM} == {"arithmetic", "pattern"}
    assert len({q.kind for q in FORM}) >= 6
    for q in FORM:
        assert q.is_correct(q.answer)
        if q.choices:
            assert q.answer in q.choices and len(set(q.choices)) == len(q.choices)


def test_number_answers_tolerate_spaces_and_commas():
    q = next(q for q in FORM if q.answer == "850")
    assert q.is_correct(" 850 ") and q.is_correct("8,50".replace(",", "")) and not q.is_correct("85")


def test_guardrails_override_and_record():
    import numpy as np
    from pilot.policy import shadow_decision

    obs = np.zeros(9, dtype=np.float32)
    obs[3] = 1.0
    d = shadow_decision(obs, "on_task", difficulty=1)
    assert d["action_name"] != "decrease_difficulty"
    if d["raw_action_name"] == "decrease_difficulty":
        assert d["guardrail"]
    assert shadow_decision(obs, "stuck", difficulty=3)["action_name"] != "advance_topic"


@pytest.mark.parametrize("kw, expected", [
    (dict(), "on_task"),
    (dict(first_input_s=None), "stuck"),
    (dict(first_input_s=25.0), "stuck"),
    (dict(correct=False, response_time_s=2.0, recent_errors=[True]), "rushing"),
    (dict(correct=False, response_time_s=20.0, recent_errors=[True, True, False]), "struggling"),
])
def test_provisional_state(kw, expected):
    assert provisional_state(_telemetry(**kw)) == expected


def test_observation_matches_dqn_input():
    obs = build_observation("rushing", _telemetry(difficulty=5, hints_used=10, response_time_s=80))
    assert obs.shape == (9,)
    assert list(obs[:4]) == [0, 1, 0, 0]
    assert obs.min() >= 0 and obs.max() <= 1
    assert obs[7] == 1.0


def test_researcher_endpoints_need_key(client):
    assert client.get("/api/researcher/sessions").status_code == 401
    assert client.get("/api/researcher/sessions", headers=KEY).status_code == 200


def test_session_needs_consent_and_assent(client):
    r = client.post("/api/researcher/sessions", headers=KEY,
                    json={"guardian_consent": True, "learner_assent": False, "recorded_by": "AN"})
    assert r.status_code == 400


def test_full_learner_session(client):
    s = client.post("/api/researcher/sessions", headers=KEY,
                    json={"guardian_consent": True, "learner_assent": True, "recorded_by": "AN"}).json()
    assert s["participant_code"] == "P01"

    j = client.post("/api/learner/join", json={"code": "p01"}).json()
    auth = {"code": "P01", "token": j["token"]}
    assert client.post("/api/learner/question", json={"code": "P01", "token": "wrong"}).status_code == 403

    for i in range(N_QUESTIONS):
        q = client.post("/api/learner/question", json=auth).json()["question"]
        assert q["index"] == i
        if i == 0:
            assert client.post("/api/learner/hint", json={**auth, "question_index": 0}).json()["hint"]
        answer = FORM[i].answer if i % 3 else "0"
        r = client.post("/api/learner/answer", json={**auth, "question_index": i, "answer": answer,
                                                     "response_time_ms": 9000, "first_input_ms": 1500}).json()
        assert r["correct"] == (answer == FORM[i].answer)
        if r["checkin_due"]:
            client.post("/api/learner/checkin", json={**auth, "after_question": i, "choice": "good"})

    assert client.post("/api/learner/question", json=auth).json() == {"done": True}
    client.post("/api/observer/observations", headers=KEY,
                json={"session_id": s["id"], "observer": "teacher", "code": "on_task"})
    assert client.post("/api/learner/finish", json=auth).json() == {"ok": True}

    d = client.get(f"/api/researcher/sessions/{s['id']}", headers=KEY).json()
    assert len(d["answers"]) == N_QUESTIONS
    assert len(d["decisions"]) == N_QUESTIONS
    assert len(d["checkins"]) == 3
    assert d["answers"][0]["hints_used"] == 1
    assert d["session"]["status"] == "finished"
    assert "learner_token" not in d["session"]
    assert client.post("/api/learner/join", json={"code": "P01"}).status_code == 409

    csv_text = client.get("/api/researcher/export/decisions.csv", headers=KEY).text
    assert csv_text.startswith("id,session_id,question_index,state")


def test_shadow_mode_never_reaches_the_learner(client):
    """The DQN's decisions are stored on the server only: no response a
    learner's device receives may contain any part of them."""
    from environment.custom_env import ACTION_NAMES

    client.post("/api/researcher/sessions", headers=KEY,
                json={"guardian_consent": True, "learner_assent": True, "recorded_by": "AN"})
    auth = {"code": "P01", "token": client.post("/api/learner/join", json={"code": "P01"}).json()["token"]}
    leaked_keys = {"action", "action_name", "action_index", "raw_action_name", "guardrail", "q_values", "state", "decision"}

    received = []
    for q in FORM:
        received.append(client.post("/api/learner/question", json=auth).json())
        received.append(client.post("/api/learner/hint", json={**auth, "question_index": q.index}).json())
        received.append(client.post("/api/learner/answer", json={**auth, "question_index": q.index, "answer": "0",
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
    assert not seen & leaked_keys
    assert not seen & set(ACTION_NAMES)
    # ...while the server did record a decision for every answer.
    sid = client.get("/api/researcher/sessions", headers=KEY).json()[0]["id"]
    assert len(client.get(f"/api/researcher/sessions/{sid}", headers=KEY).json()["decisions"]) == len(FORM)
