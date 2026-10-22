"""What the researcher can do: create sessions after consent, follow them
live, close them and export the data."""

from __future__ import annotations

import csv
import io

from pilot.services.errors import InvalidRequest, NotFound
from pilot.storage import Database, Repositories
from pilot.storage.database import TABLES

MODES = ("shadow", "adaptive")


class SessionService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, *, guardian_consent: bool, learner_assent: bool, recorded_by: str, notes: str,
               mode: str = "shadow") -> dict:
        if not (guardian_consent and learner_assent):
            raise InvalidRequest("A session needs both signed guardian consent and the learner's assent.")
        if mode not in MODES:
            raise InvalidRequest(f"Unknown mode {mode!r}.")
        with self.db.connect() as con:
            s = Repositories(con).sessions.create(recorded_by=recorded_by.strip(), notes=notes.strip(), mode=mode)
            s.pop("learner_token", None)
            return s

    def summaries(self) -> list[dict]:
        with self.db.connect() as con:
            repos = Repositories(con)
            out = []
            for s in repos.sessions.all():
                s.pop("learner_token", None)
                s["answered"], s["correct"] = repos.answers.counts(s["id"])
                s["observations"] = repos.observations.count(s["id"])
                s["latest_decision"] = repos.decisions.latest(s["id"])
                out.append(s)
            return out

    def detail(self, session_id: int) -> dict:
        with self.db.connect() as con:
            repos = Repositories(con)
            s = repos.sessions.by_id(session_id)
            if not s:
                raise NotFound("No such session.")
            s.pop("learner_token", None)
            return {"session": s,
                    "answers": repos.answers.for_session(session_id),
                    "checkins": repos.checkins.for_session(session_id),
                    "observations": repos.observations.for_session(session_id),
                    "decisions": repos.decisions.for_session(session_id),
                    "events": repos.events.for_session(session_id)}

    def close(self, session_id: int) -> None:
        with self.db.connect() as con:
            Repositories(con).sessions.close(session_id)

    def export_csv(self, table: str) -> str:
        if table not in TABLES:
            raise NotFound(f"Unknown table. Choose one of {list(TABLES)}.")
        with self.db.connect() as con:
            rows = Repositories(con).export(table)
        buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return buf.getvalue()
