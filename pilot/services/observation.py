"""What an observer (researcher or class teacher) can do: see which learners
are working, and record what they see. No answers, scores or decisions."""

from __future__ import annotations

from pilot.core.states import OBSERVER_CODES
from pilot.services.errors import InvalidRequest, NotFound
from pilot.storage import Database, Repositories


class ObservationService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def open_sessions(self) -> list[dict]:
        with self.db.connect() as con:
            return [{k: s[k] for k in ("id", "participant_code", "status", "current_index")}
                    for s in Repositories(con).sessions.all(open_only=True)]

    def record(self, session_id: int, observer: str, code: str, client_ms: int | None) -> None:
        if code not in OBSERVER_CODES:
            raise InvalidRequest(f"Unknown code {code!r}.")
        with self.db.connect() as con:
            repos = Repositories(con)
            if not repos.sessions.by_id(session_id):
                raise NotFound("No such session.")
            repos.observations.add(session_id, observer.strip(), code, client_ms)
