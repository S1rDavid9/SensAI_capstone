"""Access keys. The researcher key opens everything; the observer key (for
the class teacher) only opens the observer routes."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException


class KeyGuard:
    def __init__(self, researcher_key: str, observer_key: str) -> None:
        self._researcher = researcher_key
        self._observer = observer_key

    def researcher(self, x_researcher_key: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_researcher_key, self._researcher):
            raise HTTPException(401, "Researcher key required.")

    def observer(self, x_researcher_key: str = Header(default="")) -> None:
        if not (secrets.compare_digest(x_researcher_key, self._researcher)
                or secrets.compare_digest(x_researcher_key, self._observer)):
            raise HTTPException(401, "Observer key required.")
