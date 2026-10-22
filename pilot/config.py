"""Settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE = Path(__file__).parent


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path(os.environ.get("PILOT_DB", PACKAGE / "data" / "pilot.sqlite3")))
    researcher_key: str = field(default_factory=lambda: os.environ.get("PILOT_RESEARCHER_KEY", "sensai-dev"))
    observer_key: str = field(default_factory=lambda: os.environ.get("PILOT_OBSERVER_KEY", "sensai-observe"))
    policy_version: str = field(default_factory=lambda: os.environ.get("SENSAI_POLICY", "v2"))
    web_dist: Path = PACKAGE / "web" / "dist"
