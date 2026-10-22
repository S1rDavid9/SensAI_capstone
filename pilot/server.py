"""Entry point:

    uv run uvicorn pilot.server:app --host 0.0.0.0 --port 8001

See pilot/README.md for the environment variables (database path and keys).
"""

from pilot.api import create_app

app = create_app()
