"""HTTP layer: request schemas, access keys and routes. Routes call services
and translate service errors into HTTP responses; no logic lives here."""

from pilot.api.app import create_app

__all__ = ["create_app"]
