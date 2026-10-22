"""Builds the application: wires settings, storage, models, services and
routes together (the only place where concrete classes are chosen)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pilot.adaptation import CurriculumPlanner
from pilot.api.routes import learner_router, observer_router, researcher_router, shared_router
from pilot.api.security import KeyGuard
from pilot.config import Settings
from pilot.core.tasks import QuestionBank
from pilot.labelling import RuleBasedLabeler, Stage1Labeler
from pilot.policy import DQNPolicy, Shield
from pilot.services.errors import ServiceError
from pilot.services.learning import LearningService
from pilot.services.modes import AdaptiveMode, ShadowMode
from pilot.services.observation import ObservationService
from pilot.services.sessions import SessionService
from pilot.storage import Database


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    db = Database(settings.db_path)
    bank = QuestionBank()
    policy = DQNPolicy(settings.policy_version)
    modes = {
        "shadow": ShadowMode(bank, RuleBasedLabeler()),
        "adaptive": AdaptiveMode(bank, Stage1Labeler(), CurriculumPlanner(bank)),
    }
    keys = KeyGuard(settings.researcher_key, settings.observer_key)

    @asynccontextmanager
    async def lifespan(_app):
        db.initialise()
        policy.warm_up()   # load the DQN now, not on the first learner's answer
        yield

    app = FastAPI(title="SensAI", lifespan=lifespan)

    @app.exception_handler(ServiceError)
    async def service_error(_request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    app.include_router(shared_router())
    app.include_router(researcher_router(SessionService(db), keys))
    app.include_router(observer_router(ObservationService(db), keys))
    app.include_router(learner_router(LearningService(db, bank, modes, Shield(policy))))

    if settings.web_dist.exists():
        dist = settings.web_dist.resolve()
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def web_app(path: str):
            target = (dist / path).resolve()
            if path and target.is_relative_to(dist) and target.is_file():
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    return app
