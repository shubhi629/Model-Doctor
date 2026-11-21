"""FastAPI entry point for ModelDoctor backend."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, monitoring, predict, report, train, v1
from app.core import deployment, sessions
from app.core.config import API_VERSION, settings
from app.core.errors import ApiError, register_error_handlers
from app.core.log_config import configure_logging
from app.core.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from app.schemas.models import HealthResponse, ServiceHealthResponse
from ml.inference import ModelLoadError

configure_logging(settings.log_level, settings.log_format)
logger = logging.getLogger("modeldoctor")


@asynccontextmanager
async def lifespan(_: FastAPI):
    sessions.mark_interrupted_sessions()
    removed = sessions.cleanup_old_sessions()
    if removed:
        logger.info("Removed %d expired session(s)", removed)
    # Load the deployed model once at startup so the first request is fast and
    # broken artifacts are reported immediately (the API still starts).
    try:
        session_id, _ = deployment.require_active_engine()
        logger.info("Serving model %s on /api/v1", session_id)
    except (ApiError, ModelLoadError) as exc:
        logger.warning("No model served on /api/v1 at startup: %s", getattr(exc, "detail", exc))
    yield


app = FastAPI(
    title="ModelDoctor API",
    description=(
        "Intelligent Detection and Mitigation of Unreliable ML Predictions.\n\n"
        "Every error response is `{\"detail\", \"code\", \"request_id\"}`; send `X-Request-ID` to set your own ID."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[REQUEST_ID_HEADER],
)
# Added last so it is outermost: every response, including CORS preflights and 413s, gets a request ID.
app.add_middleware(RequestContextMiddleware)

register_error_handlers(app)

for module in (datasets, train, report, predict, monitoring, v1):
    app.include_router(module.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Liveness check for the dashboard: the API process is up."""
    return {"status": "ok"}


@app.get("/health", response_model=ServiceHealthResponse, tags=["health"])
def service_health():
    """Service health including whether the deployed model (/api/v1) is loaded and usable."""
    body = {"status": "ok", "version": API_VERSION, "model_loaded": False, "active_model": None, "detail": None}
    try:
        session_id, _ = deployment.require_active_engine()
        body.update(model_loaded=True, active_model=session_id)
    except ApiError as exc:
        body.update(active_model=deployment.active_session_id(), detail=exc.detail)
        if exc.code != "no_active_model":
            body["status"] = "degraded"
    except ModelLoadError as exc:
        body.update(status="degraded", active_model=deployment.active_session_id(), detail=str(exc))
    return body
