"""Which trained session is served by the versioned /api/v1 endpoints.

Resolution order:
1. MODELDOCTOR_MODEL_SESSION (pinned by the operator; cannot be changed through the API)
2. storage/active_model.json (set with PUT /api/v1/model/active, or automatically
   for the first model that finishes training when none is active yet)
"""

from typing import Optional, Tuple

from app.core import sessions
from app.core.config import settings
from app.core.errors import ApiError, conflict, service_unavailable
from ml.inference import InferenceEngine

ACTIVE_FILE = "active_model.json"


def _active_path():
    return sessions.storage_root() / ACTIVE_FILE


def is_pinned() -> bool:
    return settings.model_session is not None


def active_session_id() -> Optional[str]:
    if settings.model_session:
        return settings.model_session
    data = sessions.read_json_file(_active_path(), {})
    return data.get("session_id")


def activate(session_id: str) -> None:
    """Serve `session_id` from /api/v1. The session must have finished training."""
    if is_pinned():
        raise ApiError(
            409, "The served model is pinned by MODELDOCTOR_MODEL_SESSION and cannot be changed via the API.",
            "model_pinned",
        )
    sessions.session_dir(session_id)  # 404 if unknown
    if sessions.get_status(session_id).get("status") != "completed":
        raise conflict("Only a session whose training has completed can be activated.")
    sessions.load_engine(session_id)  # fail now (503) rather than on the first prediction
    sessions.write_json_file(_active_path(), {"session_id": session_id, "activated_at": sessions.now_iso()})


def activate_if_none(session_id: str) -> None:
    """Make a freshly trained model active when nothing is served yet."""
    if is_pinned():
        return
    current = active_session_id()
    if current is None or not (sessions.storage_root() / current).is_dir():
        sessions.write_json_file(_active_path(), {"session_id": session_id, "activated_at": sessions.now_iso()})


def require_active_engine() -> Tuple[str, InferenceEngine]:
    """(session_id, engine) of the served model, or a 503 explaining why there is none."""
    session_id = active_session_id()
    if session_id is None:
        raise service_unavailable(
            "No model is deployed. Train a model, then activate it with PUT /api/v1/model/active.",
            "no_active_model",
        )
    try:
        sessions.session_dir(session_id)
    except ApiError:
        raise service_unavailable(f"The deployed model '{session_id}' no longer exists.", "no_active_model")
    engine = sessions.load_engine(session_id)
    if engine is None:
        raise service_unavailable(f"The deployed model '{session_id}' has no trained artifacts.", "no_active_model")
    return session_id, engine
