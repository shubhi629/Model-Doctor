"""System report and input-schema endpoints."""

from fastapi import APIRouter

from app.api.deps import require_artifacts
from app.core import sessions
from app.core.errors import conflict
from app.schemas.models import ReportResponse, SchemaResponse

router = APIRouter(prefix="/api", tags=["report"])


@router.get("/report/{session_id}", response_model=ReportResponse)
def get_report(session_id: str):
    """Leaderboard, selected model, ModelDoctor vs confidence baseline metrics and chart data."""
    require_artifacts(session_id)
    report = sessions.get_report(session_id)
    if report is None:
        raise conflict("Report is not available yet.")
    report["session_id"] = session_id
    return report


@router.get("/schema/{session_id}", response_model=SchemaResponse)
def get_schema(session_id: str):
    """Feature names, types, categories and train ranges for auto-building the input form."""
    artifacts = require_artifacts(session_id)
    return {
        "session_id": session_id,
        "target_column": artifacts["target_column"],
        "classes": artifacts["classes"],
        "features": artifacts["feature_schema"],
    }
