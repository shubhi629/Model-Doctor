"""Feedback (true outcomes) and monitoring endpoints (Phase C)."""

from fastapi import APIRouter, Query

from app.api.deps import normalize_label, require_artifacts
from app.core import sessions
from app.schemas.models import FeedbackRequest, FeedbackResponse, MonitoringResponse
from ml.reliability.monitoring import compute_monitoring

router = APIRouter(prefix="/api", tags=["monitoring"])


@router.post("/feedback/{session_id}", response_model=FeedbackResponse)
def submit_feedback(session_id: str, body: FeedbackRequest):
    """Record actual outcomes for previously served predictions (by prediction_id)."""
    artifacts = require_artifacts(session_id)
    classes = set(artifacts["classes"])
    known_ids = {p["prediction_id"] for p in sessions.read_jsonl(session_id, "predictions.jsonl")}

    accepted, rejected = [], []
    now = sessions.now_iso()
    for item in body.items:
        actual = normalize_label(item.actual)
        if item.prediction_id not in known_ids:
            rejected.append({"prediction_id": item.prediction_id, "reason": "unknown prediction_id"})
        elif actual not in classes:
            rejected.append({
                "prediction_id": item.prediction_id,
                "reason": f"'{actual}' is not a known class (expected one of {sorted(classes)})",
            })
        else:
            accepted.append({"prediction_id": item.prediction_id, "actual": actual, "timestamp": now})
    sessions.append_jsonl(session_id, "feedback.jsonl", accepted)
    labeled = {f["prediction_id"] for f in sessions.read_jsonl(session_id, "feedback.jsonl")}
    return {"accepted": len(accepted), "rejected": rejected, "labeled_total": len(labeled & known_ids)}


@router.get("/monitoring/{session_id}", response_model=MonitoringResponse)
def get_monitoring(
    session_id: str,
    recent: int = Query(50, ge=0, le=1000),
    awaiting_only: bool = Query(False, description="Only list predictions that have no outcome yet"),
):
    """Coverage, decision distribution, observed vs estimated reliability, drift / degradation alerts."""
    artifacts = require_artifacts(session_id)
    predictions = sessions.read_jsonl(session_id, "predictions.jsonl")
    feedback = sessions.read_jsonl(session_id, "feedback.jsonl")
    summary = compute_monitoring(predictions, feedback, artifacts.get("monitoring_baseline"))

    actual_by_id = {f["prediction_id"]: f["actual"] for f in feedback}
    listed = [p for p in predictions if p["prediction_id"] not in actual_by_id] if awaiting_only else predictions
    recent_rows = []
    for p in reversed(listed[-recent:] if recent else []):
        row = {k: v for k, v in p.items() if k != "features"}
        row["actual"] = actual_by_id.get(p["prediction_id"])
        row["features"] = p.get("features")
        recent_rows.append(row)
    return {"session_id": session_id, **summary, "recent_predictions": recent_rows}
