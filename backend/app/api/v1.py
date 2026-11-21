"""Versioned deployment API: serves one active, already-trained model.

    GET  /api/v1/model/info       metadata, thresholds, feature schema, held-out test metrics
    PUT  /api/v1/model/active     choose which trained session is served
    POST /api/v1/predict          score one record
    POST /api/v1/predict/batch    score up to MODELDOCTOR_MAX_BATCH_ROWS records (JSON)

Every prediction is logged to the served session, so it appears on that session's
Overview and Monitoring pages.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Request

from app.api.deps import request_id
from app.core import deployment, sessions
from app.core.config import settings
from app.core.errors import ApiError
from app.core.predictions import score_and_log
from app.schemas.models import (
    ErrorResponse,
    V1ActivateRequest,
    V1BatchRequest,
    V1BatchResponse,
    V1ModelInfo,
    V1PredictRequest,
    V1PredictResponse,
)

router = APIRouter(prefix="/api/v1", tags=["v1"])

DECISIONS = ("TRUST", "REVIEW", "REJECT")
_ERRORS = {
    status: {"model": ErrorResponse, "description": text}
    for status, text in {
        400: "Malformed JSON",
        413: "Request body or batch too large",
        422: "Input does not match the model's feature schema",
        503: "No model deployed, or its artifacts cannot be loaded",
    }.items()
}
TEST_METRIC_KEYS = ("overall_error_rate", "comparison", "auroc", "auprc", "verdict")


def _public(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction_id": result["prediction_id"],
        "prediction": result["prediction"],
        "model_confidence": result["confidence"],
        "trustworthiness": result["trustworthiness"],
        "decision": result["decision"],
        "risk_level": result["risk_level"],
        "explanation": result["explanation"],
        "reasons": result["reasons"],
        "warnings": result["warnings"],
        "class_probabilities": result["class_probabilities"],
    }


@router.get("/model/info", response_model=V1ModelInfo, responses={503: _ERRORS[503]})
def model_info():
    """Describe the served model: classes, features, thresholds and its held-out test-split metrics."""
    session_id, engine = deployment.require_active_engine()
    report = sessions.get_report(session_id) or {}
    calibration = (report.get("calibration") or {}).get("modeldoctor") or {}
    test_metrics = {k: report[k] for k in TEST_METRIC_KEYS if k in report}
    if calibration:
        test_metrics["modeldoctor_ece"] = calibration.get("ece")
    return {
        "model_id": session_id,
        "pinned": deployment.is_pinned(),
        "target_column": engine.target_column,
        "task": report.get("task_detail") or report.get("task") or "classification",
        "classes": engine.classes,
        "selected_model": report.get("selected_model", "unknown"),
        "calibration_method": report.get("calibration_method"),
        "reliability_mode": engine.reliability_mode,
        "reliability_model": (report.get("model2") or {}).get("winner"),
        "thresholds": engine.mitigation.to_dict(),
        "features": engine.feature_schema,
        "test_metrics": test_metrics,
        "trained_at": sessions.get_status(session_id).get("updated_at"),
        "max_batch_rows": settings.max_batch_rows,
    }


@router.put("/model/active", response_model=V1ModelInfo, responses={
    404: {"model": ErrorResponse, "description": "Unknown session"},
    409: {"model": ErrorResponse, "description": "Training not completed, or the model is pinned"},
    503: _ERRORS[503],
})
def activate_model(body: V1ActivateRequest):
    """Serve a different trained session from /api/v1 (refused when pinned by MODELDOCTOR_MODEL_SESSION)."""
    deployment.activate(body.session_id)
    return model_info()


@router.post("/predict", response_model=V1PredictResponse,
             responses={k: _ERRORS[k] for k in (400, 413, 422, 503)})
def predict(body: V1PredictRequest, request: Request):
    """Score one record with the deployed model."""
    session_id, engine = deployment.require_active_engine()
    rid = request_id(request)
    result = score_and_log(session_id, engine, [body.features], "api", rid)[0]
    return {**_public(result), "request_id": rid, "model_id": session_id, "reliability_mode": engine.reliability_mode}


@router.post("/predict/batch", response_model=V1BatchResponse,
             responses={k: _ERRORS[k] for k in (400, 413, 422, 503)})
def predict_batch(body: V1BatchRequest, request: Request):
    """
    Score several records in one call. The whole batch is validated first: if any record
    is invalid, nothing is scored and the error names the field and row position.
    """
    if len(body.records) > settings.max_batch_rows:
        raise ApiError(
            413, f"Batch has {len(body.records)} records; the limit is {settings.max_batch_rows}.", "too_large",
        )
    session_id, engine = deployment.require_active_engine()
    rid = request_id(request)
    results: List[Dict[str, Any]] = score_and_log(session_id, engine, body.records, "api_batch", rid)
    summary = {d: sum(1 for r in results if r["decision"] == d) for d in DECISIONS}
    return {
        "request_id": rid,
        "model_id": session_id,
        "reliability_mode": engine.reliability_mode,
        "count": len(results),
        "summary": summary,
        "results": [_public(r) for r in results],
    }
