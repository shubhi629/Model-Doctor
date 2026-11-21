"""Score records with a session's inference engine and log them for monitoring.

Shared by the session-scoped endpoints (/api/predict/{session_id}) and the
versioned deployment endpoints (/api/v1/predict) so both behave identically.
"""

import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core import sessions
from app.core.config import settings
from app.core.errors import unprocessable
from ml.inference import InferenceEngine, Records
from ml.pipeline import RecordValidationError

PREDICTION_LOG = "predictions.jsonl"


def validate(engine: InferenceEngine, records: Records) -> pd.DataFrame:
    """Coerce records to the model's schema; schema problems become 422 responses."""
    try:
        return engine.validate(records)
    except RecordValidationError as exc:
        raise unprocessable(str(exc))


def _json_value(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def log_predictions(
    session_id: str,
    results: List[Dict[str, Any]],
    records: pd.DataFrame,
    source: str,
    request_id: Optional[str] = None,
) -> None:
    """Assign a prediction_id to each result and append it to the session's prediction log."""
    now = sessions.now_iso()
    raw = records.astype(object).where(records.notna(), None).to_dict(orient="records")
    rows = []
    for res, rec in zip(results, raw):
        res["prediction_id"] = uuid.uuid4().hex
        rows.append({
            "prediction_id": res["prediction_id"],
            "timestamp": now,
            "source": source,
            "request_id": request_id,
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "trustworthiness": res["trustworthiness"],
            "decision": res["decision"],
            "warning_codes": [w["code"] for w in res["warnings"]],
            "features": {k: _json_value(v) for k, v in rec.items()} if settings.store_prediction_inputs else None,
        })
    sessions.append_jsonl(session_id, PREDICTION_LOG, rows)


def score_and_log(
    session_id: str,
    engine: InferenceEngine,
    records: Records,
    source: str,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Validate, score and log records (the validated values are what gets logged)."""
    X = validate(engine, records)
    results = engine.predict_validated(X)
    log_predictions(session_id, results, X, source, request_id)
    return results
