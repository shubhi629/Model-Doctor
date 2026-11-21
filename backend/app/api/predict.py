"""Session-scoped prediction endpoints (Phase B: true outcome unknown)."""

import pandas as pd
from fastapi import APIRouter, File, Request, UploadFile

from app.api.deps import normalize_label, parse_csv_bytes, read_upload_limited, request_id, require_engine
from app.core import sessions
from app.core.config import settings
from app.core.errors import bad_request, unprocessable
from app.core.predictions import score_and_log
from app.schemas.models import BatchPredictResponse, PredictRequest, PredictResponse

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/{session_id}", response_model=PredictResponse)
def predict_single(session_id: str, body: PredictRequest, request: Request):
    """Score one record: prediction, confidence, trustworthiness, TRUST/REVIEW/REJECT, warnings and explanation."""
    engine = require_engine(session_id)
    if not body.features:
        raise unprocessable("'features' must contain at least one field.")
    records = pd.DataFrame([body.features])
    return score_and_log(session_id, engine, records, "single", request_id(request))[0]


@router.post("/{session_id}/batch", response_model=BatchPredictResponse)
async def predict_batch(session_id: str, request: Request, file: UploadFile = File(...)):
    """
    Score a CSV of records. Extra columns are ignored. If the CSV also contains the
    target column, those values are recorded as true outcomes (feedback) for monitoring;
    they are never used to make the predictions.
    """
    engine = require_engine(session_id)
    df = parse_csv_bytes(await read_upload_limited(file))
    if len(df) == 0:
        raise bad_request("Batch CSV has no rows.")
    if len(df) > settings.max_batch_rows:
        raise bad_request(f"Batch CSV has {len(df)} rows; the limit is {settings.max_batch_rows}.")

    target = engine.target_column
    actuals = df[target].tolist() if target in df.columns else None
    expected = set(engine.feature_names)
    ignored = [c for c in df.columns if c not in expected]
    features_df = df[[c for c in df.columns if c in expected]]
    missing = [name for name in engine.feature_names if name not in features_df.columns]
    if missing:
        raise unprocessable(f"Batch CSV is missing required column(s): {missing}")

    results = score_and_log(session_id, engine, features_df, "batch", request_id(request))

    labels_recorded = 0
    if actuals is not None:
        now = sessions.now_iso()
        fb = [
            {"prediction_id": r["prediction_id"], "actual": normalize_label(a), "timestamp": now}
            for r, a in zip(results, actuals) if a is not None and not pd.isna(a)
        ]
        sessions.append_jsonl(session_id, "feedback.jsonl", fb)
        labels_recorded = len(fb)
        for r, a in zip(results, actuals):
            r["actual"] = None if a is None or pd.isna(a) else normalize_label(a)

    summary = {d: sum(1 for r in results if r["decision"] == d) for d in ["TRUST", "REVIEW", "REJECT"]}
    summary["labels_recorded"] = labels_recorded
    for r in results:
        r.pop("reliability_features", None)
    return {
        "session_id": session_id,
        "count": len(results),
        "ignored_columns": ignored,
        "summary": summary,
        "results": results,
    }
