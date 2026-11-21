"""Dataset upload and profiling endpoints."""

from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Query, UploadFile

from app.api.deps import df_to_records, parse_csv_bytes, read_upload_limited
from app.core import sessions
from app.core.errors import bad_request, conflict
from app.core.training import is_running
from app.schemas.models import ProfileResponse, SessionSummary, UploadResponse
from ml.data.loader import (
    MIN_REQUIRED_ROWS,
    DatasetValidationError,
    is_continuous_target,
    load_dataset,
)
from ml.data.profiler import profile_dataset

router = APIRouter(prefix="/api", tags=["datasets"])

_TARGET_HINTS = ["target", "label", "class", "default", "outcome", "y", "survived", "churn", "diagnosis"]


def _is_target_candidate(series: pd.Series) -> bool:
    n_unique = series.nunique(dropna=True)
    return 2 <= n_unique <= 50 and not is_continuous_target(series)


def _suggest_target(df: pd.DataFrame) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for hint in _TARGET_HINTS:
        if hint in lowered and _is_target_candidate(df[lowered[hint]]):
            return lowered[hint]
    for col in reversed(list(df.columns)):
        if _is_target_candidate(df[col]):
            return col
    return None


@router.post("/datasets/upload", response_model=UploadResponse, status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV. Creates a session and returns columns plus preview rows."""
    name = file.filename or "dataset.csv"
    if not name.lower().endswith((".csv", ".txt")):
        raise bad_request("Only CSV files are supported.")
    data = await read_upload_limited(file)
    df = parse_csv_bytes(data)
    if df.shape[1] < 2:
        raise bad_request("CSV needs at least two columns (features plus a target).")
    if len(df) == 0:
        raise bad_request("CSV has a header but no data rows.")

    sessions.cleanup_old_sessions()
    columns = [
        {
            "name": str(c),
            "dtype": "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical",
            "missing_count": int(df[c].isna().sum()),
            "unique_count": int(df[c].nunique(dropna=True)),
            "is_target_candidate": bool(_is_target_candidate(df[c])),
        }
        for c in df.columns
    ]
    suggested = _suggest_target(df)
    session_id = sessions.create_session(
        data, name, {"rows": int(len(df)), "columns": [c["name"] for c in columns], "suggested_target": suggested}
    )
    return {
        "session_id": session_id,
        "filename": name,
        "rows": int(len(df)),
        "columns": columns,
        "preview": df_to_records(df.head(10)),
        "suggested_target": suggested,
    }


@router.get("/datasets/{session_id}/profile", response_model=ProfileResponse)
def get_profile(session_id: str, target_column: Optional[str] = Query(None)):
    """Rows, types, missing values, class balance, detected task, and whether the target is valid."""
    meta = sessions.get_meta(session_id)
    target = target_column or meta.get("target_column") or meta.get("suggested_target")
    df = pd.read_csv(sessions.raw_csv_path(session_id))
    df.columns = [str(c).strip() for c in df.columns]
    if not target:
        raise bad_request("No target column given and none could be suggested; pass ?target_column=...")
    if target not in df.columns:
        raise bad_request(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    valid, error = True, None
    try:
        load_dataset(str(sessions.raw_csv_path(session_id)), target)
    except DatasetValidationError as exc:
        valid, error = False, str(exc)

    profile = profile_dataset(df, target)
    if not valid:
        profile["detected_task"] = None
    profile["min_required_rows"] = MIN_REQUIRED_ROWS
    return {"session_id": session_id, "valid": valid, "validation_error": error, **profile}


@router.get("/sessions", response_model=list[SessionSummary], tags=["sessions"])
def list_sessions():
    return sessions.list_sessions()


@router.delete("/sessions/{session_id}", status_code=204, tags=["sessions"])
def delete_session(session_id: str):
    sessions.session_dir(session_id)
    if is_running(session_id):
        raise conflict("Cannot delete a session while it is training.")
    sessions.delete_session(session_id)
