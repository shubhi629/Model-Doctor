"""Shared helpers for route modules."""

import io
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import Request, UploadFile

from app.core import sessions
from app.core.config import settings
from app.core.errors import ApiError, bad_request, conflict
from ml.inference import InferenceEngine


async def read_upload_limited(file: UploadFile) -> bytes:
    """Read an uploaded file, enforcing the configured size limit."""
    limit = settings.max_upload_bytes
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ApiError(413, f"File exceeds the {settings.max_upload_mb:g} MB upload limit.", "too_large")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data.strip():
        raise bad_request("Uploaded file is empty.")
    return data


def parse_csv_bytes(data: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
        except Exception as exc:
            raise bad_request(f"Could not parse CSV: {exc}")
    except Exception as exc:
        raise bad_request(f"Could not parse CSV: {exc}")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        dups = sorted(set(df.columns[df.columns.duplicated()]))
        raise bad_request(f"CSV has duplicate column names: {dups}")
    return df


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """JSON-safe records (NaN -> null, numpy -> python)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def require_engine(session_id: str) -> InferenceEngine:
    """Return the session's inference engine or raise 409 if training has not completed."""
    status = sessions.get_status(session_id)
    state = status.get("status")
    if state in ("pending", "running"):
        raise conflict(f"Training is still in progress (stage: {status.get('stage') or 'starting'}).")
    if state == "failed":
        raise conflict(f"Training failed: {status.get('error')}. Start training again.")
    engine = sessions.load_engine(session_id)
    if engine is None:
        raise conflict("Training has not been run for this session yet. POST /api/train/{session_id} first.")
    return engine


def require_artifacts(session_id: str) -> Dict[str, Any]:
    """Fitted artifacts as a dict (see require_engine)."""
    return require_engine(session_id).artifacts


def request_id(request: Request) -> str:
    """The request ID assigned by RequestContextMiddleware."""
    return getattr(request.state, "request_id", "-")


def normalize_label(value: Any) -> str:
    """Make outcome labels comparable with training class names (which are strings)."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()
