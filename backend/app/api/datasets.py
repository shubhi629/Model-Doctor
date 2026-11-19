import uuid
import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.dataset import (
    DatasetUploadResponse,
    DatasetProfileResponse
)
from ml.data.loader import load_dataset, DatasetValidationError
from ml.data.profiler import profile_dataset

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    content = await file.read()

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "File size exceeds 50 MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Invalid CSV file.")

    if len(df) < 500:
        raise HTTPException(422, "Dataset must contain at least 500 rows.")

    session_id = str(uuid.uuid4())
    session_dir = STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    (session_dir / "dataset.csv").write_bytes(content)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "columns": df.columns.tolist(),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "row_count": len(df)
    }


@router.get("/{session_id}/profile", response_model=DatasetProfileResponse)
def get_dataset_profile(session_id: str, target_column: str):

    dataset_path = STORAGE_DIR / session_id / "dataset.csv"

    if not dataset_path.exists():
        raise HTTPException(404, "Dataset session not found.")

    try:
        df = load_dataset(str(dataset_path), target_column)
        profile = profile_dataset(df, target_column)
        return profile

    except DatasetValidationError as exc:
        raise HTTPException(422, str(exc))