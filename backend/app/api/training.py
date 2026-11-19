import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas.training import (
    TrainingRequest,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from app.core.training_service import train_session

router = APIRouter(prefix="/api/train", tags=["Training"])

STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage"


@router.post("/{session_id}", response_model=TrainingStartResponse)
def start_training(
    session_id: str,
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    session_dir = STORAGE_DIR / session_id

    if not (session_dir / "dataset.csv").exists():
        raise HTTPException(404, "Dataset session not found.")

    status_path = session_dir / "status.json"

    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") in ("pending", "running"):
            raise HTTPException(409, "Training already in progress.")

    status_path.write_text(json.dumps({
        "session_id": session_id,
        "status": "pending",
        "stage": "queued",
        "progress": 0,
        "error": None,
    }))

    background_tasks.add_task(
        train_session,
        session_id,
        request.target_column,
    )

    return {
        "session_id": session_id,
        "status": "pending",
        "message": "Training started successfully.",
    }


@router.get("/{session_id}/status", response_model=TrainingStatusResponse)
def get_training_status(session_id: str):

    status_path = STORAGE_DIR / session_id / "status.json"

    if not status_path.exists():
        raise HTTPException(404, "Training session not found.")

    return json.loads(status_path.read_text())