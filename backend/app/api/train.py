"""Training endpoints: start background training and poll status."""

import warnings

from fastapi import APIRouter, BackgroundTasks

from app.core import sessions, training
from app.core.config import settings
from app.core.errors import bad_request, conflict
from app.schemas.models import TrainRequest, TrainStartResponse, TrainStatusResponse
from ml.data.loader import DatasetValidationError, load_dataset
from ml.pipeline import STAGES, TrainingConfig

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("/{session_id}", response_model=TrainStartResponse, status_code=202)
def start_training(session_id: str, body: TrainRequest, background_tasks: BackgroundTasks):
    """Validate the target column, then run the full ModelDoctor setup as a background task."""
    sessions.session_dir(session_id)
    if training.is_running(session_id):
        raise conflict("Training is already running for this session.")

    # Fail fast on invalid targets so the client gets a 400 instead of a failed job
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_dataset(str(sessions.raw_csv_path(session_id)), body.target_column)
    except DatasetValidationError as exc:
        raise bad_request(str(exc))

    if not training.try_claim(session_id):
        raise conflict("Training is already running for this session.")

    config = TrainingConfig(
        target_risk=body.target_risk,
        min_coverage=body.min_coverage,
        force_model=body.force_model,
        force_review_on_anomalies=body.force_review_on_anomalies,
        fast=settings.fast_training if body.fast is None else body.fast,
    )
    sessions.clear_training_outputs(session_id)
    sessions.update_meta(session_id, target_column=body.target_column, config=config.to_dict())
    sessions.set_status(session_id, "pending", message="Training queued.", progress=0.0)
    background_tasks.add_task(training.run_training, session_id, body.target_column, config)
    return {"session_id": session_id, "status": "pending", "message": "Training started."}


@router.get("/{session_id}/status", response_model=TrainStatusResponse)
def training_status(session_id: str):
    status = sessions.get_status(session_id)
    return {**status, "session_id": session_id, "stages": STAGES}
