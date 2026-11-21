"""Background training task: runs the full ModelDoctor setup for one session."""

import logging
import threading
import warnings
from typing import Dict

from app.core import deployment, sessions
from ml.data.loader import DatasetValidationError, load_dataset
from ml.pipeline import STAGES, TrainingConfig, train_modeldoctor

logger = logging.getLogger("modeldoctor")

_running: Dict[str, bool] = {}
_running_lock = threading.Lock()


def is_running(session_id: str) -> bool:
    with _running_lock:
        return _running.get(session_id, False)


def try_claim(session_id: str) -> bool:
    """Atomically mark a session as training; False if a run is already in progress."""
    with _running_lock:
        if _running.get(session_id):
            return False
        _running[session_id] = True
        return True


def _release(session_id: str) -> None:
    with _running_lock:
        _running.pop(session_id, None)


def run_training(session_id: str, target_column: str, config: TrainingConfig) -> None:
    """Executed in a worker thread by FastAPI BackgroundTasks."""

    def on_stage(stage: str, message: str) -> None:
        progress = round(STAGES.index(stage) / len(STAGES), 3) if stage in STAGES else None
        sessions.set_status(session_id, "running", stage=stage, message=message, progress=progress)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_dataset(str(sessions.raw_csv_path(session_id)), target_column)
            artifacts, report = train_modeldoctor(df, target_column, config, on_stage=on_stage)
            on_stage("saving", "Saving models, thresholds and report")
            report["session_id"] = session_id
            sessions.save_training_outputs(session_id, artifacts, report)
            sessions.reset_monitoring(session_id)
        sessions.set_status(session_id, "completed", stage="saving", message="Training complete.", progress=1.0)
        try:
            deployment.activate_if_none(session_id)
        except Exception:  # serving the model is optional; training itself succeeded
            logger.exception("Could not activate session %s for /api/v1", session_id)
    except DatasetValidationError as exc:
        sessions.set_status(session_id, "failed", error=str(exc))
    except Exception as exc:  # never leak a traceback to the client
        logger.exception("Training failed for session %s", session_id)
        stage = sessions.get_status(session_id).get("stage")
        sessions.set_status(
            session_id, "failed", stage=stage,
            error=f"Training failed during stage '{stage}': {type(exc).__name__}: {exc}",
        )
    finally:
        _release(session_id)
