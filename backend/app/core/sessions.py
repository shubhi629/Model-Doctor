"""File-based session manager.

Layout under storage/{session_id}/:
    raw.csv              uploaded dataset (exactly as received)
    meta.json            filename, created_at, columns, target, config
    status.json          pending / running (stage) / completed / failed
    model1.joblib        calibrated Model 1 (includes its fitted preprocessing pipeline)
    extractor.joblib     reliability feature extractor (fit on train split)
    model2.joblib        Model 2 (absent when Model 2 had insufficient signal)
    engine.joblib        mitigation engine (thresholds)
    thresholds.json      threshold configuration (human-readable copy)
    artifacts_meta.joblib classes, schema, monitoring baseline, reliability mode
    report.json          system report
    schema.json          input schema for the prediction form
    predictions.jsonl    every prediction served (for monitoring)
    feedback.jsonl       true outcomes submitted later
"""

import json
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.errors import not_found
from ml.inference import InferenceEngine, artifact_files, has_artifacts, save_artifacts

_SESSION_RE = re.compile(r"^[a-f0-9]{32}$")
_lock = threading.RLock()
_engine_cache: Dict[str, InferenceEngine] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def storage_root() -> Path:
    root = Path(settings.storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_dir(session_id: str, must_exist: bool = True) -> Path:
    if not _SESSION_RE.match(session_id or ""):
        raise not_found(f"Session '{session_id}' not found.")
    path = storage_root() / session_id
    if must_exist and not path.is_dir():
        raise not_found(f"Session '{session_id}' not found.")
    return path


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_file(path: Path, default: Any = None) -> Any:
    """Read a JSON file under the storage root (default when absent)."""
    return _read_json(path, default)


def write_json_file(path: Path, data: Any) -> None:
    """Atomically write a JSON file under the storage root."""
    with _lock:
        _write_json(path, data)


def create_session(raw_bytes: bytes, filename: str, meta: Dict[str, Any]) -> str:
    session_id = uuid.uuid4().hex
    path = session_dir(session_id, must_exist=False)
    path.mkdir(parents=True)
    (path / "raw.csv").write_bytes(raw_bytes)
    _write_json(path / "meta.json", {"session_id": session_id, "filename": filename, "created_at": now_iso(), **meta})
    set_status(session_id, "uploaded", message="Dataset uploaded; choose a target column and start training.")
    return session_id


def raw_csv_path(session_id: str) -> Path:
    return session_dir(session_id) / "raw.csv"


def get_meta(session_id: str) -> Dict[str, Any]:
    return _read_json(session_dir(session_id) / "meta.json", {})


def update_meta(session_id: str, **fields: Any) -> Dict[str, Any]:
    with _lock:
        meta = get_meta(session_id)
        meta.update(fields)
        _write_json(session_dir(session_id) / "meta.json", meta)
        return meta


def set_status(
    session_id: str,
    status: str,
    stage: Optional[str] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    progress: Optional[float] = None,
) -> Dict[str, Any]:
    with _lock:
        path = session_dir(session_id) / "status.json"
        prev = _read_json(path, {})
        data = {
            "session_id": session_id,
            "status": status,
            "stage": stage,
            "message": message,
            "error": error,
            "progress": progress,
            "started_at": now_iso() if status == "pending" else prev.get("started_at"),
            "updated_at": now_iso(),
        }
        _write_json(path, data)
        return data


def get_status(session_id: str) -> Dict[str, Any]:
    return _read_json(session_dir(session_id) / "status.json", {"session_id": session_id, "status": "uploaded"})


def save_training_outputs(session_id: str, artifacts: Dict[str, Any], report: Dict[str, Any]) -> None:
    path = session_dir(session_id)
    save_artifacts(path, artifacts)
    _write_json(path / "thresholds.json", artifacts["engine"].to_dict())
    _write_json(path / "schema.json", artifacts["feature_schema"])
    _write_json(path / "report.json", report)
    with _lock:
        _engine_cache.pop(session_id, None)


def clear_training_outputs(session_id: str) -> None:
    """Remove a previous training run's outputs before retraining."""
    path = session_dir(session_id)
    for file in artifact_files(path) + [path / n for n in ("thresholds.json", "schema.json", "report.json")]:
        file.unlink(missing_ok=True)
    with _lock:
        _engine_cache.pop(session_id, None)


def get_report(session_id: str) -> Optional[Dict[str, Any]]:
    return _read_json(session_dir(session_id) / "report.json")


def load_engine(session_id: str) -> Optional[InferenceEngine]:
    """
    Load (and cache) the inference engine for a session; None if training has not completed.
    Raises ml.inference.ModelLoadError if artifacts exist but are corrupt or inconsistent.
    """
    with _lock:
        if session_id in _engine_cache:
            return _engine_cache[session_id]
    path = session_dir(session_id)
    if not has_artifacts(path):
        return None
    engine = InferenceEngine.from_directory(path)
    with _lock:
        _engine_cache[session_id] = engine
    return engine


def load_artifacts(session_id: str) -> Optional[Dict[str, Any]]:
    """Fitted artifacts as a dict (see load_engine)."""
    engine = load_engine(session_id)
    return engine.artifacts if engine is not None else None


def append_jsonl(session_id: str, name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with _lock:
        with open(session_dir(session_id) / name, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")


def read_jsonl(session_id: str, name: str) -> List[Dict[str, Any]]:
    path = session_dir(session_id) / name
    if not path.exists():
        return []
    rows = []
    with _lock:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def reset_monitoring(session_id: str) -> None:
    path = session_dir(session_id)
    with _lock:
        (path / "predictions.jsonl").unlink(missing_ok=True)
        (path / "feedback.jsonl").unlink(missing_ok=True)


def list_sessions() -> List[Dict[str, Any]]:
    out = []
    for p in storage_root().iterdir():
        if p.is_dir() and _SESSION_RE.match(p.name):
            meta = _read_json(p / "meta.json", {})
            status = _read_json(p / "status.json", {})
            out.append({
                "session_id": p.name,
                "filename": meta.get("filename"),
                "created_at": meta.get("created_at"),
                "target_column": meta.get("target_column"),
                "rows": meta.get("rows"),
                "status": status.get("status", "uploaded"),
                "stage": status.get("stage"),
            })
    out.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return out


def delete_session(session_id: str) -> None:
    path = session_dir(session_id)
    with _lock:
        _engine_cache.pop(session_id, None)
    shutil.rmtree(path, ignore_errors=True)


def cleanup_old_sessions(max_age_hours: Optional[float] = None) -> int:
    """Delete sessions whose directory has not been modified within the TTL. Returns count removed."""
    ttl = settings.session_ttl_hours if max_age_hours is None else max_age_hours
    if ttl <= 0:
        return 0
    cutoff = time.time() - ttl * 3600
    removed = 0
    for p in storage_root().iterdir():
        if not (p.is_dir() and _SESSION_RE.match(p.name)):
            continue
        status = _read_json(p / "status.json", {})
        if status.get("status") == "running":
            continue
        latest = max((f.stat().st_mtime for f in p.iterdir()), default=p.stat().st_mtime)
        if latest < cutoff:
            shutil.rmtree(p, ignore_errors=True)
            with _lock:
                _engine_cache.pop(p.name, None)
            removed += 1
    return removed


def mark_interrupted_sessions() -> None:
    """On startup, sessions left 'running' by a previous process can never finish; mark them failed."""
    for p in storage_root().iterdir():
        if p.is_dir() and _SESSION_RE.match(p.name):
            status = _read_json(p / "status.json", {})
            if status.get("status") in ("running", "pending"):
                set_status(p.name, "failed", stage=status.get("stage"),
                           error="Training was interrupted because the server restarted. Start training again.")
