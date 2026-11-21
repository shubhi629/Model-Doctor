"""Runtime configuration (overridable with environment variables)."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[2]
API_VERSION = "1.1.0"
MB = 1024 * 1024
# Multipart encoding adds boundaries and headers around an uploaded file.
MULTIPART_OVERHEAD_BYTES = 1 * MB


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _env_optional(name: str) -> Optional[str]:
    val = os.getenv(name, "").strip()
    return val or None


@dataclass
class Settings:
    storage_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODELDOCTOR_STORAGE_DIR", str(BACKEND_DIR / "storage")))
    )
    max_upload_mb: float = float(os.getenv("MODELDOCTOR_MAX_UPLOAD_MB", "50"))
    max_json_mb: float = float(os.getenv("MODELDOCTOR_MAX_JSON_MB", "5"))
    session_ttl_hours: float = float(os.getenv("MODELDOCTOR_SESSION_TTL_HOURS", "72"))
    fast_training: bool = _env_bool("MODELDOCTOR_FAST_TRAINING", False)
    max_batch_rows: int = int(os.getenv("MODELDOCTOR_MAX_BATCH_ROWS", "10000"))
    cors_origins: List[str] = field(
        default_factory=lambda: _env_list("MODELDOCTOR_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    )
    # Pin the model served by /api/v1 to one session; otherwise the one activated via the API is used.
    model_session: Optional[str] = field(default_factory=lambda: _env_optional("MODELDOCTOR_MODEL_SESSION"))
    # Keep raw feature values in predictions.jsonl (audit trail); set false to store only outputs.
    store_prediction_inputs: bool = _env_bool("MODELDOCTOR_STORE_INPUTS", True)
    log_level: str = os.getenv("MODELDOCTOR_LOG_LEVEL", "INFO").upper()
    log_format: str = os.getenv("MODELDOCTOR_LOG_FORMAT", "text").lower()  # "text" or "json"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * MB)

    @property
    def max_json_bytes(self) -> int:
        return int(self.max_json_mb * MB)

    @property
    def cors_allow_credentials(self) -> bool:
        # Browsers refuse credentialed requests to a wildcard origin.
        return "*" not in self.cors_origins


settings = Settings()
