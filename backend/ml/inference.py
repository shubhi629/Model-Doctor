"""Production inference engine (Phase B) built on the trained ModelDoctor artifacts.

The engine never trains or refits anything. It wires together the existing parts:

    raw records -> coerce_records (schema validation, same preprocessing contract as training)
                -> predict_records (Model 1 -> reliability features -> Model 2 -> mitigation)
                -> MitigationEngine.explain (risk level, explanation, reasons)

Artifacts live in one directory per trained model:

    model1.joblib          calibrated Model 1 (includes its fitted preprocessing pipeline)
    extractor.joblib       reliability feature extractor (fit on the train split)
    model2.joblib          Model 2 (absent in confidence_fallback mode)
    engine.joblib          mitigation engine (thresholds tuned on the calibration split)
    artifacts_meta.joblib  classes, feature schema, monitoring baseline, reliability mode
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union
import warnings

import joblib
import pandas as pd

from ml.pipeline import RecordValidationError, coerce_records, predict_records
from ml.reliability.mitigation import WARNING_CATEGORIES, MitigationEngine

MODEL_FILES = ("model1", "extractor", "model2", "engine")
OPTIONAL_MODEL_FILES = {"model2"}
META_FILE = "artifacts_meta.joblib"
REQUIRED_META_KEYS = ("classes", "feature_schema", "target_column", "reliability_mode")
RELIABILITY_MODES = ("model2", "confidence_fallback")

Records = Union[pd.DataFrame, Sequence[Mapping[str, Any]]]


class ModelLoadError(RuntimeError):
    """Raised when trained artifacts are missing, unreadable or inconsistent."""


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(directory: Path, artifacts: Dict[str, Any]) -> None:
    """Write fitted artifacts to `directory` (overwriting a previous run)."""
    directory = Path(directory)
    for key in MODEL_FILES:
        target = directory / f"{key}.joblib"
        if artifacts.get(key) is None:
            target.unlink(missing_ok=True)
        else:
            joblib.dump(artifacts[key], target)
    meta = {k: v for k, v in artifacts.items() if k not in MODEL_FILES}
    joblib.dump(meta, directory / META_FILE)


def artifact_files(directory: Path) -> List[Path]:
    """Every file `save_artifacts` may create (used to clear a previous run)."""
    directory = Path(directory)
    return [directory / f"{k}.joblib" for k in MODEL_FILES] + [directory / META_FILE]


def has_artifacts(directory: Path) -> bool:
    return (Path(directory) / META_FILE).exists()


def load_artifacts(directory: Path) -> Dict[str, Any]:
    """Load and validate artifacts from `directory`; raises ModelLoadError on any problem."""
    directory = Path(directory)
    meta_path = directory / META_FILE
    if not meta_path.exists():
        raise ModelLoadError("No trained model artifacts found.")
    try:
        artifacts = dict(joblib.load(meta_path))
    except Exception as exc:
        raise ModelLoadError(f"Model metadata could not be read ({type(exc).__name__}).") from exc

    for key in MODEL_FILES:
        path = directory / f"{key}.joblib"
        if not path.exists():
            artifacts[key] = None
            continue
        try:
            artifacts[key] = joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(f"Model artifact '{key}' could not be read ({type(exc).__name__}).") from exc
    validate_artifacts(artifacts)
    return artifacts


def validate_artifacts(artifacts: Mapping[str, Any]) -> None:
    """Check that an artifact bundle is complete and internally consistent."""
    missing = [k for k in REQUIRED_META_KEYS if k not in artifacts]
    missing += [k for k in MODEL_FILES if k not in OPTIONAL_MODEL_FILES and artifacts.get(k) is None]
    if missing:
        raise ModelLoadError(f"Model artifacts are incomplete (missing: {', '.join(missing)}).")

    mode = artifacts["reliability_mode"]
    if mode not in RELIABILITY_MODES:
        raise ModelLoadError(f"Unknown reliability mode '{mode}'.")
    if mode == "model2" and artifacts.get("model2") is None:
        raise ModelLoadError("Reliability mode is 'model2' but the Model 2 artifact is missing.")
    if not hasattr(artifacts["model1"], "predict_proba"):
        raise ModelLoadError("Model 1 artifact does not provide predict_proba.")
    if not hasattr(artifacts["extractor"], "transform_with_flags"):
        raise ModelLoadError("Reliability feature extractor artifact is invalid.")
    if not isinstance(artifacts["engine"], MitigationEngine):
        raise ModelLoadError("Mitigation engine artifact is invalid.")
    if artifacts.get("model2") is not None and not hasattr(artifacts["model2"], "predict_trustworthiness"):
        raise ModelLoadError("Model 2 artifact is invalid.")

    classes = artifacts["classes"]
    schema = artifacts["feature_schema"]
    if not isinstance(classes, list) or len(classes) < 2:
        raise ModelLoadError("Model artifacts list fewer than two classes.")
    if not isinstance(schema, list) or not schema or not all(
        isinstance(f, dict) and "name" in f and f.get("type") in ("numeric", "categorical") for f in schema
    ):
        raise ModelLoadError("Model feature schema is invalid.")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _to_frame(records: Records, expected: List[str], allow_extra_columns: bool) -> pd.DataFrame:
    """
    Records as a DataFrame. Mappings are checked one by one, because building a frame
    from records with different keys would silently turn an absent field into a
    missing value.
    """
    if isinstance(records, pd.DataFrame):
        return records
    rows = list(records)
    expected_set = set(expected)
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RecordValidationError(f"Record {i} must be an object of feature values.")
        keys = {str(k) for k in row}
        unknown = sorted(keys - expected_set)
        missing = [name for name in expected if name not in keys]
        if unknown and not allow_extra_columns:
            raise RecordValidationError(f"Record {i}: unknown field(s) {unknown}. Expected features: {expected}")
        if missing:
            raise RecordValidationError(
                f"Record {i}: missing required field(s) {missing}. Send null for a value that is unknown."
            )
    return pd.DataFrame(rows, columns=None if allow_extra_columns else expected)


class InferenceEngine:
    """Scores new records with a fixed, already-trained ModelDoctor bundle."""

    def __init__(self, artifacts: Dict[str, Any]):
        validate_artifacts(artifacts)
        self.artifacts = artifacts

    @classmethod
    def from_directory(cls, directory: Path) -> "InferenceEngine":
        return cls(load_artifacts(directory))

    # -- metadata ---------------------------------------------------------
    @property
    def feature_schema(self) -> List[Dict[str, Any]]:
        return self.artifacts["feature_schema"]

    @property
    def feature_names(self) -> List[str]:
        return [f["name"] for f in self.feature_schema]

    @property
    def classes(self) -> List[str]:
        return self.artifacts["classes"]

    @property
    def target_column(self) -> str:
        return self.artifacts["target_column"]

    @property
    def reliability_mode(self) -> str:
        return self.artifacts["reliability_mode"]

    @property
    def mitigation(self) -> MitigationEngine:
        return self.artifacts["engine"]

    # -- scoring ----------------------------------------------------------
    def validate(self, records: Records, allow_extra_columns: bool = False) -> pd.DataFrame:
        """Coerce raw records to the training schema (raises RecordValidationError)."""
        frame = _to_frame(records, self.feature_names, allow_extra_columns)
        return coerce_records(frame, self.feature_schema, allow_extra_columns=allow_extra_columns)

    def predict(self, records: Records, allow_extra_columns: bool = False) -> List[Dict[str, Any]]:
        """
        Validate and score records. Each result contains the prediction, Model 1's
        confidence, the trustworthiness estimate, the TRUST / REVIEW / REJECT decision,
        categorised warnings, the risk level and a plain-English explanation.
        """
        return self.predict_validated(self.validate(records, allow_extra_columns=allow_extra_columns))

    def predict_validated(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Score records already returned by `validate`."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = predict_records(self.artifacts, X)
        return [self._explain(r) for r in results]

    def predict_one(self, features: Mapping[str, Any]) -> Dict[str, Any]:
        return self.predict([features])[0]

    def _explain(self, result: Dict[str, Any]) -> Dict[str, Any]:
        for w in result["warnings"]:
            w["category"] = WARNING_CATEGORIES.get(w["code"], "other")
        result.update(self.mitigation.explain(
            result["trustworthiness"], result["decision"], result["warnings"], self.reliability_mode,
        ))
        result["reliability_mode"] = self.reliability_mode
        return result
