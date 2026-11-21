"""End-to-end ModelDoctor training and inference orchestration.

Phase A (train_modeldoctor):
    profiling -> splitting -> model_selection -> reliability_features ->
    reliability_training -> evaluation  (saving is done by the caller)

Phase B (predict_records):
    validate raw record(s) -> Model 1 prediction + confidence -> reliability features
    -> Model 2 trustworthiness -> TRUST / REVIEW / REJECT + warnings

LEAKAGE RULES (see project_spec.md section 4/5):
- Model 1, its preprocessing, its probability calibration and the reliability
  feature extractor are fit on the TRAIN split only.
- Model 2 and the mitigation thresholds use the CALIBRATION split only
  (thresholds from out-of-fold Model 2 predictions).
- The TEST split is touched exactly once, in the evaluation stage, after
  everything has been fixed.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

from ml.data.loader import DatasetValidationError
from ml.data.profiler import profile_dataset
from ml.data.splitter import split_data
from ml.models.selector import select_model
from ml.reliability.evaluation import evaluate_system
from ml.reliability.features import ReliabilityFeatureExtractor
from ml.reliability.mitigation import MitigationEngine, tune_mitigation_thresholds
from ml.reliability.model2 import compute_reliability_table, train_model2
from ml.reliability.monitoring import ANOMALY_CODES

STAGES = [
    "profiling",
    "splitting",
    "model_selection",
    "reliability_features",
    "reliability_training",
    "evaluation",
    "saving",
]

MAX_FORM_CATEGORIES = 200


class RecordValidationError(ValueError):
    """Raised when an inference record does not match the trained feature schema."""


@dataclass
class TrainingConfig:
    """User-tunable training options."""
    target_risk: float = 0.05
    min_coverage: float = 0.30
    force_model: Optional[str] = None
    n_bootstraps: int = 10
    force_review_on_anomalies: bool = False
    fast: bool = False
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_training_frame(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, List[str], List[str], List[Dict[str, str]]]:
    """
    Normalise raw training columns and drop columns with no usable signal.

    - Boolean feature columns become 0/1 integers (numeric).
    - Categorical values are stored as strings (missing stays NaN), so values
      typed later as numbers (e.g. 1 vs "1") match training categories.
    - Constant columns and string identifier columns (100% unique) are dropped.

    Returns (df, numeric_features, categorical_features, dropped_columns).
    """
    df = df.copy()
    numeric: List[str] = []
    categorical: List[str] = []
    dropped: List[Dict[str, str]] = []
    n = len(df)

    for col in df.columns:
        if col == target_column:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            df[col] = series.astype(int)
            series = df[col]

        non_null_unique = series.nunique(dropna=True)
        if non_null_unique <= 1:
            dropped.append({"column": str(col), "reason": "constant or empty column"})
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
        else:
            if non_null_unique == n:
                dropped.append({"column": str(col), "reason": "identifier-like (every value unique)"})
                continue
            df[col] = series.where(series.isna(), series.astype(str))
            categorical.append(col)

    drop_cols = [d["column"] for d in dropped]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if not numeric and not categorical:
        raise DatasetValidationError("No usable feature columns remain after removing constant / ID columns.")

    return df, numeric, categorical, dropped


def build_feature_schema(
    X_train_raw: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[Dict[str, Any]]:
    """Per-feature metadata (from the TRAIN split) used to auto-build the React form."""
    schema: List[Dict[str, Any]] = []
    for col in X_train_raw.columns:
        series = X_train_raw[col]
        entry: Dict[str, Any] = {
            "name": str(col),
            "missing_rate": round(float(series.isna().mean()), 4),
        }
        if col in numeric_features:
            s = pd.to_numeric(series, errors="coerce").dropna()
            is_int = bool(len(s) and np.allclose(s, np.round(s)))
            entry.update({
                "type": "numeric",
                "integer": is_int,
                "min": float(s.min()) if len(s) else None,
                "max": float(s.max()) if len(s) else None,
                "median": float(s.median()) if len(s) else None,
            })
        else:
            counts = series.dropna().astype(str).value_counts()
            entry.update({
                "type": "categorical",
                "categories": sorted(counts.index[:MAX_FORM_CATEGORIES].tolist()),
                "most_frequent": str(counts.index[0]) if len(counts) else None,
                "n_categories": int(len(counts)),
            })
        schema.append(entry)
    return schema


def coerce_records(
    records: pd.DataFrame,
    feature_schema: List[Dict[str, Any]],
    allow_extra_columns: bool = False,
) -> pd.DataFrame:
    """
    Validate and coerce raw inference records against the trained schema.

    - Unknown columns -> error (or silently ignored for batch CSVs when allowed).
    - Absent feature columns -> error (send null / leave blank for a missing value).
    - Non-numeric or infinite values in numeric columns -> error naming the column and rows.
    Missing values (null / NaN / empty string) are ALLOWED: they are a reliability signal.
    """
    expected = [f["name"] for f in feature_schema]
    cols = [str(c) for c in records.columns]
    records = records.copy()
    records.columns = cols

    unknown = [c for c in cols if c not in expected]
    if unknown and not allow_extra_columns:
        raise RecordValidationError(f"Unknown column(s): {unknown}. Expected features: {expected}")
    missing = [c for c in expected if c not in cols]
    if missing:
        raise RecordValidationError(
            f"Missing required field(s): {missing}. Send null for a value that is unknown."
        )

    out = pd.DataFrame(index=records.index)
    for f in feature_schema:
        name = f["name"]
        series = records[name].replace({"": np.nan})
        series = series.where(series.notna(), np.nan)
        if f["type"] == "numeric":
            if series.dtype == bool:
                series = series.astype(int)
            converted = pd.to_numeric(series.map(lambda v: int(v) if isinstance(v, bool) else v),
                                      errors="coerce")
            bad = series.notna() & converted.isna()
            if bad.any():
                bad_rows = [int(i) for i in np.where(bad.values)[0][:5]]
                bad_vals = [str(v) for v in series[bad].head(3).tolist()]
                raise RecordValidationError(
                    f"Field '{name}' must be numeric; got {bad_vals} (row positions {bad_rows})."
                )
            converted = converted.astype(float)
            infinite = np.isinf(converted.to_numpy())
            if infinite.any():
                bad_rows = [int(i) for i in np.where(infinite)[0][:5]]
                raise RecordValidationError(
                    f"Field '{name}' must be a finite number; got infinity (row positions {bad_rows}). "
                    "Send null for an unknown value."
                )
            out[name] = converted
        else:
            out[name] = series.map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    return out


# ---------------------------------------------------------------------------
# Training (Phase A)
# ---------------------------------------------------------------------------

def _noop_stage(stage: str, message: str = "") -> None:
    return None


def train_modeldoctor(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[TrainingConfig] = None,
    on_stage: Callable[[str, str], None] = _noop_stage,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the full ModelDoctor setup on a validated DataFrame.

    Returns
    -------
    (artifacts, report)
        artifacts: fitted objects + metadata needed for inference (joblib-able)
        report: JSON-serializable system report for the dashboard
    """
    cfg = config or TrainingConfig()
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    def stage(name: str, msg: str) -> float:
        on_stage(name, msg)
        return time.perf_counter()

    # 1. Profiling -----------------------------------------------------------
    t0 = stage("profiling", "Profiling dataset")
    profile = profile_dataset(df, target_column)
    df_clean, num_cols, cat_cols, dropped = prepare_training_frame(df, target_column)
    timings["profiling"] = time.perf_counter() - t0

    # 2. Splitting -----------------------------------------------------------
    t0 = stage("splitting", "Stratified 60/20/20 train / calibration / test split")
    splits = split_data(df_clean, target_column, random_state=cfg.random_state)
    num_classes = len(splits.classes)
    timings["splitting"] = time.perf_counter() - t0

    # 3. Model 1 selection (TRAIN split only) -------------------------------
    t0 = stage("model_selection", "Cross-validating candidate models on the train split")
    m1 = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model=cfg.force_model,
        random_state=cfg.random_state,
        fast=cfg.fast,
    )
    model1 = m1["model"]
    timings["model_selection"] = time.perf_counter() - t0

    # 4. Reliability features (extractor fit on TRAIN; applied to calibration) --
    t0 = stage("reliability_features", "Fitting reliability signals and scoring the calibration split")
    extractor = ReliabilityFeatureExtractor(
        n_bootstraps=cfg.n_bootstraps,
        random_state=cfg.random_state,
    )
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=m1["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )
    calib_feats, _ = extractor.transform_with_flags(splits.X_calib_raw, model1)
    calib_pred = np.argmax(model1.predict_proba(splits.X_calib_raw), axis=1)
    calib_correct = (calib_pred == splits.y_calib).astype(int)
    timings["reliability_features"] = time.perf_counter() - t0

    # 5. Model 2 + thresholds (CALIBRATION split only) ----------------------
    t0 = stage("reliability_training", "Training Model 2 and tuning thresholds on calibration out-of-fold scores")
    m2 = train_model2(
        calib_feats,
        calib_correct,
        num_classes=num_classes,
        random_state=cfg.random_state,
    )
    if m2["status"] == "success":
        reliability_mode = "model2"
        model2 = m2["model2"]
        threshold_scores = m2["oof_trustworthiness"]
    else:
        # Fallback: Model 1 never saw the calibration split, so its confidence there
        # is an honest out-of-sample score for tuning thresholds.
        reliability_mode = "confidence_fallback"
        model2 = None
        threshold_scores = calib_feats["max_prob"].to_numpy()

    thresholds = tune_mitigation_thresholds(
        threshold_scores,
        calib_correct,
        target_risk=cfg.target_risk,
        min_coverage=cfg.min_coverage,
    )
    engine = MitigationEngine(
        trust_threshold=thresholds["trust_threshold"],
        reject_threshold=thresholds["reject_threshold"],
        target_risk=thresholds["target_risk"],
        min_coverage=thresholds["min_coverage"],
        target_risk_met=thresholds["target_risk_met"],
        message=thresholds["message"],
        force_review_on_anomalies=cfg.force_review_on_anomalies,
        calibration_coverage=thresholds["calibration_coverage"],
        calibration_selective_risk=thresholds["calibration_selective_risk"],
        num_classes=num_classes,
        band_metadata=thresholds["band_metadata"],
    )
    timings["reliability_training"] = time.perf_counter() - t0

    artifacts: Dict[str, Any] = {
        "model1": model1,
        "extractor": extractor,
        "model2": model2,
        "engine": engine,
        "reliability_mode": reliability_mode,
        "classes": list(splits.classes),
        "num_classes": num_classes,
        "target_column": target_column,
        "feature_schema": build_feature_schema(splits.X_train_raw, num_cols, cat_cols),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
    }

    # 6. Evaluation (TEST split, touched exactly once) ----------------------
    t0 = stage("evaluation", "Evaluating ModelDoctor vs confidence baseline on the held-out test split")
    test_results = predict_records(artifacts, splits.X_test_raw)
    test_pred = np.array([splits.class_mapping[r["prediction"]] for r in test_results])
    test_correct = (test_pred == splits.y_test).astype(int)
    evaluation = evaluate_system(
        correct=test_correct,
        confidence=np.array([r["confidence"] for r in test_results]),
        trustworthiness=np.array([r["trustworthiness"] for r in test_results]),
        decisions=[r["decision"] for r in test_results],
        reliability_mode=reliability_mode,
    )
    timings["evaluation"] = time.perf_counter() - t0

    # Baseline rates of input-anomaly warnings on test data, used by monitoring drift checks
    test_anomaly_rate = float(np.mean([
        any(w["code"] in ANOMALY_CODES for w in r["warnings"]) for r in test_results
    ])) if test_results else 0.0
    artifacts["monitoring_baseline"] = {
        "test_anomaly_rate": round(test_anomaly_rate, 4),
        "test_trust_accuracy": round(1.0 - evaluation["comparison"][2]["selective_risk"], 4),
        "test_coverage": evaluation["comparison"][2]["coverage"],
        "test_accuracy": round(1.0 - evaluation["overall_error_rate"], 4),
    }

    leaderboard = [
        {
            "model": row["model"],
            "score": row.get(m1["primary_metric"]),
            "train_seconds": row.get("train_seconds"),
            **{k: v for k, v in row.items() if k not in ("model",)},
        }
        for row in m1["leaderboard"]
    ]

    model2_effects = None
    if model2 is not None and m2["winner_name"] == "Logistic Regression":
        coefs = model2.pipeline.named_steps["classifier"].coef_[0]
        model2_effects = sorted(
            [{"feature": f, "coefficient": round(float(c), 4)} for f, c in zip(model2.feature_names, coefs)],
            key=lambda d: -abs(d["coefficient"]),
        )

    report: Dict[str, Any] = {
        "task": "classification",
        "task_detail": profile["detected_task"],
        "target_column": target_column,
        "classes": list(splits.classes),
        "selected_model": m1["winner_name"],
        "primary_metric": m1["primary_metric"],
        "calibration_method": m1["calibration_method"],
        "leaderboard": leaderboard,
        "profile": profile,
        "dropped_columns": dropped,
        "features": {"numeric": num_cols, "categorical": cat_cols},
        "splits": splits.get_split_summary(),
        "reliability_mode": reliability_mode,
        "model2": {
            "status": m2["status"],
            "message": m2["message"],
            "winner": m2["winner_name"],
            "feature_names": m2["feature_names"],
            "candidates": m2["candidates_scores"],
            "calibration_ece_oof": m2["ece"] if reliability_mode == "model2" else None,
            "calibration_brier_oof": m2["brier_score"] if reliability_mode == "model2" else None,
            "n_correct_calibration": m2["n_correct"],
            "n_incorrect_calibration": m2["n_incorrect"],
            "reliability_table_oof": m2["reliability_table"] if reliability_mode == "model2"
            else compute_reliability_table(calib_correct, threshold_scores),
            "feature_effects": model2_effects,
        },
        "thresholds": engine.to_dict(),
        **evaluation,
        "config": cfg.to_dict(),
        "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    return artifacts, report


# ---------------------------------------------------------------------------
# Inference (Phase B)
# ---------------------------------------------------------------------------

def predict_records(artifacts: Dict[str, Any], X_raw: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Score raw records (already coerced to the training schema).
    True outcomes are never used here.
    """
    if len(X_raw) == 0:
        return []
    feature_order = [f["name"] for f in artifacts["feature_schema"]]
    X_raw = X_raw[feature_order]

    model1 = artifacts["model1"]
    extractor: ReliabilityFeatureExtractor = artifacts["extractor"]
    engine: MitigationEngine = artifacts["engine"]
    classes: List[str] = artifacts["classes"]

    probs = np.asarray(model1.predict_proba(X_raw), dtype=float)
    pred_idx = np.argmax(probs, axis=1)
    feats, flags = extractor.transform_with_flags(X_raw, model1)

    if artifacts.get("model2") is not None:
        model2 = artifacts["model2"]
        trust = model2.predict_trustworthiness(feats)
        for i, outside in enumerate(model2.support_violations(feats)):
            if outside:
                flags[i] = list(flags[i]) + [{"type": "outside_reliability_support", "features": outside}]
    else:
        trust = feats["max_prob"].to_numpy()

    results: List[Dict[str, Any]] = []
    feat_records = feats.to_dict(orient="records")
    for i in range(len(X_raw)):
        rec_feats = {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                     for k, v in feat_records[i].items()}
        out = engine.evaluate_record(float(trust[i]), flags[i], rec_feats)
        results.append({
            "prediction": classes[int(pred_idx[i])],
            "confidence": round(float(probs[i, pred_idx[i]]), 4),
            "trustworthiness": out["trustworthiness"],
            "decision": out["decision"],
            "warnings": out["warnings"],
            "class_probabilities": {classes[j]: round(float(probs[i, j]), 4) for j in range(len(classes))},
            "reliability_features": {k: round(v, 4) if isinstance(v, float) else v for k, v in rec_feats.items()},
        })
    return results
