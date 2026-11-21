"""Automatic Model Selection and Calibration for Model 1.

CRITICAL LEAKAGE RULES:
- `select_model` receives ONLY the train split (X_train_raw, y_train).
- Calibration and test splits must NEVER be passed to this module.
- Model 1 probability calibration is performed strictly INSIDE the train split
  using internal 5-fold cross-validation. The external calibration split is
  strictly reserved for Model 2.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ml.models.candidates import get_candidate_models


def evaluate_candidate_cv(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
) -> Dict[str, Any]:
    """
    Evaluate a candidate model across pre-computed cross-validation folds.
    Catches and logs failures without crashing the overall selection process.
    """
    fold_accuracies = []
    fold_weighted_f1 = []
    fold_macro_f1 = []
    fold_roc_auc = []
    fold_log_loss = []

    start_time = time.perf_counter()

    for train_idx, val_idx in cv_splits:
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

        # Clone to guarantee fresh state in each fold
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_tr, y_tr)

        preds = fold_pipeline.predict(X_va)
        probs = fold_pipeline.predict_proba(X_va)

        # 1. Accuracy
        fold_accuracies.append(accuracy_score(y_va, preds))

        # 2. Weighted F1
        fold_weighted_f1.append(f1_score(y_va, preds, average="weighted", zero_division=0))

        # 3. Macro F1
        fold_macro_f1.append(f1_score(y_va, preds, average="macro", zero_division=0))

        # 4. ROC-AUC
        try:
            if num_classes == 2:
                # Use positive class probability
                auc = roc_auc_score(y_va, probs[:, 1])
            else:
                # One-vs-Rest weighted ROC-AUC for multiclass
                auc = roc_auc_score(y_va, probs, multi_class="ovr", average="weighted")
            fold_roc_auc.append(auc)
        except Exception:
            fold_roc_auc.append(0.5)

        # 5. Log Loss
        try:
            loss = log_loss(y_va, probs, labels=list(range(num_classes)))
            fold_log_loss.append(loss)
        except Exception:
            fold_log_loss.append(np.nan)

    elapsed = time.perf_counter() - start_time

    return {
        "model": name,
        "status": "success",
        "train_seconds": round(elapsed, 3),
        "accuracy_mean": round(float(np.mean(fold_accuracies)), 4),
        "accuracy_std": round(float(np.std(fold_accuracies)), 4),
        "weighted_f1_mean": round(float(np.mean(fold_weighted_f1)), 4),
        "weighted_f1_std": round(float(np.std(fold_weighted_f1)), 4),
        "macro_f1_mean": round(float(np.mean(fold_macro_f1)), 4),
        "macro_f1_std": round(float(np.std(fold_macro_f1)), 4),
        "roc_auc_mean": round(float(np.mean(fold_roc_auc)), 4),
        "roc_auc_std": round(float(np.std(fold_roc_auc)), 4),
        "log_loss_mean": round(float(np.nanmean(fold_log_loss)), 4),
        "log_loss_std": round(float(np.nanstd(fold_log_loss)), 4),
        "error_message": None,
    }


def select_model(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    primary_metric: str = "auto",
    force_model: Optional[str] = None,
    handle_imbalance: bool = True,
    random_state: int = 42,
    fast: bool = False,
) -> Dict[str, Any]:
    """
    Run automated model selection over candidate algorithms on the Train split,
    evaluate on shared cross-validation folds, pick the best algorithm, retrain,
    and calibrate its probabilities using internal CV inside the train split.

    Parameters
    ----------
    X_train_raw : pd.DataFrame
        Raw training feature data. Calibration and test data MUST NOT be passed.
    y_train : np.ndarray
        Encoded training target array.
    numeric_features : list[str]
        List of numeric column names.
    categorical_features : list[str]
        List of categorical column names.
    primary_metric : str, default="auto"
        Metric used to rank candidates.
        "auto" defaults to "roc_auc_mean" for binary classification and
        "weighted_f1_mean" for multiclass classification.
    force_model : str, optional
        Algorithm name to force-select (e.g. "Random Forest"), skipping
        metric-based winner selection.
    handle_imbalance : bool, default=True
        Whether to enable balanced class weighting.
    random_state : int, default=42
        Reproducibility seed.
    fast : bool, default=False
        Use smaller tree ensembles (quick demos and tests).

    Returns
    -------
    dict
        - "model": fitted CalibratedClassifierCV wrapping the winning pipeline
        - "winner_name": str name of winning algorithm
        - "primary_metric": str metric key used for ranking
        - "leaderboard": list[dict] of sorted candidates
        - "calibration_method": str ("isotonic" or "sigmoid")
    """
    classes, counts = np.unique(y_train, return_counts=True)
    num_classes = len(classes)
    class_counts_dict = {int(c): int(cnt) for c, cnt in zip(classes, counts)}

    # Build candidate pool
    candidates = get_candidate_models(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        num_classes=num_classes,
        class_counts=class_counts_dict,
        handle_imbalance=handle_imbalance,
        random_state=random_state,
        **({"fast": True} if fast else {}),
    )

    # Validate force_model if provided
    if force_model is not None:
        if force_model not in candidates:
            raise ValueError(
                f"Unknown candidate '{force_model}'. Available candidates: {list(candidates.keys())}"
            )

    # Resolve primary metric
    # RATIONALE: Accuracy is highly misleading on imbalanced datasets because predicting the
    # majority class achieves high accuracy while failing completely on the minority class.
    # For binary classification, ROC-AUC measures ranking and discrimination independent of
    # decision thresholds. For multiclass, weighted F1 accounts for class imbalance across all classes.
    if primary_metric == "auto":
        metric_key = "roc_auc_mean" if num_classes == 2 else "weighted_f1_mean"
    elif not primary_metric.endswith("_mean"):
        metric_key = f"{primary_metric}_mean"
    else:
        metric_key = primary_metric

    # Generate the EXACT SAME StratifiedKFold splits for all candidates
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_splits = list(skf.split(X_train_raw, y_train))

    leaderboard: List[Dict[str, Any]] = []

    # If force_model is provided, skip cross-validating other candidates
    candidates_to_evaluate = (
        {force_model: candidates[force_model]}
        if force_model is not None
        else candidates
    )

    # Evaluate candidate(s) across identical folds
    for name, pipeline in candidates_to_evaluate.items():
        try:
            result = evaluate_candidate_cv(
                name=name,
                pipeline=pipeline,
                X_train=X_train_raw,
                y_train=y_train,
                cv_splits=cv_splits,
                num_classes=num_classes,
            )
            if force_model is not None:
                result["status"] = "forced"
                result["forced"] = True
            leaderboard.append(result)
        except Exception as exc:
            # Fault-tolerant: log failure and continue
            leaderboard.append({
                "model": name,
                "status": "failed",
                "forced": bool(force_model is not None),
                "train_seconds": 0.0,
                "accuracy_mean": 0.0,
                "accuracy_std": 0.0,
                "weighted_f1_mean": 0.0,
                "weighted_f1_std": 0.0,
                "macro_f1_mean": 0.0,
                "macro_f1_std": 0.0,
                "roc_auc_mean": 0.0,
                "roc_auc_std": 0.0,
                "log_loss_mean": 999.0,
                "log_loss_std": 0.0,
                "error_message": str(exc),
            })

    # Sort leaderboard: ascending for log loss, descending for all other metrics
    is_loss_metric = "log_loss" in metric_key
    default_sort_val = 9999.0 if is_loss_metric else -9999.0
    leaderboard.sort(
        key=lambda item: item.get(metric_key, default_sort_val),
        reverse=not is_loss_metric,
    )

    # Determine winner
    if force_model is not None:
        winner_name = force_model
    else:
        # First non-failed candidate
        successful_candidates = [item for item in leaderboard if item["status"] == "success"]
        if not successful_candidates:
            raise RuntimeError("All candidate models failed during cross-validation.")
        winner_name = successful_candidates[0]["model"]

    winner_pipeline = candidates[winner_name]

    # Probability calibration INSIDE train split only
    # RULE: Isotonic regression is non-parametric and can correct arbitrary mis-calibrations,
    # but requires at least 1,000 samples to avoid overfitting and stepwise probabilities.
    # For smaller datasets (<1,000 train samples), Platt scaling (sigmoid) is used to prevent overfitting.
    train_size = len(X_train_raw)
    calib_method = "isotonic" if train_size >= 1000 else "sigmoid"

    # Wrap the winner pipeline with CalibratedClassifierCV using internal 5-fold CV.
    # This ensures the pipeline (including ColumnTransformer) is re-fit inside the calibration folds.
    calibrated_model = CalibratedClassifierCV(
        estimator=winner_pipeline,
        cv=5,
        method=calib_method,
    )
    calibrated_model.fit(X_train_raw, y_train)

    return {
        "model": calibrated_model,
        "base_pipeline": clone(winner_pipeline),
        "winner_name": winner_name,
        "primary_metric": metric_key,
        "leaderboard": leaderboard,
        "calibration_method": calib_method,
        "num_classes": num_classes,
    }
