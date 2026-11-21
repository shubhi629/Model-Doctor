"""Model 2 (The Reliability Doctor) for ModelDoctor.

Trains a secondary meta-model on the CALIBRATION split to estimate
the empirical probability of correctness:
    trustworthiness = P(prediction is correct | reliability features)

CRITICAL RULES:
- Trained on CALIBRATION split features only.
- True labels are used ONLY to construct the binary target `correct = (pred == y_true)`.
  True labels must NEVER be included as an input feature.
- `predicted_class` is excluded by default so Model 2 does not memorize class priors.
- For binary classification, `entropy` and `margin` are excluded by default because
  they are exact mathematical functions of `max_prob` (redundant/collinear).
- Model 2 uses unweighted fitting (no class_weight="balanced") so output probabilities
  reflect the true empirical base rate.
- Gradient Boosting is calibrated with CalibratedClassifierCV(method="sigmoid").
- Produces Out-Of-Fold (OOF) probabilities of correctness for threshold tuning.
- If fewer than 30 incorrect or correct predictions exist, flags 'insufficient_signal'.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Relative tolerance when deciding whether a value is outside Model 2's training range.
SUPPORT_TOLERANCE = 1e-6

# Direction in which each reliability signal indicates MORE risk (+1: higher is riskier,
# -1: lower is riskier). Leaving the calibration range in the safe direction (e.g. an input
# that is more typical than any calibration row) is clipped but not treated as unsupported.
# Signals not listed count in both directions.
RISK_DIRECTION = {
    "max_prob": -1,
    "margin": -1,
    "entropy": +1,
    "bootstrap_disagreement": +1,
    "bootstrap_std": +1,
    "knn_distance": +1,
    "knn_distance_percentile": +1,
    "isolation_score": +1,
    "missing_count": +1,
    "missing_fraction": +1,
    "out_of_range_count": +1,
    "unseen_category_count": +1,
}


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) across n equal-width bins in [0, 1].
    
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true_arr)

    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob_arr >= low) & (y_prob_arr <= high if i == n_bins - 1 else y_prob_arr < high)
        n_bin = int(np.sum(mask))
        if n_bin > 0:
            bin_acc = float(np.mean(y_true_arr[mask]))
            bin_conf = float(np.mean(y_prob_arr[mask]))
            ece += (n_bin / n_total) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probability predictions."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return round(float(np.mean((y_prob_arr - y_true_arr) ** 2)), 4)


def compute_reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    """Compute equal-count (quantile) reliability binning table."""
    df_bins = pd.DataFrame({
        "prob": np.asarray(y_prob, dtype=float),
        "true": np.asarray(y_true, dtype=int),
    })

    try:
        df_bins["bin"] = pd.qcut(df_bins["prob"], q=n_bins, duplicates="drop")
    except Exception:
        df_bins["bin"] = pd.cut(df_bins["prob"], bins=n_bins)

    table = []
    for b_label, grp in df_bins.groupby("bin", observed=False):
        mean_p = float(grp["prob"].mean())
        obs_acc = float(grp["true"].mean())
        table.append({
            "bin": str(b_label),
            "count": int(len(grp)),
            "mean_predicted": round(mean_p, 4),
            "observed_accuracy": round(obs_acc, 4),
            "gap": round(abs(mean_p - obs_acc), 4),
        })
    return table


class Model2Detector(BaseEstimator, ClassifierMixin):
    """
    Wrapper for the final trained Model 2, exposing predict_trustworthiness.
    Can be persisted and loaded with joblib.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: List[str],
        winner_name: str,
        cv_auroc: float,
        confidence_ref_auroc: float,
        ece: float = 0.0,
        brier_score: float = 0.0,
    ):
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.winner_name = winner_name
        self.cv_auroc = cv_auroc
        self.confidence_ref_auroc = confidence_ref_auroc
        self.ece = ece
        self.brier_score = brier_score

    def predict_trustworthiness(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Estimate the probability of correctness for each input record.

        Parameters
        ----------
        features_df : pd.DataFrame
            Reliability features DataFrame.

        Returns
        -------
        np.ndarray
            1D array of trustworthiness scores P(correct) in [0, 1].
        """
        if not isinstance(features_df, pd.DataFrame):
            features_df = pd.DataFrame(features_df)

        X = self._clip_to_support(features_df[self.feature_names])
        probs = self.pipeline.predict_proba(X)

        if probs.shape[1] == 2:
            return np.clip(probs[:, 1], 0.0, 1.0)
        else:
            return np.clip(probs[:, 0], 0.0, 1.0)

    # ------------------------------------------------------------------
    # Support of the training data (CALIBRATION split reliability features)
    # ------------------------------------------------------------------
    # Model 2 is fitted on a finite range of each reliability signal. Outside that
    # range a linear model extrapolates without limit (e.g. a positive weight on
    # knn_distance would make ever more distant inputs look ever more trustworthy).
    # Inputs are therefore clipped to the observed range, and rows that needed
    # clipping are reported so the mitigation layer can refuse to TRUST them.

    def set_support(self, X: pd.DataFrame) -> "Model2Detector":
        """Record the per-feature [min, max] of the data Model 2 was fitted on."""
        X = X[self.feature_names].astype(float)
        self.feature_min_ = {c: float(X[c].min()) for c in self.feature_names}
        self.feature_max_ = {c: float(X[c].max()) for c in self.feature_names}
        return self

    def _has_support(self) -> bool:
        # Artifacts saved before support tracking existed do not have these attributes.
        return hasattr(self, "feature_min_") and hasattr(self, "feature_max_")

    def _clip_to_support(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._has_support():
            return X
        lo = pd.Series(self.feature_min_)[self.feature_names]
        hi = pd.Series(self.feature_max_)[self.feature_names]
        return X.astype(float).clip(lower=lo, upper=hi, axis=1)

    def support_violations(self, features_df: pd.DataFrame) -> List[List[str]]:
        """
        For each row, the reliability features that lie outside the range seen when
        Model 2 was trained, in their risky direction (see RISK_DIRECTION); an empty list
        means the estimate is supported by training data. Non-finite values always count.
        """
        if not self._has_support():
            return [[] for _ in range(len(features_df))]
        X = features_df[self.feature_names].astype(float)
        out_mask = pd.DataFrame(False, index=X.index, columns=self.feature_names)
        for c in self.feature_names:
            lo, hi = self.feature_min_[c], self.feature_max_[c]
            tol = SUPPORT_TOLERANCE * (max(abs(lo), abs(hi)) + 1.0)
            col = X[c].to_numpy()
            direction = RISK_DIRECTION.get(c, 0)
            below = (col < lo - tol) if direction <= 0 else np.zeros(len(col), dtype=bool)
            above = (col > hi + tol) if direction >= 0 else np.zeros(len(col), dtype=bool)
            out_mask[c] = ~np.isfinite(col) | below | above
        return [[c for c in self.feature_names if row[c]] for _, row in out_mask.iterrows()]


def evaluate_oof_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, float]:
    """
    Compute Out-Of-Fold (OOF) predicted probabilities and CV AUROC for detecting errors.
    Target: 1 = correct, 0 = incorrect.
    Error AUROC: ranking incorrect predictions higher (using P(incorrect) = 1 - P(correct)).
    """
    n_samples = len(X)
    oof_probs = np.zeros(n_samples, dtype=np.float64)

    for train_idx, val_idx in cv_splits:
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_tr, y_tr)

        p = fold_pipe.predict_proba(X_va)
        if p.shape[1] == 2:
            oof_probs[val_idx] = p[:, 1]
        else:
            oof_probs[val_idx] = p[:, 0]

    # AUROC for identifying incorrect predictions
    # y=0 is incorrect, y=1 is correct.
    # Score for incorrect: 1 - oof_probs
    is_incorrect = (1 - y).astype(int)
    error_risk = 1.0 - oof_probs
    try:
        auroc = float(roc_auc_score(is_incorrect, error_risk))
    except Exception:
        auroc = 0.5

    return oof_probs, auroc


def train_model2(
    features_df: pd.DataFrame,
    y_correct: np.ndarray,
    num_classes: int = 2,
    exclude_predicted_class: bool = True,
    exclude_redundant_binary_features: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and select Model 2 on CALIBRATION split reliability features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Reliability feature DataFrame (must NOT contain true labels).
    y_correct : np.ndarray
        Binary correctness target: 1 = correct, 0 = incorrect.
    num_classes : int, default=2
        Number of classes in the primary classification task.
    exclude_predicted_class : bool, default=True
        Whether to exclude `predicted_class` from features to prevent prior memorization.
    exclude_redundant_binary_features : bool, default=True
        For binary classification, exclude entropy and margin since they are exact
        deterministic functions of max_prob (margin = 2*p - 1, entropy = -p*log(p) - (1-p)*log(1-p)).
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    dict
        - "status": "success" or "insufficient_signal"
        - "model2": Model2Detector instance (if success)
        - "winner_name": str
        - "oof_trustworthiness": np.ndarray of OOF probabilities
        - "candidates_scores": dict with CV AUROC, ECE, Brier per candidate
        - "confidence_ref_auroc": float AUROC of max_prob alone
        - "feature_names": list of feature names used
        - "ece": float ECE of winning model
        - "brier_score": float Brier score of winning model
        - "reliability_table": list[dict]
        - "oof_candidates": dict of OOF predictions per candidate
        - "n_correct": int
        - "n_incorrect": int
        - "message": str
    """
    n_samples = len(features_df)
    y_arr = np.asarray(y_correct, dtype=int)

    n_correct = int(np.sum(y_arr == 1))
    n_incorrect = int(np.sum(y_arr == 0))

    # Signal Check: at least 30 samples in each class
    if n_correct < 30 or n_incorrect < 30:
        return {
            "status": "insufficient_signal",
            "model2": None,
            "winner_name": None,
            "oof_trustworthiness": None,
            "candidates_scores": {},
            "confidence_ref_auroc": None,
            "feature_names": [],
            "ece": 0.0,
            "brier_score": 0.0,
            "reliability_table": [],
            "oof_candidates": {},
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "message": (
                f"Calibration split has insufficient error signal ({n_incorrect} incorrect, "
                f"{n_correct} correct; minimum 30 required in each class). "
                "Primary model is either near-perfect or dataset is too small."
            ),
        }

    # Select features (guarantee true labels are NEVER present)
    feature_cols = [c for c in features_df.columns if c not in ["target", "label", "y_true", "y"]]
    if exclude_predicted_class and "predicted_class" in feature_cols:
        feature_cols.remove("predicted_class")

    # For binary classification, remove redundant mathematical functions of max_prob
    if num_classes == 2 and exclude_redundant_binary_features:
        # Documented rationale: For binary targets, margin = 2*max_prob - 1 (linear)
        # and entropy is a symmetric concave function of max_prob. Keeping them causes
        # multicollinearity and feature redundancy for Model 2.
        for redundant in ["entropy", "margin"]:
            if redundant in feature_cols:
                feature_cols.remove(redundant)

    X = features_df[feature_cols].copy()

    # Stratified 5-fold CV splits on calibration split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_splits = list(skf.split(X, y_arr))

    # Reference: Confidence-only model (using max_prob alone, unweighted)
    conf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=random_state)),
    ])
    X_conf = features_df[["max_prob"]].copy()
    oof_conf_probs, conf_ref_auroc = evaluate_oof_cv(conf_pipe, X_conf, y_arr, cv_splits)

    # Candidate 1: Logistic Regression (UNWEIGHTED for honest probability calibration)
    # RATIONALE: We remove class_weight="balanced" because we want true empirical probabilities,
    # not probabilities artificially shifted toward a synthetic 50/50 prior.
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )),
    ])
    oof_lr_probs, lr_auroc = evaluate_oof_cv(lr_pipe, X, y_arr, cv_splits)

    # Candidate 2: Small regularized Gradient Boosting (calibrated with sigmoid)
    # RATIONALE: Wrapped in CalibratedClassifierCV with method="sigmoid" because the calibration
    # split is relatively small and isotonic regression tends to overfit / produce stepwise artifacts.
    gb_base = HistGradientBoostingClassifier(
        max_iter=50,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=random_state,
    )
    gb_calibrated = CalibratedClassifierCV(estimator=gb_base, cv=3, method="sigmoid")
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", gb_calibrated),
    ])
    oof_gb_probs, gb_auroc = evaluate_oof_cv(gb_pipe, X, y_arr, cv_splits)

    # Compute ECE and Brier scores for all candidates
    candidates_scores = {
        "Logistic Regression": {
            "cv_auroc": round(lr_auroc, 4),
            "ece": compute_ece(y_arr, oof_lr_probs),
            "brier": compute_brier_score(y_arr, oof_lr_probs),
        },
        "Gradient Boosting": {
            "cv_auroc": round(gb_auroc, 4),
            "ece": compute_ece(y_arr, oof_gb_probs),
            "brier": compute_brier_score(y_arr, oof_gb_probs),
        },
        "Confidence Baseline (max_prob only)": {
            "cv_auroc": round(conf_ref_auroc, 4),
            "ece": compute_ece(y_arr, oof_conf_probs),
            "brier": compute_brier_score(y_arr, oof_conf_probs),
        },
    }

    if gb_auroc > lr_auroc:
        winner_name = "Gradient Boosting"
        winner_pipe = gb_pipe
        oof_trustworthiness = oof_gb_probs
        winner_auroc = gb_auroc
    else:
        winner_name = "Logistic Regression"
        winner_pipe = lr_pipe
        oof_trustworthiness = oof_lr_probs
        winner_auroc = lr_auroc

    # Final fit on the entire calibration split
    final_pipeline = clone(winner_pipe)
    final_pipeline.fit(X, y_arr)

    final_ece = compute_ece(y_arr, oof_trustworthiness)
    final_brier = compute_brier_score(y_arr, oof_trustworthiness)
    rel_table = compute_reliability_table(y_arr, oof_trustworthiness, n_bins=10)

    detector = Model2Detector(
        pipeline=final_pipeline,
        feature_names=feature_cols,
        winner_name=winner_name,
        cv_auroc=winner_auroc,
        confidence_ref_auroc=conf_ref_auroc,
        ece=final_ece,
        brier_score=final_brier,
    )
    detector.set_support(X)

    return {
        "status": "success",
        "model2": detector,
        "winner_name": winner_name,
        "oof_trustworthiness": oof_trustworthiness,
        "candidates_scores": candidates_scores,
        "confidence_ref_auroc": round(conf_ref_auroc, 4),
        "feature_names": feature_cols,
        "ece": final_ece,
        "brier_score": final_brier,
        "reliability_table": rel_table,
        "oof_candidates": {
            "Logistic Regression": oof_lr_probs,
            "Gradient Boosting": oof_gb_probs,
            "Confidence Baseline": oof_conf_probs,
        },
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "message": None,
    }
