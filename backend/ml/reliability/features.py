"""Model-agnostic reliability feature extraction for ModelDoctor.

All features extracted here are model-agnostic and work with any candidate algorithm.
CRITICAL LEAKAGE RULES:
- `fit()` learns strictly from the TRAIN split (X_train_raw, y_train).
  The calibration and test splits must NEVER be passed to `fit()`.
- `transform()` accepts ONLY raw input features (X_raw) and Model 1 (model1).
  It must NOT accept or use true labels in any way.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from ml.data.preprocessing import build_preprocessor


class ReliabilityFeatureExtractor:
    """
    Extracts model-agnostic reliability signals from primary model outputs,
    input data geometry, bootstrap disagreement, missingness, and domain boundaries.
    """

    def __init__(
        self,
        n_bootstraps: int = 10,
        k_neighbors: int = 5,
        isolation_n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.n_bootstraps = n_bootstraps
        self.k_neighbors = k_neighbors
        self.isolation_n_estimators = isolation_n_estimators
        self.random_state = random_state

        # Fitted attributes (populated in fit)
        self.numeric_features_: List[str] = []
        self.categorical_features_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.train_min_: Dict[str, float] = {}
        self.train_max_: Dict[str, float] = {}
        self.train_categories_: Dict[str, set] = {}
        self.classes_: np.ndarray = np.array([])
        self.num_classes_: int = 2

        self.preprocessor_: Optional[Pipeline] = None
        self.knn_: Optional[NearestNeighbors] = None
        self.train_knn_distances_: np.ndarray = np.array([])
        self.isolation_forest_: Optional[IsolationForest] = None
        self.isolation_threshold_: float = 0.0
        self.bootstrap_models_: List[Pipeline] = []

        self.last_flags_: List[List[Dict[str, Any]]] = []

    def fit(
        self,
        X_train_raw: pd.DataFrame,
        base_pipeline: Pipeline,
        y_train: np.ndarray,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> "ReliabilityFeatureExtractor":
        """
        Fit all reliability signal extractors strictly on the TRAIN split.

        Parameters
        ----------
        X_train_raw : pd.DataFrame
            Raw features from the train split only.
        base_pipeline : Pipeline
            Unfitted clone of the primary model's sklearn Pipeline.
        y_train : np.ndarray
            Encoded target labels on the train split only.
        numeric_features : list[str]
            List of numeric feature column names.
        categorical_features : list[str]
            List of categorical feature column names.

        Returns
        -------
        ReliabilityFeatureExtractor
            Fitted extractor instance.
        """
        self.numeric_features_ = list(numeric_features)
        self.categorical_features_ = list(categorical_features)
        self.feature_names_in_ = list(X_train_raw.columns)
        self.classes_ = np.unique(y_train)
        self.num_classes_ = len(self.classes_)

        # 1. Compute training boundaries and known categories directly from raw data
        self.train_min_ = {}
        self.train_max_ = {}
        for col in self.numeric_features_:
            if col in X_train_raw.columns:
                series = pd.to_numeric(X_train_raw[col], errors="coerce").dropna()
                # A column with no observed train values has no known range; leaving it
                # out avoids flagging every inference value as out-of-range against (0, 0).
                if len(series) > 0:
                    self.train_min_[col] = float(series.min())
                    self.train_max_[col] = float(series.max())

        self.train_categories_ = {}
        for col in self.categorical_features_:
            if col in X_train_raw.columns:
                series = X_train_raw[col].dropna().astype(str)
                self.train_categories_[col] = set(series.unique())

        # 2. Build and fit a preprocessor specifically for distance & OOD metrics
        self.preprocessor_ = build_preprocessor(
            numeric_features=self.numeric_features_,
            categorical_features=self.categorical_features_,
        )
        X_train_preproc = self.preprocessor_.fit_transform(X_train_raw)
        if hasattr(X_train_preproc, "toarray"):
            X_train_preproc = X_train_preproc.toarray()
        X_train_preproc = np.asarray(X_train_preproc, dtype=np.float64)

        # 3. Fit Nearest Neighbors on preprocessed train data
        actual_k = min(self.k_neighbors, max(1, len(X_train_preproc) - 1))
        self.knn_ = NearestNeighbors(
            n_neighbors=actual_k,
            algorithm="auto",
            metric="euclidean",
            n_jobs=-1,
        )
        self.knn_.fit(X_train_preproc)

        # Compute leave-one-out reference distances on train split (neighbors 1 to actual_k)
        knn_loo = NearestNeighbors(
            n_neighbors=actual_k + 1,
            algorithm="auto",
            metric="euclidean",
            n_jobs=-1,
        )
        knn_loo.fit(X_train_preproc)
        loo_distances, _ = knn_loo.kneighbors(X_train_preproc)
        # Neighbor 0 is the sample itself (distance 0), neighbors 1: are the true nearest points
        train_mean_dists = loo_distances[:, 1:].mean(axis=1)
        self.train_knn_distances_ = np.sort(train_mean_dists)

        # 4. Fit Isolation Forest on preprocessed train data
        self.isolation_forest_ = IsolationForest(
            n_estimators=self.isolation_n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.isolation_forest_.fit(X_train_preproc)
        # Compute baseline isolation scores on train set (higher = more unusual)
        train_iso_scores = -self.isolation_forest_.score_samples(X_train_preproc)
        self.isolation_threshold_ = float(np.percentile(train_iso_scores, 95.0))

        # 5. Train B bootstrap copies of base_pipeline on bootstrap samples of the train split
        self.bootstrap_models_ = []
        rng = np.random.RandomState(self.random_state)
        n_train = len(X_train_raw)

        # Stratified bootstrap: resample WITHIN each class so every bootstrap copy
        # sees every class. A plain bootstrap can drop a rare class entirely, which
        # makes predict_proba return fewer columns (misaligned probabilities) and
        # makes XGBoost refuse to fit (it requires contiguous labels).
        class_indices = [np.where(y_train == c)[0] for c in self.classes_]

        for _ in range(self.n_bootstraps):
            boot_idx = np.concatenate([
                rng.choice(idx, size=len(idx), replace=True) for idx in class_indices
            ])
            rng.shuffle(boot_idx)
            X_boot = X_train_raw.iloc[boot_idx]
            y_boot = y_train[boot_idx]

            boot_pipe = clone(base_pipeline)
            boot_pipe.fit(X_boot, y_boot)
            self.bootstrap_models_.append(boot_pipe)

        return self

    def transform(self, X_raw: pd.DataFrame, model1) -> pd.DataFrame:
        """
        Extract model-agnostic reliability features for new inputs.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Raw input features (unprocessed).
        model1 : Any
            Trained and calibrated Model 1 exposing predict_proba.

        Returns
        -------
        pd.DataFrame
            Matrix of reliability features matching X_raw index.
        """
        features_df, flags = self.transform_with_flags(X_raw, model1)
        return features_df

    def transform_with_flags(
        self,
        X_raw: pd.DataFrame,
        model1,
    ) -> Tuple[pd.DataFrame, List[List[Dict[str, Any]]]]:
        """
        Extract reliability features and structured diagnostic flags.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Raw input records.
        model1 : Any
            Fitted primary model (predict / predict_proba).

        Returns
        -------
        Tuple[pd.DataFrame, List[List[Dict[str, Any]]]]
            Features DataFrame and per-record lists of diagnostic flags.
        """
        if not isinstance(X_raw, pd.DataFrame):
            X_raw = pd.DataFrame(X_raw)

        n_samples = len(X_raw)
        if n_samples == 0:
            empty_df = pd.DataFrame(columns=self._get_feature_names())
            return empty_df, []

        # -----------------------------------------------------------------
        # 1. Primary Model (Model 1) Probabilities & Confidence
        # -----------------------------------------------------------------
        probs = model1.predict_proba(X_raw)
        probs = np.asarray(probs, dtype=np.float64)

        # In case single-class or weird dimensions
        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs])

        max_prob = np.max(probs, axis=1)
        pred_class = np.argmax(probs, axis=1)

        # Normalized Entropy: -sum(p * log(p)) / log(K) in [0, 1]
        eps = 1e-12
        clipped_p = np.clip(probs, eps, 1.0)
        raw_entropy = -np.sum(clipped_p * np.log(clipped_p), axis=1)
        max_entropy = np.log(self.num_classes_) if self.num_classes_ > 1 else 1.0
        entropy = np.clip(raw_entropy / max_entropy, 0.0, 1.0)

        # Margin: top probability minus second top probability
        if probs.shape[1] > 1:
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        else:
            margin = np.ones(n_samples, dtype=np.float64)

        # -----------------------------------------------------------------
        # 2. Bootstrap Disagreement & Variance
        # -----------------------------------------------------------------
        if len(self.bootstrap_models_) > 0:
            boot_probs_list = []
            boot_preds_list = []

            for boot_pipe in self.bootstrap_models_:
                b_p = boot_pipe.predict_proba(X_raw)
                b_p = np.asarray(b_p, dtype=np.float64)
                if b_p.ndim == 1:
                    b_p = np.column_stack([1.0 - b_p, b_p])
                # Align columns to the full class set (defensive: a copy that never
                # saw a class gets probability 0 for it instead of shifted columns).
                boot_classes = getattr(boot_pipe, "classes_", None)
                if boot_classes is not None and len(boot_classes) != self.num_classes_:
                    aligned = np.zeros((n_samples, self.num_classes_), dtype=np.float64)
                    for j, c in enumerate(boot_classes):
                        pos = int(np.searchsorted(self.classes_, c))
                        if pos < self.num_classes_ and self.classes_[pos] == c:
                            aligned[:, pos] = b_p[:, j]
                    b_p = aligned
                boot_probs_list.append(b_p)
                boot_preds_list.append(np.argmax(b_p, axis=1))

            # Stack: (B, n_samples, n_classes)
            boot_probs_stack = np.array(boot_probs_list)
            # Stack: (B, n_samples)
            boot_preds_stack = np.array(boot_preds_list)

            # (a) Std of predicted probability for Model 1's predicted class
            pred_class_probs = np.zeros((len(self.bootstrap_models_), n_samples), dtype=np.float64)
            for b_idx in range(len(self.bootstrap_models_)):
                for i in range(n_samples):
                    c = pred_class[i]
                    if c < boot_probs_stack.shape[2]:
                        pred_class_probs[b_idx, i] = boot_probs_stack[b_idx, i, c]
                    else:
                        pred_class_probs[b_idx, i] = 0.0

            bootstrap_std = np.std(pred_class_probs, axis=0, ddof=0)

            # (b) Disagreement AMONG bootstrap models: 1 minus fraction agreeing with majority class
            bootstrap_disagreement = np.zeros(n_samples, dtype=np.float64)
            num_models = float(len(self.bootstrap_models_))
            for i in range(n_samples):
                sample_votes = boot_preds_stack[:, i]
                counts = np.bincount(sample_votes, minlength=self.num_classes_)
                max_agreement = float(np.max(counts))
                bootstrap_disagreement[i] = 1.0 - (max_agreement / num_models)
        else:
            bootstrap_std = np.zeros(n_samples, dtype=np.float64)
            bootstrap_disagreement = np.zeros(n_samples, dtype=np.float64)

        # -----------------------------------------------------------------
        # 3. Preprocessed Space Signals: kNN Distance & Isolation Forest
        # -----------------------------------------------------------------
        X_preproc = self.preprocessor_.transform(X_raw)
        if hasattr(X_preproc, "toarray"):
            X_preproc = X_preproc.toarray()
        X_preproc = np.asarray(X_preproc, dtype=np.float64)

        # kNN distance & percentile relative to train LOO distribution
        distances, _ = self.knn_.kneighbors(X_preproc)
        knn_distance = distances.mean(axis=1)

        # Percentile rank in train LOO distances (in [0, 100])
        if len(self.train_knn_distances_) > 0:
            rank = np.searchsorted(self.train_knn_distances_, knn_distance, side="right")
            knn_distance_percentile = np.clip(
                (rank / len(self.train_knn_distances_)) * 100.0,
                0.0,
                100.0,
            )
        else:
            knn_distance_percentile = np.zeros(n_samples, dtype=np.float64)

        # Isolation Forest score (higher = more anomalous/unusual)
        isolation_score = -self.isolation_forest_.score_samples(X_preproc)

        # -----------------------------------------------------------------
        # 4. Raw Input Data Quality Signals
        # -----------------------------------------------------------------
        total_cols = max(1, len(X_raw.columns))
        missing_count = X_raw.isna().sum(axis=1).values.astype(int)
        missing_fraction = missing_count / float(total_cols)

        out_of_range_count = np.zeros(n_samples, dtype=int)
        unseen_category_count = np.zeros(n_samples, dtype=int)

        # Diagnostic flags per sample
        flags: List[List[Dict[str, Any]]] = [[] for _ in range(n_samples)]

        # Check missing fields
        for col in X_raw.columns:
            is_na = X_raw[col].isna().values
            for i in np.where(is_na)[0]:
                flags[i].append({"type": "missing_feature", "feature": str(col)})

        # Check numeric out-of-range
        for col in self.numeric_features_:
            if col in X_raw.columns:
                series = pd.to_numeric(X_raw[col], errors="coerce").values
                min_val = self.train_min_.get(col, -np.inf)
                max_val = self.train_max_.get(col, np.inf)

                non_na = ~np.isnan(series)
                below = non_na & (series < min_val)
                above = non_na & (series > max_val)
                outliers = below | above

                for i in np.where(outliers)[0]:
                    out_of_range_count[i] += 1
                    flags[i].append({
                        "type": "out_of_range",
                        "feature": str(col),
                        "value": float(series[i]),
                        "train_min": min_val,
                        "train_max": max_val,
                    })

        # Check unseen categories
        for col in self.categorical_features_:
            if col in X_raw.columns:
                known = self.train_categories_.get(col, set())
                vals = X_raw[col].values
                for i in range(n_samples):
                    v = vals[i]
                    if pd.notna(v) and str(v) not in known:
                        unseen_category_count[i] += 1
                        flags[i].append({
                            "type": "unseen_category",
                            "feature": str(col),
                            "value": str(v),
                        })

        # Contextual extreme signals flags
        for i in range(n_samples):
            if knn_distance_percentile[i] >= 95.0:
                flags[i].append({
                    "type": "high_knn_distance",
                    "percentile": round(float(knn_distance_percentile[i]), 1),
                    "distance": round(float(knn_distance[i]), 4),
                })
            if isolation_score[i] >= self.isolation_threshold_:
                flags[i].append({
                    "type": "high_isolation_score",
                    "score": round(float(isolation_score[i]), 4),
                    "threshold": round(float(self.isolation_threshold_), 4),
                })
            if bootstrap_disagreement[i] >= 0.30:
                flags[i].append({
                    "type": "high_bootstrap_disagreement",
                    "disagreement": round(float(bootstrap_disagreement[i]), 3),
                })
            if entropy[i] >= 0.80:
                flags[i].append({
                    "type": "high_entropy",
                    "entropy": round(float(entropy[i]), 3),
                })
            if margin[i] <= 0.10:
                flags[i].append({
                    "type": "low_margin",
                    "margin": round(float(margin[i]), 3),
                })

        # -----------------------------------------------------------------
        # 5. Assemble Features DataFrame
        # -----------------------------------------------------------------
        features_dict = {
            "max_prob": max_prob,
            "entropy": entropy,
            "margin": margin,
            "predicted_class": pred_class,
            "bootstrap_disagreement": bootstrap_disagreement,
            "bootstrap_std": bootstrap_std,
            "knn_distance": knn_distance,
            "knn_distance_percentile": knn_distance_percentile,
            "isolation_score": isolation_score,
            "missing_count": missing_count,
            "missing_fraction": missing_fraction,
            "out_of_range_count": out_of_range_count,
            "unseen_category_count": unseen_category_count,
        }

        features_df = pd.DataFrame(features_dict, index=X_raw.index)
        features_df.attrs["flags"] = flags
        self.last_flags_ = flags

        return features_df, flags

    def extract_flags(
        self,
        X_raw: pd.DataFrame,
        model1,
    ) -> List[List[Dict[str, Any]]]:
        """Convenience method returning only the structured flags."""
        _, flags = self.transform_with_flags(X_raw, model1)
        return flags

    def _get_feature_names(self) -> List[str]:
        return [
            "max_prob",
            "entropy",
            "margin",
            "predicted_class",
            "bootstrap_disagreement",
            "bootstrap_std",
            "knn_distance",
            "knn_distance_percentile",
            "isolation_score",
            "missing_count",
            "missing_fraction",
            "out_of_range_count",
            "unseen_category_count",
        ]
