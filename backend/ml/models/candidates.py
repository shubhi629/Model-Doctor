"""Candidate classification models pool for ModelDoctor.

All models are returned as complete scikit-learn Pipelines combining
the column preprocessor with the estimator. This ensures that preprocessing
is strictly re-fit inside every cross-validation fold without data leakage.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from ml.data.preprocessing import build_preprocessor


def get_candidate_models(
    numeric_features: List[str],
    categorical_features: List[str],
    num_classes: int = 2,
    class_counts: Optional[Dict[int, int]] = None,
    handle_imbalance: bool = True,
    random_state: int = 42,
    fast: bool = False,
) -> Dict[str, Pipeline]:
    """
    Construct candidate classification pipelines with sensible defaults
    and class imbalance handling.

    Parameters
    ----------
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.
    num_classes : int, default=2
        Number of target classes.
    class_counts : dict[int, int], optional
        Sample counts per class label for computing XGBoost scale_pos_weight.
    handle_imbalance : bool, default=True
        Whether to configure class weights / scale_pos_weight to balance classes.
    random_state : int, default=42
        Reproducibility seed.
    fast : bool, default=False
        Use fewer trees for Random Forest / XGBoost (quick demos and tests).

    Returns
    -------
    dict[str, Pipeline]
        Mapping of candidate model name to full sklearn Pipeline.
    """
    candidates: Dict[str, Pipeline] = {}
    rf_trees = 30 if fast else 200
    xgb_trees = 30 if fast else 100

    # Determine class weight configuration
    class_weight = "balanced" if handle_imbalance else None

    # Calculate scale_pos_weight for binary XGBoost
    scale_pos_weight = 1.0
    if handle_imbalance and num_classes == 2 and class_counts is not None:
        count_0 = class_counts.get(0, 1)
        count_1 = class_counts.get(1, 1)
        if count_1 > 0:
            scale_pos_weight = float(count_0 / count_1)

    # 1. Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs",
    )
    candidates["Logistic Regression"] = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", lr),
    ])

    # 2. Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=6,
        class_weight=class_weight,
        random_state=random_state,
    )
    candidates["Decision Tree"] = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", dt),
    ])

    # 3. Random Forest
    rf = RandomForestClassifier(
        n_estimators=rf_trees,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    candidates["Random Forest"] = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", rf),
    ])

    # 4. XGBoost Classifier
    if num_classes == 2:
        xgb = XGBClassifier(
            n_estimators=xgb_trees,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        xgb = XGBClassifier(
            n_estimators=xgb_trees,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="mlogloss",
            objective="multi:softprob",
            random_state=random_state,
            n_jobs=-1,
        )
    candidates["XGBoost"] = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", xgb),
    ])

    # 5. K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    candidates["KNN"] = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", knn),
    ])

    return candidates
