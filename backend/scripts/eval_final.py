"""Task 6: Final Evaluation of ModelDoctor.

Compares:
1. ModelDoctor (TRUST-only acceptance)
2. Confidence-only baseline (matched coverage)
3. No filtering (100% coverage)

Evaluates error detection, AUROC, AUPRC, ECE,
selective risk, coverage, and risk-coverage curves.
"""

from pathlib import Path
import sys
import json

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Ensure backend root is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.data.loader import load_dataset
from ml.data.profiler import profile_dataset
from ml.data.splitter import split_data

from ml.models.selector import select_model

from ml.reliability.features import ReliabilityFeatureExtractor
from ml.reliability.model2 import train_model2
from ml.reliability.mitigation import (
    tune_mitigation_thresholds,
    MitigationEngine,
)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def safe_auc(y_true, scores):
    """Calculate AUROC safely."""
    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, scores))


def safe_auprc(y_true, scores):
    """Calculate AUPRC safely."""
    if len(np.unique(y_true)) < 2:
        return None

    return float(average_precision_score(y_true, scores))


def calculate_ece(y_true, probabilities, n_bins=10):
    """Calculate Expected Calibration Error."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probabilities >= bins[i])
                & (probabilities <= bins[i + 1])
            )
        else:
            mask = (
                (probabilities >= bins[i])
                & (probabilities < bins[i + 1])
            )

        count = int(np.sum(mask))

        if count == 0:
            continue

        mean_predicted = float(np.mean(probabilities[mask]))
        observed_accuracy = float(np.mean(y_true[mask]))

        ece += (
            count / len(y_true)
        ) * abs(mean_predicted - observed_accuracy)

        reliability.append({
            "bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}",
            "count": count,
            "mean_predicted": mean_predicted,
            "observed_accuracy": observed_accuracy,
        })

    return float(ece), reliability


def calculate_risk_coverage_curve(reliability_scores, y_error):
    """Calculate risk as progressively less reliable records are accepted."""

    order = np.argsort(-reliability_scores, kind="stable")

    sorted_errors = y_error[order]

    cumulative_errors = np.cumsum(sorted_errors)
    sample_counts = np.arange(1, len(y_error) + 1)

    coverage = sample_counts / len(y_error)
    risk = cumulative_errors / sample_counts

    # Keep at most 100 points for a compact JSON report.
    selected = np.unique(
        np.linspace(
            0,
            len(coverage) - 1,
            min(100, len(coverage)),
            dtype=int,
        )
    )

    return {
        "coverage": coverage[selected].tolist(),
        "risk": risk[selected].tolist(),
    }


def calculate_error_detection_metrics(y_error, predicted_error):
    """Calculate precision, recall and confusion matrix."""

    return {
        "precision": float(
            precision_score(
                y_error,
                predicted_error,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_error,
                predicted_error,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_error,
            predicted_error,
            labels=[0, 1],
        ).tolist(),
    }


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def run_evaluation():

    print("\n" + "=" * 65)
    print("MODELDOCTOR - TASK 6 FINAL EVALUATION")
    print("=" * 65)

    # 1. Locate dataset
    candidates = [
        Path("sample_data/loans.csv"),
        Path("../sample_data/loans.csv"),
        BASE_DIR.parent / "sample_data" / "loans.csv",
    ]

    data_path = next(
        (p for p in candidates if p.exists()),
        None,
    )

    if data_path is None:
        raise FileNotFoundError(
            "Could not find sample_data/loans.csv"
        )

    print(f"\nDataset: {data_path.resolve()}")

    # 2. Load dataset
    target_column = "default"

    df = load_dataset(
        str(data_path),
        target_column=target_column,
    )

    profile = profile_dataset(
        df,
        target_column=target_column,
    )

    numeric_features = profile["numeric_columns"]
    categorical_features = profile["categorical_columns"]

    # 3. Split dataset
    splits = split_data(
        df,
        target_column=target_column,
        random_state=42,
    )

    print("\nDataset split:")
    print(f"Train       : {len(splits.X_train_raw)}")
    print(f"Calibration : {len(splits.X_calib_raw)}")
    print(f"Test        : {len(splits.X_test_raw)}")

    # -----------------------------------------------------
    # STEP 1: TRAIN MODEL 1
    # -----------------------------------------------------

    print("\n[1/7] Training Model 1...")

    model1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=42,
    )

    model1 = model1_res["model"]
    base_pipeline = model1_res["base_pipeline"]

    print(f"Selected Model 1: {model1_res['winner_name']}")

    # -----------------------------------------------------
    # STEP 2: FIT RELIABILITY FEATURE EXTRACTOR
    # -----------------------------------------------------

    print("\n[2/7] Fitting reliability feature extractor...")

    extractor = ReliabilityFeatureExtractor(
        n_bootstraps=10,
        k_neighbors=5,
        isolation_n_estimators=100,
        random_state=42,
    )

    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=base_pipeline,
        y_train=splits.y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    # -----------------------------------------------------
    # STEP 3: TRAIN MODEL 2 USING CALIBRATION DATA
    # -----------------------------------------------------

    print("\n[3/7] Training Model 2...")

    calib_feats, _ = extractor.transform_with_flags(
        splits.X_calib_raw,
        model1=model1,
    )

    calib_preds = model1.predict(splits.X_calib_raw)

    y_correct = (
        calib_preds == splits.y_calib
    ).astype(int)

    m2_res = train_model2(
        features_df=calib_feats,
        y_correct=y_correct,
        num_classes=2,
        exclude_predicted_class=True,
        exclude_redundant_binary_features=True,
        random_state=42,
    )

    if m2_res["status"] != "success":
        raise RuntimeError(
            f"Model 2 training failed: {m2_res.get('message')}"
        )

    model2 = m2_res["model2"]

    print(f"Selected Model 2: {m2_res['winner_name']}")

    # -----------------------------------------------------
    # STEP 4: TUNE MITIGATION THRESHOLDS
    # -----------------------------------------------------

    print("\n[4/7] Tuning mitigation thresholds...")

    thresholds = tune_mitigation_thresholds(
        oof_trustworthiness=m2_res["oof_trustworthiness"],
        y_correct=y_correct,
        target_risk=0.05,
        min_coverage=0.30,
        reject_min_error=0.30,
        reject_min_rows=30,
    )

    engine = MitigationEngine(**thresholds)

    print("Trust threshold:", thresholds["trust_threshold"])
    print("Reject threshold:", thresholds.get("reject_threshold"))

    # -----------------------------------------------------
    # STEP 5: EVALUATE ON UNTOUCHED TEST DATA
    # -----------------------------------------------------

    print("\n[5/7] Evaluating on TEST split...")

    test_feats, test_flags = extractor.transform_with_flags(
        splits.X_test_raw,
        model1=model1,
    )

    test_preds = model1.predict(splits.X_test_raw)

    test_probs = model1.predict_proba(splits.X_test_raw)

    confidence_scores = np.max(test_probs, axis=1)

    trust_scores = model2.predict_trustworthiness(test_feats)

    # Actual prediction correctness
    y_error = (
        test_preds != splits.y_test
    ).astype(int)

    y_correct_test = 1 - y_error

    print(f"Test records: {len(y_error)}")
    print(f"Correct predictions: {int(np.sum(y_correct_test))}")
    print(f"Incorrect predictions: {int(np.sum(y_error))}")

    # -----------------------------------------------------
    # STEP 6: MODELDOCTOR DECISIONS
    # -----------------------------------------------------

    print("\n[6/7] Applying mitigation decisions...")

    decisions = np.array([
        engine.decide(
            float(trust_scores[i]),
            test_flags[i],
        )
        for i in range(len(trust_scores))
    ])

    # TRUST records are accepted.
    md_accepted = decisions == "TRUST"

    md_predicted_error = (~md_accepted).astype(int)

    md_accepted_count = int(np.sum(md_accepted))

    md_coverage = float(np.mean(md_accepted))

    md_risk = (
        float(np.mean(y_error[md_accepted]))
        if md_accepted_count > 0
        else None
    )

    # -----------------------------------------------------
    # CONFIDENCE BASELINE AT MATCHED COVERAGE
    # -----------------------------------------------------

    # Accept the same number of records as ModelDoctor,
    # selecting records with the highest Model 1 confidence.

    baseline_accepted = np.zeros(
        len(confidence_scores),
        dtype=bool,
    )

    if md_accepted_count > 0:
        baseline_order = np.argsort(
            -confidence_scores,
            kind="stable",
        )

        baseline_accepted[
            baseline_order[:md_accepted_count]
        ] = True

    baseline_predicted_error = (
        ~baseline_accepted
    ).astype(int)

    baseline_risk = (
        float(np.mean(y_error[baseline_accepted]))
        if np.any(baseline_accepted)
        else None
    )

    baseline_coverage = float(
        np.mean(baseline_accepted)
    )

    # -----------------------------------------------------
    # NO-FILTERING BASELINE
    # -----------------------------------------------------

    no_filtering_risk = float(np.mean(y_error))

    no_filtering_coverage = 1.0

    # -----------------------------------------------------
    # STEP 7: CALCULATE METRICS
    # -----------------------------------------------------

    print("\n[7/7] Calculating evaluation metrics...")

    # Error detection scores:
    # Higher score = more likely to be incorrect.

    md_error_scores = 1.0 - trust_scores

    confidence_error_scores = 1.0 - confidence_scores

    # Calibration metrics
    md_ece, md_reliability = calculate_ece(
        y_correct_test,
        trust_scores,
    )

    confidence_ece, confidence_reliability = calculate_ece(
        y_correct_test,
        confidence_scores,
    )

    # Final report
    report = {

        "project": "ModelDoctor",

        "task": "Task 6 - Final Evaluation",

        "dataset": str(data_path),

        "test_samples": len(y_error),

        "models": {
            "model1": model1_res["winner_name"],
            "model2": m2_res["winner_name"],
        },

        "thresholds": {
            key: (
                float(value)
                if isinstance(value, (int, float, np.integer, np.floating))
                else value
            )
            for key, value in thresholds.items()
        },

        "prediction_performance": {
            "accuracy": float(np.mean(y_correct_test)),
            "error_rate": float(np.mean(y_error)),
        },

        "modeldoctor": {

            "coverage": md_coverage,

            "selective_risk": md_risk,

            "error_detection": calculate_error_detection_metrics(
                y_error,
                md_predicted_error,
            ),

            "error_detection_auroc": safe_auc(
                y_error,
                md_error_scores,
            ),

            "error_detection_auprc": safe_auprc(
                y_error,
                md_error_scores,
            ),

            "ece": md_ece,

            "reliability_curve": md_reliability,

            "risk_coverage_curve": calculate_risk_coverage_curve(
                trust_scores,
                y_error,
            ),
        },

        "confidence_baseline": {

            "coverage": baseline_coverage,

            "selective_risk": baseline_risk,

            "error_detection": calculate_error_detection_metrics(
                y_error,
                baseline_predicted_error,
            ),

            "error_detection_auroc": safe_auc(
                y_error,
                confidence_error_scores,
            ),

            "error_detection_auprc": safe_auprc(
                y_error,
                confidence_error_scores,
            ),

            "ece": confidence_ece,

            "reliability_curve": confidence_reliability,

            "risk_coverage_curve": calculate_risk_coverage_curve(
                confidence_scores,
                y_error,
            ),
        },

        "no_filtering": {

            "coverage": no_filtering_coverage,

            "selective_risk": no_filtering_risk,

            "description": "All predictions are accepted.",
        },

        "comparison": {

            "modeldoctor_vs_confidence_same_coverage": True,

            "modeldoctor_coverage": md_coverage,

            "confidence_coverage": baseline_coverage,

            "modeldoctor_risk": md_risk,

            "confidence_risk": baseline_risk,

            "no_filtering_risk": no_filtering_risk,
        },
    }

    # -----------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------

    report_path = BASE_DIR / "evaluation_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else str(value)
            ),
        )

    # -----------------------------------------------------
    # PRINT SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("TASK 6 - EVALUATION RESULTS")
    print("=" * 65)

    print("\nMODELDOCTOR")
    print("------------------------------")
    print(f"Coverage       : {md_coverage:.4f}")
    print(f"Selective Risk : {md_risk}")

    print("\nCONFIDENCE BASELINE")
    print("------------------------------")
    print(f"Coverage       : {baseline_coverage:.4f}")
    print(f"Selective Risk : {baseline_risk}")

    print("\nNO FILTERING")
    print("------------------------------")
    print(f"Coverage       : {no_filtering_coverage:.4f}")
    print(f"Selective Risk : {no_filtering_risk:.4f}")

    print("\nERROR DETECTION")
    print("------------------------------")
    print(
        "ModelDoctor AUROC:",
        safe_auc(y_error, md_error_scores),
    )
    print(
        "Confidence AUROC:",
        safe_auc(y_error, confidence_error_scores),
    )

    print("\nCALIBRATION")
    print("------------------------------")
    print(f"ModelDoctor ECE : {md_ece:.4f}")
    print(f"Confidence ECE  : {confidence_ece:.4f}")

    print("\nReport saved at:")
    print(report_path)

    print("\n[TASK 6 COMPLETE]")


if __name__ == "__main__":
    run_evaluation()