import json
import joblib
import numpy as np

from pathlib import Path
from datetime import datetime, timezone

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


STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage"


def train_session(session_id: str, target_column: str):

    session_dir = STORAGE_DIR / session_id
    dataset_path = session_dir / "dataset.csv"
    status_path = session_dir / "status.json"

    if not dataset_path.exists():
        raise FileNotFoundError("Dataset session not found.")

    def update_status(stage, progress, status="running", error=None):
        status_data = {
            "session_id": session_id,
            "status": status,
            "stage": stage,
            "progress": progress,
            "error": error,
        }
        status_path.write_text(json.dumps(status_data))

    try:
        # STEP 1: Load and profile dataset
        update_status("profiling", 10)

        df = load_dataset(str(dataset_path), target_column)
        profile = profile_dataset(df, target_column)

        numeric_features = profile["numeric_columns"]
        categorical_features = profile["categorical_columns"]

        # STEP 2: Split dataset
        update_status("splitting", 20)

        splits = split_data(
            df,
            target_column,
            random_state=42,
        )

        # STEP 3: Train Model 1
        update_status("model_selection", 35)

        model1_res = select_model(
            X_train_raw=splits.X_train_raw,
            y_train=splits.y_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=42,
        )

        model1 = model1_res["model"]
        base_pipeline = model1_res["base_pipeline"]

        # STEP 4: Fit reliability extractor on TRAIN only
        update_status("reliability_features", 50)

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

        # STEP 5: Train Model 2 using CALIBRATION only
        update_status("reliability_training", 65)

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
                m2_res.get("message", "Model 2 training failed.")
            )

        model2 = m2_res["model2"]

        # STEP 6: Tune thresholds using calibration OOF only
        thresholds = tune_mitigation_thresholds(
            oof_trustworthiness=m2_res["oof_trustworthiness"],
            y_correct=y_correct,
            target_risk=0.05,
            min_coverage=0.30,
            reject_min_error=0.30,
            reject_min_rows=30,
        )

        engine = MitigationEngine(**thresholds)

        # STEP 7: Evaluate on untouched TEST data
        update_status("evaluation", 80)

        test_feats, test_flags = extractor.transform_with_flags(
            splits.X_test_raw,
            model1=model1,
        )

        test_preds = model1.predict(splits.X_test_raw)
        test_probs = model1.predict_proba(splits.X_test_raw)

        confidence_scores = np.max(test_probs, axis=1)
        trust_scores = model2.predict_trustworthiness(test_feats)

        y_error = (
            test_preds != splits.y_test
        ).astype(int)

        decisions = np.array([
            engine.decide(
                float(trust_scores[i]),
                test_flags[i],
            )
            for i in range(len(trust_scores))
        ])

        accepted = decisions == "TRUST"

        coverage = float(np.mean(accepted))

        selective_risk = (
            float(np.mean(y_error[accepted]))
            if np.any(accepted)
            else None
        )

        # Confidence baseline at matched coverage
        baseline_accepted = np.zeros(
            len(confidence_scores),
            dtype=bool,
        )

        accepted_count = int(np.sum(accepted))

        if accepted_count > 0:
            order = np.argsort(
                -confidence_scores,
                kind="stable",
            )
            baseline_accepted[order[:accepted_count]] = True

        baseline_risk = (
            float(np.mean(y_error[baseline_accepted]))
            if np.any(baseline_accepted)
            else None
        )

        report = {
            "project": "ModelDoctor",
            "session_id": session_id,
            "target_column": target_column,
            "dataset_profile": profile,
            "models": {
                "model1": model1_res["winner_name"],
                "model2": m2_res["winner_name"],
            },
            "thresholds": {
                "trust_threshold": float(
                    thresholds["trust_threshold"]
                ),
                "reject_threshold": (
                    float(thresholds["reject_threshold"])
                    if thresholds.get("reject_threshold") is not None
                    else None
                ),
            },
            "test_samples": len(y_error),
            "prediction_performance": {
                "accuracy": float(np.mean(1 - y_error)),
                "error_rate": float(np.mean(y_error)),
            },
            "modeldoctor": {
                "coverage": coverage,
                "selective_risk": selective_risk,
            },
            "confidence_baseline": {
                "coverage": float(np.mean(baseline_accepted)),
                "selective_risk": baseline_risk,
            },
            "no_filtering": {
                "coverage": 1.0,
                "selective_risk": float(np.mean(y_error)),
            },
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        # STEP 8: Save trained artifacts
        update_status("saving", 90)

        joblib.dump(model1, session_dir / "model1.joblib")
        joblib.dump(model2, session_dir / "model2.joblib")
        joblib.dump(extractor, session_dir / "extractor.joblib")
        joblib.dump(engine, session_dir / "mitigation.joblib")

        (session_dir / "report.json").write_text(
            json.dumps(report, indent=2, default=str)
        )

        schema = {
            "target_column": target_column,
            "feature_columns": list(splits.feature_names),
            "classes": [str(c) for c in splits.classes],
            "class_mapping": {
                str(k): int(v)
                for k, v in splits.class_mapping.items()
            },
        }

        (session_dir / "schema.json").write_text(
            json.dumps(schema, indent=2)
        )

        update_status("completed", 100, status="completed")

        return report

    except Exception as exc:
        update_status(
            "failed",
            100,
            status="failed",
            error=str(exc),
        )
        raise