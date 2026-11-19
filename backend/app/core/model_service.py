"""Core training and prediction service for ModelDoctor."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.loader import load_dataset
from ml.data.splitter import split_data
from ml.models.selector import select_model
from ml.reliability.features import ReliabilityFeatureExtractor
from ml.reliability.model2 import train_model2
from ml.reliability.mitigation import (
    tune_mitigation_thresholds,
    MitigationEngine,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR.parent / "sample_data" / "loans.csv"
TARGET_COLUMN = "default"


class ModelDoctorService:

    def __init__(self):
        self.is_ready = False

        # 1. Load and validate dataset
        df = load_dataset(DATA_PATH, TARGET_COLUMN)

        self.numeric_features = (
            df.drop(columns=[TARGET_COLUMN])
            .select_dtypes(include=np.number)
            .columns.tolist()
        )

        self.categorical_features = [
            col for col in df.drop(columns=[TARGET_COLUMN]).columns
            if col not in self.numeric_features
        ]

        # 2. Split dataset
        splits = split_data(
            df,
            target_column=TARGET_COLUMN,
            random_state=42,
        )

        # 3. Train Model 1
        model1_res = select_model(
            X_train_raw=splits.X_train_raw,
            y_train=splits.y_train,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )

        self.model1 = model1_res["model"]

        # 4. Fit reliability feature extractor
        self.extractor = ReliabilityFeatureExtractor(
            n_bootstraps=10,
            k_neighbors=5,
            isolation_n_estimators=100,
            random_state=42,
        )

        self.extractor.fit(
            X_train_raw=splits.X_train_raw,
            base_pipeline=model1_res["base_pipeline"],
            y_train=splits.y_train,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )

        # 5. Generate calibration reliability features
        calibration_features = self.extractor.transform(
            splits.X_calib_raw,
            self.model1,
        )

        calibration_predictions = self.model1.predict(
            splits.X_calib_raw
        )

        y_correct = (
            calibration_predictions == splits.y_calib
        ).astype(int)

        # 6. Train Model 2
        model2_res = train_model2(
            features_df=calibration_features,
            y_correct=y_correct,
            num_classes=model1_res["num_classes"],
            exclude_predicted_class=True,
            exclude_redundant_binary_features=True,
            random_state=42,
        )

        if model2_res["status"] != "success":
            raise RuntimeError(model2_res["message"])

        self.model2 = model2_res["model2"]

        # 7. Tune mitigation thresholds
        thresholds = tune_mitigation_thresholds(
            oof_trustworthiness=model2_res["oof_trustworthiness"],
            y_correct=y_correct,
            target_risk=0.05,
            min_coverage=0.30,
            reject_min_error=0.30,
            reject_min_rows=30,
        )

        self.mitigation = MitigationEngine(**thresholds)

        self.class_mapping = splits.inverse_class_mapping
        self.feature_names = splits.feature_names
        self.model1_name = model1_res["winner_name"]
        self.model2_name = model2_res["winner_name"]

        self.is_ready = True

    def predict(self, records):
        if not self.is_ready:
            raise RuntimeError("ModelDoctor service is not ready.")

        X = pd.DataFrame(records)

        missing = [
            col for col in self.feature_names
            if col not in X.columns
        ]

        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = X[self.feature_names]

        # Model 1 prediction
        predictions = self.model1.predict(X)
        probabilities = self.model1.predict_proba(X)

        # Reliability features and diagnostic flags
        features, flags = self.extractor.transform_with_flags(
            X,
            self.model1,
        )

        # Model 2 trustworthiness
        trust_scores = self.model2.predict_trustworthiness(features)

        results = []

        for i, prediction in enumerate(predictions):
            evaluation = self.mitigation.evaluate_record(
                trustworthiness=float(trust_scores[i]),
                record_flags=flags[i],
                features=features.iloc[i].to_dict(),
            )

            results.append({
                "prediction": self.class_mapping[int(prediction)],
                "class_probabilities": {
                    str(label): float(probabilities[i][idx])
                    for idx, label in enumerate(self.model1.classes_)
                },
                "trustworthiness": evaluation["trustworthiness"],
                "decision": evaluation["decision"],
                "warnings": evaluation["warnings"],
            })

        return results


# Shared service instance
model_service = None


def get_model_service():
    global model_service

    if model_service is None:
        model_service = ModelDoctorService()

    return model_service
