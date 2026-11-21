import json
from pathlib import Path

import joblib
import pandas as pd

STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage"


def predict_session(session_id: str, records: list):
    session_dir = STORAGE_DIR / session_id

    required_files = [
        "model1.joblib",
        "model2.joblib",
        "extractor.joblib",
        "mitigation.joblib",
        "schema.json",
    ]

    for filename in required_files:
        if not (session_dir / filename).exists():
            raise FileNotFoundError(
                f"Trained model artifact not found: {filename}"
            )

    model1 = joblib.load(session_dir / "model1.joblib")
    model2 = joblib.load(session_dir / "model2.joblib")
    extractor = joblib.load(session_dir / "extractor.joblib")
    mitigation = joblib.load(session_dir / "mitigation.joblib")

    schema = json.loads(
        (session_dir / "schema.json").read_text()
    )

    feature_names = schema["feature_columns"]
    class_mapping = schema["class_mapping"]

    inverse_mapping = {
        int(encoded): original
        for original, encoded in class_mapping.items()
    }

    X = pd.DataFrame(records)

    missing = [
        col for col in feature_names
        if col not in X.columns
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = X[feature_names]

    predictions = model1.predict(X)
    probabilities = model1.predict_proba(X)

    features, flags = extractor.transform_with_flags(X, model1)

    trust_scores = model2.predict_trustworthiness(features)

    results = []

    for i, prediction in enumerate(predictions):
        evaluation = mitigation.evaluate_record(
            trustworthiness=float(trust_scores[i]),
            record_flags=flags[i],
            features=features.iloc[i].to_dict(),
        )

        results.append({
            "prediction": inverse_mapping[int(prediction)],
            "class_probabilities": {
                inverse_mapping[int(label)]: float(probabilities[i][idx])
                for idx, label in enumerate(model1.classes_)
            },
            "trustworthiness": evaluation["trustworthiness"],
            "decision": evaluation["decision"],
            "warnings": evaluation["warnings"],
        })

    return results