"""Unit tests for the production inference engine (ml/inference.py) and the safety
rules it relies on in Model 2 and the mitigation engine."""

import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.data.loader import load_dataset
from ml.inference import InferenceEngine, ModelLoadError, load_artifacts, save_artifacts
from ml.pipeline import RecordValidationError, TrainingConfig, train_modeldoctor
from ml.reliability.mitigation import RISK_LEVELS, WARNING_CATEGORIES, MitigationEngine
from ml.reliability.model2 import Model2Detector

DATA = Path(__file__).resolve().parents[2] / "sample_data"
EXPECTED_KEYS = {
    "prediction", "confidence", "trustworthiness", "decision", "warnings", "class_probabilities",
    "reliability_features", "risk_level", "explanation", "reasons", "reliability_mode",
}


@pytest.fixture(scope="module")
def loans():
    return load_dataset(str(DATA / "loans.csv"), "default")


@pytest.fixture(scope="module")
def bundle(loans, tmp_path_factory):
    """Train once (fast mode), save to disk, return (directory, artifacts, report)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifacts, report = train_modeldoctor(loans, "default", TrainingConfig(fast=True))
    directory = tmp_path_factory.mktemp("bundle")
    save_artifacts(directory, artifacts)
    return directory, artifacts, report


@pytest.fixture(scope="module")
def engine(bundle):
    return InferenceEngine.from_directory(bundle[0])


@pytest.fixture
def records(loans):
    return loans.drop(columns=["default"]).head(25).to_dict(orient="records")


def _copy_bundle(bundle, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(bundle[0], target)
    return target


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def test_loads_from_directory_and_matches_in_memory(bundle, engine, records):
    in_memory = InferenceEngine(bundle[1])
    assert engine.reliability_mode == "model2"
    assert engine.classes == bundle[1]["classes"]
    assert engine.predict(records) == in_memory.predict(records)


def test_missing_metadata_raises(tmp_path):
    with pytest.raises(ModelLoadError, match="No trained model"):
        load_artifacts(tmp_path)


def test_missing_required_artifact_raises(bundle, tmp_path):
    path = _copy_bundle(bundle, tmp_path)
    (path / "model1.joblib").unlink()
    with pytest.raises(ModelLoadError, match="model1"):
        load_artifacts(path)


def test_missing_model2_in_model2_mode_raises(bundle, tmp_path):
    path = _copy_bundle(bundle, tmp_path)
    (path / "model2.joblib").unlink()
    with pytest.raises(ModelLoadError, match="Model 2"):
        load_artifacts(path)


@pytest.mark.parametrize("name", ["model2.joblib", "engine.joblib", "artifacts_meta.joblib"])
def test_corrupted_artifact_raises(bundle, tmp_path, name):
    path = _copy_bundle(bundle, tmp_path)
    (path / name).write_bytes(b"this is not a joblib file")
    with pytest.raises(ModelLoadError, match="could not be read"):
        load_artifacts(path)


def test_wrong_artifact_type_raises(bundle, tmp_path):
    path = _copy_bundle(bundle, tmp_path)
    joblib.dump({"not": "an engine"}, path / "engine.joblib")
    with pytest.raises(ModelLoadError, match="Mitigation engine"):
        load_artifacts(path)


def test_inference_does_not_modify_models(engine, records):
    before = {k: joblib.hash(engine.artifacts[k]) for k in ("model1", "extractor", "model2", "engine")}
    engine.predict(records)
    after = {k: joblib.hash(engine.artifacts[k]) for k in ("model1", "extractor", "model2", "engine")}
    assert before == after


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_unknown_field_rejected(engine, records):
    with pytest.raises(RecordValidationError, match="unknown field"):
        engine.predict([{**records[0], "favourite_colour": "blue"}])


def test_missing_field_rejected(engine, records):
    rec = dict(records[0])
    rec.pop("age")
    with pytest.raises(RecordValidationError, match="missing required field"):
        engine.predict([rec])


def test_batch_with_one_incomplete_record_names_the_row(engine, records):
    bad = dict(records[1])
    bad.pop("credit_score")
    with pytest.raises(RecordValidationError, match="Record 1"):
        engine.predict([records[0], bad, records[2]])


def test_non_numeric_value_rejected(engine, records):
    with pytest.raises(RecordValidationError, match="must be numeric"):
        engine.predict([{**records[0], "age": "thirty"}])


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_value_rejected(engine, records, value):
    with pytest.raises(RecordValidationError, match="finite"):
        engine.predict([{**records[0], "annual_income": value}])


@pytest.mark.parametrize("value", [None, np.nan, ""])
def test_missing_values_are_allowed_and_flagged(engine, records, value):
    res = engine.predict([{**records[0], "credit_score": value}])[0]
    codes = {w["code"] for w in res["warnings"]}
    assert "MISSING_FIELD" in codes


def test_numbers_for_categorical_and_bools_for_numeric_are_coerced(engine, records):
    res = engine.predict([{**records[0], "home_ownership": 7, "employment_years": True}])[0]
    assert "UNSEEN_CATEGORY" in {w["code"] for w in res["warnings"]}


def test_empty_batch_returns_empty_list(engine):
    assert engine.predict([]) == []


def test_non_mapping_record_rejected(engine):
    with pytest.raises(RecordValidationError, match="must be an object"):
        engine.predict([[1, 2, 3]])


# ---------------------------------------------------------------------------
# Predictions, trustworthiness, decisions
# ---------------------------------------------------------------------------

def test_result_structure(engine, records):
    for res in engine.predict(records):
        assert EXPECTED_KEYS <= set(res)
        assert res["prediction"] in engine.classes
        assert 0.0 <= res["confidence"] <= 1.0 and 0.0 <= res["trustworthiness"] <= 1.0
        assert abs(sum(res["class_probabilities"].values()) - 1.0) < 1e-3
        assert res["risk_level"] == RISK_LEVELS[res["decision"]]
        assert res["explanation"] and res["reasons"]
        for w in res["warnings"]:
            assert w["category"] == WARNING_CATEGORIES.get(w["code"], "other")


def test_trustworthiness_comes_from_model2_not_confidence(engine, loans):
    res = engine.predict(loans.drop(columns=["default"]).head(300))
    conf = np.array([r["confidence"] for r in res])
    trust = np.array([r["trustworthiness"] for r in res])
    assert not np.allclose(conf, trust, atol=1e-3)


def test_decisions_follow_the_calibrated_thresholds(engine, loans):
    m = engine.mitigation
    for res in engine.predict(loans.drop(columns=["default"]).head(400)):
        t, codes = res["trustworthiness"], {w["code"] for w in res["warnings"]}
        if res["decision"] == "TRUST":
            assert t >= m.trust_threshold
        elif res["decision"] == "REJECT":
            assert m.reject_threshold is not None and t < m.reject_threshold
        else:
            forced = "UNSUPPORTED_ESTIMATE" in codes or m.force_review_on_anomalies
            assert t < m.trust_threshold or forced


def test_batch_and_single_predictions_agree(engine, records):
    batch = engine.predict(records)
    singles = [engine.predict([r])[0] for r in records]
    assert batch == singles


def test_predictions_are_deterministic(engine, records):
    assert engine.predict(records) == engine.predict(records)


def test_concurrent_predictions_match_sequential(engine, records):
    sequential = [engine.predict_one(r) for r in records]
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(engine.predict_one, records))
    assert concurrent == sequential


@pytest.mark.parametrize("value", [1e6, 1e308, -1e308])
def test_extreme_values_are_never_auto_trusted(engine, records, value):
    res = engine.predict([{**records[0], "age": value}])[0]
    codes = {w["code"] for w in res["warnings"]}
    assert res["decision"] != "TRUST"
    assert {"OUT_OF_RANGE", "UNSUPPORTED_ESTIMATE"} <= codes
    assert 0.0 <= res["trustworthiness"] <= 1.0


def test_reject_explanation_mentions_abstention(engine, loans):
    res = engine.predict(loans.drop(columns=["default"]).head(1000))
    rejected = [r for r in res if r["decision"] == "REJECT"]
    assert rejected, "expected at least one REJECT among 1,000 loans"
    assert "abstains" in rejected[0]["explanation"]
    assert "reject threshold" in rejected[0]["reasons"][0]


def test_confidence_fallback_mode_is_explained():
    df = load_dataset(str(DATA / "breast_cancer.csv"), "diagnosis")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifacts, _ = train_modeldoctor(df, "diagnosis", TrainingConfig(fast=True))
    assert artifacts["reliability_mode"] == "confidence_fallback"
    eng = InferenceEngine(artifacts)
    res = eng.predict(df.drop(columns=["diagnosis"]).head(5))
    for r in res:
        assert r["trustworthiness"] == pytest.approx(r["confidence"], abs=1e-4)
        assert any("calibrated confidence" in reason for reason in r["reasons"])


# ---------------------------------------------------------------------------
# Model 2 support and mitigation rules (unit level)
# ---------------------------------------------------------------------------

def test_model2_support_clips_and_reports_violations():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X = pd.DataFrame({"a": np.linspace(0, 1, 50), "b": np.linspace(0, 10, 50)})
    y = (X["a"] > 0.5).astype(int)
    pipe = Pipeline([("clf", LogisticRegression())]).fit(X, y)
    det = Model2Detector(pipe, ["a", "b"], "Logistic Regression", 0.9, 0.8).set_support(X)

    probe = pd.DataFrame({"a": [0.5, 50.0, 0.2], "b": [5.0, 5.0, np.inf]})
    assert det.support_violations(probe) == [[], ["a"], ["b"]]
    assert det.support_violations(pd.DataFrame({"a": [-5.0], "b": [-1.0]})) == [["a", "b"]]  # no direction known
    clipped = det.predict_trustworthiness(probe)
    edge = det.predict_trustworthiness(pd.DataFrame({"a": [1.0], "b": [5.0]}))
    assert clipped[1] == pytest.approx(edge[0])


def test_support_violations_only_count_in_the_risky_direction():
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({"knn_distance": np.linspace(1, 2, 40), "max_prob": np.linspace(0.6, 0.9, 40)})
    y = np.tile([0, 1], 20)
    det = Model2Detector(LogisticRegression().fit(X, y), list(X.columns), "LR", 0.5, 0.5).set_support(X)
    probe = pd.DataFrame({"knn_distance": [0.1, 9.0, 1.5, 1.5], "max_prob": [0.75, 0.75, 0.99, 0.2]})
    # closer than any calibration row / more confident than any: safe side -> supported
    assert det.support_violations(probe) == [[], ["knn_distance"], [], ["max_prob"]]


def test_typical_record_is_not_flagged_as_unsupported(engine):
    typical = {f["name"]: (f["median"] if f["type"] == "numeric" else f["most_frequent"]) for f in engine.feature_schema}
    res = engine.predict_one(typical)
    assert "UNSUPPORTED_ESTIMATE" not in {w["code"] for w in res["warnings"]}


def test_model2_without_support_attributes_still_works():
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0]})
    det = Model2Detector(LogisticRegression().fit(X, [0, 1, 0, 1]), ["a"], "LR", 0.5, 0.5)
    assert det.support_violations(X) == [[], [], [], []]
    assert det.predict_trustworthiness(X).shape == (4,)


def test_unsupported_estimate_forces_review():
    m = MitigationEngine(trust_threshold=0.8, reject_threshold=0.4)
    flag = [{"type": "outside_reliability_support", "features": ["knn_distance"]}]
    assert m.decide(0.99, []) == "TRUST"
    assert m.decide(0.99, flag) == "REVIEW"
    assert m.decide(0.1, flag) == "REJECT"
    out = m.evaluate_record(0.99, flag, {"max_prob": 0.9})
    assert "UNSUPPORTED_ESTIMATE" in {w["code"] for w in out["warnings"]}


@pytest.mark.parametrize("trust,decision,expected", [
    (0.95, "TRUST", "at or above the trust threshold"),
    (0.60, "REVIEW", "below the trust threshold 80.0% but not below the reject threshold 40.0%"),
    (0.20, "REJECT", "below the reject threshold 40.0%"),
])
def test_explain_states_the_threshold_comparison(trust, decision, expected):
    m = MitigationEngine(trust_threshold=0.8, reject_threshold=0.4)
    out = m.explain(trust, decision, [{"code": "LOW_TRUSTWORTHINESS", "message": "x"}])
    assert expected in out["reasons"][0]
    assert out["risk_level"] == RISK_LEVELS[decision]
    assert len(out["reasons"]) == 1  # the generic fallback warning is not repeated as a reason
