"""Unit tests for test-split evaluation and monitoring summaries."""

import json

import numpy as np
import pandas as pd
import pytest

from ml.pipeline import RecordValidationError, coerce_records, prepare_training_frame
from ml.reliability.evaluation import (
    accept_top_k,
    area_under_risk_coverage,
    calibration_curve,
    error_detection_metrics,
    evaluate_system,
    risk_coverage_curve,
)
from ml.reliability.monitoring import compute_monitoring


def test_accept_top_k_selects_highest_scores():
    mask = accept_top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2)
    assert mask.tolist() == [False, True, False, True]
    assert accept_top_k(np.array([0.3, 0.2]), 0).sum() == 0


def test_risk_coverage_curve_endpoints_and_monotone_for_perfect_ranking():
    correct = np.array([1] * 80 + [0] * 20)
    score = np.linspace(1.0, 0.0, 100)  # all correct rows ranked first
    curve = risk_coverage_curve(correct, score, n_points=20)
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["selective_risk"] == pytest.approx(0.20)
    assert all(p["selective_risk"] == 0.0 for p in curve if p["coverage"] <= 0.8)
    # Perfect ranking has lower AURC than reversed ranking
    assert area_under_risk_coverage(correct, score) < area_under_risk_coverage(correct, -score)


def test_error_detection_precision_recall():
    correct = np.array([1, 1, 0, 0, 1])
    flagged = np.array([False, True, True, False, False])
    m = error_detection_metrics(correct, flagged)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["errors_caught"] == 1 and m["errors_total"] == 2


def test_calibration_curve_bins_cover_all_rows():
    rng = np.random.RandomState(0)
    prob = rng.uniform(0, 1, 5000)
    correct = (rng.uniform(0, 1, 5000) < prob).astype(int)
    bins = calibration_curve(correct, prob)
    assert sum(b["count"] for b in bins) == 5000
    for b in bins:
        assert abs(b["mean_predicted"] - b["observed_accuracy"]) < 0.08


def test_evaluate_system_compares_at_same_coverage_and_is_json_serializable():
    rng = np.random.RandomState(1)
    n = 400
    correct = rng.binomial(1, 0.85, n)
    confidence = np.clip(0.7 + 0.2 * correct + rng.normal(0, 0.15, n), 0.5, 1.0)
    trust = np.clip(0.6 + 0.3 * correct + rng.normal(0, 0.10, n), 0.0, 1.0)
    decisions = np.where(trust >= 0.85, "TRUST", np.where(trust < 0.6, "REJECT", "REVIEW")).tolist()

    res = evaluate_system(correct, confidence, trust, decisions)
    json.dumps(res)
    methods = {r["method"]: r for r in res["comparison"]}
    assert set(methods) == {"No filtering", "Confidence baseline", "ModelDoctor"}
    assert methods["No filtering"]["coverage"] == 1.0
    # Fair comparison: baseline evaluated at exactly the same coverage
    assert methods["Confidence baseline"]["accepted"] == methods["ModelDoctor"]["accepted"]
    assert res["auroc"]["modeldoctor"] > res["auroc"]["confidence_baseline"]
    assert "lowers error" in res["verdict"]
    assert sum(d["count"] for d in res["decision_distribution"]) == n
    assert res["risk_coverage_curve"]["modeldoctor"] and res["risk_coverage_curve"]["baseline"]


def test_evaluate_system_fallback_verdict():
    correct = np.array([1, 0, 1, 1])
    conf = np.array([0.9, 0.6, 0.8, 0.95])
    res = evaluate_system(correct, conf, conf, ["TRUST", "REVIEW", "TRUST", "TRUST"],
                          reliability_mode="confidence_fallback")
    assert "fell back" in res["verdict"]


def test_prepare_training_frame_drops_constant_and_id_columns():
    df = pd.DataFrame({
        "id": [f"row{i}" for i in range(50)],
        "const": [7] * 50,
        "flag": [True, False] * 25,
        "num": np.arange(50, dtype=float),
        "cat": ["a", "b"] * 25,
        "y": [0, 1] * 25,
    })
    out, num, cat, dropped = prepare_training_frame(df, "y")
    assert {d["column"] for d in dropped} == {"id", "const"}
    assert "flag" in num and out["flag"].dtype.kind in "iu"
    assert cat == ["cat"]


def test_coerce_records_validation():
    schema = [
        {"name": "age", "type": "numeric"},
        {"name": "color", "type": "categorical"},
    ]
    ok = coerce_records(pd.DataFrame([{"age": "42", "color": 1}]), schema)
    assert ok["age"].iloc[0] == 42.0 and ok["color"].iloc[0] == "1"

    nulls = coerce_records(pd.DataFrame([{"age": None, "color": ""}]), schema)
    assert nulls.isna().all(axis=None)

    with pytest.raises(RecordValidationError, match="Unknown column"):
        coerce_records(pd.DataFrame([{"age": 1, "color": "x", "zzz": 3}]), schema)
    with pytest.raises(RecordValidationError, match="Missing required"):
        coerce_records(pd.DataFrame([{"age": 1}]), schema)
    with pytest.raises(RecordValidationError, match="must be numeric"):
        coerce_records(pd.DataFrame([{"age": "old", "color": "x"}]), schema)


def test_monitoring_summary_and_alerts():
    preds = []
    for i in range(60):
        preds.append({
            "prediction_id": f"p{i}",
            "timestamp": "2026-01-01T00:00:00",
            "prediction": "1",
            "confidence": 0.95,
            "trustworthiness": 0.95,
            "decision": "TRUST",
            "warning_codes": ["OUT_OF_RANGE"] if i % 2 else [],
        })
    # Half of TRUST predictions are wrong -> degradation + miscalibration alerts
    feedback = [{"prediction_id": f"p{i}", "actual": "1" if i % 2 else "0"} for i in range(60)]
    baseline = {"test_trust_accuracy": 0.95, "test_anomaly_rate": 0.05, "test_coverage": 0.5}
    res = compute_monitoring(preds, feedback, baseline)
    json.dumps(res)
    assert res["total_predictions"] == 60 and res["labeled_predictions"] == 60
    assert res["observed_vs_estimated"]["observed_accuracy"] == 0.5
    codes = {a["code"] for a in res["alerts"]}
    assert {"TRUST_DEGRADATION", "MISCALIBRATION", "INPUT_DRIFT"} <= codes
    assert res["retraining_recommended"] is True
    assert len(res["timeline"]) >= 2

    empty = compute_monitoring([], [], baseline)
    assert empty["total_predictions"] == 0 and empty["alerts"] == []


def test_reject_band_does_not_swallow_review_when_base_error_is_high():
    """Regression: with ~35% overall error, REJECT must not take every non-TRUST row."""
    from ml.reliability.mitigation import tune_mitigation_thresholds

    rng = np.random.RandomState(0)
    n = 1000
    oof = rng.uniform(0.3, 0.95, n)
    # Error probability ~ 1 - score, but compressed so the low band is ~45-55% wrong
    y_correct = (rng.uniform(0, 1, n) < 0.30 + 0.6 * oof).astype(int)
    assert 0.25 < 1 - y_correct.mean() < 0.40

    t = tune_mitigation_thresholds(oof, y_correct, target_risk=0.05, min_coverage=0.30)
    bands = {b["name"]: b for b in t["band_metadata"]}
    if t["reject_threshold"] is not None:
        # The REJECT band must be clearly worse than the overall error, not just the leftover rows
        assert bands["REJECT"]["observed_error_rate"] >= 0.5
        assert bands["REVIEW"]["count"] > 30
