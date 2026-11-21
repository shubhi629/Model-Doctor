"""API tests for every endpoint (including error cases) and an end-to-end flow.

Uses a temporary storage directory and fast training mode (smaller ensembles).
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

LOANS = Path(__file__).resolve().parents[2] / "sample_data" / "loans.csv"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    original = settings.storage_dir
    settings.storage_dir = tmp_path_factory.mktemp("storage")
    with TestClient(app) as c:
        yield c
    settings.storage_dir = original


def _upload(client, df: pd.DataFrame, name="data.csv"):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    return client.post("/api/datasets/upload", files={"file": (name, buf, "text/csv")})


@pytest.fixture(scope="module")
def trained_session(client):
    """Upload loans.csv and train once (fast mode); reused by downstream tests."""
    df = pd.read_csv(LOANS)
    r = _upload(client, df, "loans.csv")
    assert r.status_code == 201, r.text
    sid = r.json()["session_id"]
    r = client.post(f"/api/train/{sid}", json={"target_column": "default", "fast": True})
    assert r.status_code == 202, r.text
    # TestClient runs background tasks before returning, so training is finished here
    status = client.get(f"/api/train/{sid}/status").json()
    assert status["status"] == "completed", status
    return sid, df


# ---------------------------------------------------------------------------
# Upload / profile
# ---------------------------------------------------------------------------

def test_upload_returns_columns_preview_and_suggested_target(client):
    df = pd.read_csv(LOANS).head(600)
    r = _upload(client, df)
    assert r.status_code == 201
    body = r.json()
    assert body["rows"] == 600
    assert len(body["preview"]) == 10
    assert body["suggested_target"] == "default"
    assert {c["name"] for c in body["columns"]} == set(df.columns)


def test_upload_rejects_non_csv_and_empty(client):
    r = client.post("/api/datasets/upload", files={"file": ("x.json", b"{}", "application/json")})
    assert r.status_code == 400
    r = client.post("/api/datasets/upload", files={"file": ("x.csv", b"", "text/csv")})
    assert r.status_code == 400
    assert "detail" in r.json() and "Traceback" not in r.text


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0.0001)
    r = _upload(client, pd.read_csv(LOANS).head(600))
    assert r.status_code == 413


def test_profile_valid_and_invalid_targets(client):
    sid = _upload(client, pd.read_csv(LOANS).head(600)).json()["session_id"]
    r = client.get(f"/api/datasets/{sid}/profile", params={"target_column": "default"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True and body["detected_task"] == "binary_classification"
    assert body["row_count"] == 600

    # Continuous target -> profile works but marked invalid with the regression message
    r = client.get(f"/api/datasets/{sid}/profile", params={"target_column": "annual_income"})
    assert r.status_code == 200
    assert r.json()["valid"] is False
    assert "Only classification targets are supported" in r.json()["validation_error"]

    r = client.get(f"/api/datasets/{sid}/profile", params={"target_column": "nope"})
    assert r.status_code == 400


def test_unknown_session_returns_404(client):
    for url in ["/api/datasets/" + "0" * 32 + "/profile", "/api/train/" + "0" * 32 + "/status",
                "/api/report/" + "0" * 32, "/api/schema/../../etc"]:
        assert client.get(url).status_code == 404, url
    r = client.post("/api/predict/" + "a" * 32, json={"features": {"x": 1}})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_train_rejects_invalid_target(client):
    sid = _upload(client, pd.read_csv(LOANS).head(600)).json()["session_id"]
    r = client.post(f"/api/train/{sid}", json={"target_column": "annual_income"})
    assert r.status_code == 400
    assert "Only classification targets are supported in this version." in r.json()["detail"]
    r = client.post(f"/api/train/{sid}", json={"target_column": "missing_col"})
    assert r.status_code == 400
    r = client.post(f"/api/train/{sid}", json={"target_column": "default", "target_risk": 5})
    assert r.status_code == 422


def test_too_few_rows_rejected(client):
    sid = _upload(client, pd.read_csv(LOANS).head(100)).json()["session_id"]
    r = client.post(f"/api/train/{sid}", json={"target_column": "default"})
    assert r.status_code == 400 and "500 rows" in r.json()["detail"]


def test_report_and_predict_before_training_return_409(client):
    sid = _upload(client, pd.read_csv(LOANS).head(600)).json()["session_id"]
    assert client.get(f"/api/report/{sid}").status_code == 409
    assert client.get(f"/api/schema/{sid}").status_code == 409
    assert client.post(f"/api/predict/{sid}", json={"features": {"age": 30}}).status_code == 409
    assert client.get(f"/api/monitoring/{sid}").status_code == 409


def test_status_after_training(client, trained_session):
    sid, _ = trained_session
    s = client.get(f"/api/train/{sid}/status").json()
    assert s["status"] == "completed" and s["progress"] == 1.0
    assert s["stages"] == ["profiling", "splitting", "model_selection", "reliability_features",
                           "reliability_training", "evaluation", "saving"]


# ---------------------------------------------------------------------------
# Report / schema
# ---------------------------------------------------------------------------

def test_report_shape(client, trained_session):
    sid, _ = trained_session
    r = client.get(f"/api/report/{sid}")
    assert r.status_code == 200
    rep = r.json()
    assert rep["task"] == "classification"
    assert rep["selected_model"] in {"Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "KNN"}
    assert len(rep["leaderboard"]) == 5
    methods = [c["method"] for c in rep["comparison"]]
    assert methods == ["No filtering", "Confidence baseline", "ModelDoctor"]
    assert rep["comparison"][1]["coverage"] == rep["comparison"][2]["coverage"]
    assert set(rep["auroc"]) == {"confidence_baseline", "modeldoctor"}
    assert rep["risk_coverage_curve"]["modeldoctor"] and rep["risk_coverage_curve"]["baseline"]
    assert rep["calibration_curve"]["modeldoctor"]
    assert rep["reliability_mode"] == "model2"
    # Leakage: Model 2 never sees the target, the predicted class, or label-like columns
    assert "default" not in rep["model2"]["feature_names"]
    assert "predicted_class" not in rep["model2"]["feature_names"]
    # Splits are 60/20/20 and disjoint by construction
    sp = rep["splits"]
    assert (sp["train"]["rows"], sp["calibration"]["rows"], sp["test"]["rows"]) == (3000, 1000, 1000)


def test_schema(client, trained_session):
    sid, df = trained_session
    r = client.get(f"/api/schema/{sid}")
    assert r.status_code == 200
    body = r.json()
    names = [f["name"] for f in body["features"]]
    assert "default" not in names
    assert set(names) == set(df.columns) - {"default"}
    emp = next(f for f in body["features"] if f["name"] == "employment_type")
    assert emp["type"] == "categorical" and "salaried" in emp["categories"]
    inc = next(f for f in body["features"] if f["name"] == "annual_income")
    assert inc["type"] == "numeric" and inc["min"] < inc["max"]
    assert body["classes"] == ["0", "1"]


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def _typical_record(df):
    rec = df.drop(columns=["default"]).iloc[0].to_dict()
    return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in rec.items()}


def test_predict_single(client, trained_session):
    sid, df = trained_session
    r = client.post(f"/api/predict/{sid}", json={"features": _typical_record(df)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {"0", "1"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["trustworthiness"] <= 1.0
    assert body["decision"] in {"TRUST", "REVIEW", "REJECT"}
    assert isinstance(body["warnings"], list)
    assert body["prediction_id"]


def test_predict_anomalous_record_gets_warnings(client, trained_session):
    sid, df = trained_session
    rec = _typical_record(df)
    rec.update({"annual_income": 9_000_000, "credit_score": None, "employment_type": "astronaut"})
    body = client.post(f"/api/predict/{sid}", json={"features": rec}).json()
    codes = {w["code"] for w in body["warnings"]}
    assert "OUT_OF_RANGE" in codes
    assert "UNSEEN_CATEGORY" in codes
    assert codes & {"MISSING_FIELD", "MISSING_FIELDS"}


def test_predict_validation_errors(client, trained_session):
    sid, df = trained_session
    rec = _typical_record(df)
    r = client.post(f"/api/predict/{sid}", json={"features": {**rec, "bogus": 1}})
    assert r.status_code == 422 and "Unknown column" in r.json()["detail"]
    partial = dict(rec)
    partial.pop("age")
    r = client.post(f"/api/predict/{sid}", json={"features": partial})
    assert r.status_code == 422 and "Missing required" in r.json()["detail"]
    r = client.post(f"/api/predict/{sid}", json={"features": {**rec, "age": "forty"}})
    assert r.status_code == 422 and "must be numeric" in r.json()["detail"]
    r = client.post(f"/api/predict/{sid}", json={"wrong": {}})
    assert r.status_code == 422


def test_batch_predict_with_labels_feeds_monitoring(client, trained_session):
    sid, df = trained_session
    batch = df.sample(80, random_state=3)
    buf = io.BytesIO(batch.to_csv(index=False).encode())
    r = client.post(f"/api/predict/{sid}/batch", files={"file": ("batch.csv", buf, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 80
    assert body["summary"]["labels_recorded"] == 80
    assert body["ignored_columns"] == ["default"]
    assert sum(body["summary"][d] for d in ["TRUST", "REVIEW", "REJECT"]) == 80

    mon = client.get(f"/api/monitoring/{sid}").json()
    assert mon["labeled_predictions"] >= 80
    assert mon["total_predictions"] >= 80
    assert mon["observed_vs_estimated"]["observed_accuracy"] is not None
    assert len(mon["decision_distribution"]) == 3


def test_batch_missing_columns(client, trained_session):
    sid, df = trained_session
    buf = io.BytesIO(df[["age", "annual_income"]].head(5).to_csv(index=False).encode())
    r = client.post(f"/api/predict/{sid}/batch", files={"file": ("b.csv", buf, "text/csv")})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Feedback / monitoring
# ---------------------------------------------------------------------------

def test_feedback_flow(client, trained_session):
    sid, df = trained_session
    pid = client.post(f"/api/predict/{sid}", json={"features": _typical_record(df)}).json()["prediction_id"]
    r = client.post(f"/api/feedback/{sid}", json={"items": [
        {"prediction_id": pid, "actual": 1},
        {"prediction_id": "does-not-exist", "actual": "1"},
        {"prediction_id": pid, "actual": "maybe"},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == 1
    assert len(body["rejected"]) == 2

    mon = client.get(f"/api/monitoring/{sid}").json()
    row = next(p for p in mon["recent_predictions"] if p["prediction_id"] == pid)
    assert row["actual"] == "1"

    assert client.post(f"/api/feedback/{sid}", json={"items": []}).status_code == 422


# ---------------------------------------------------------------------------
# Small / multiclass dataset: insufficient-signal fallback path end to end
# ---------------------------------------------------------------------------

def test_multiclass_small_dataset_end_to_end(client):
    rng = np.random.RandomState(0)
    n = 600
    x1, x2 = rng.randn(n), rng.randn(n)
    y = np.where(x1 + x2 > 0.8, "high", np.where(x1 + x2 < -0.8, "low", "mid"))
    df = pd.DataFrame({"x1": x1, "x2": x2, "grp": rng.choice(["a", "b", "c"], n), "tier": y})
    sid = _upload(client, df).json()["session_id"]
    r = client.post(f"/api/train/{sid}", json={"target_column": "tier", "fast": True})
    assert r.status_code == 202
    status = client.get(f"/api/train/{sid}/status").json()
    assert status["status"] == "completed", status
    rep = client.get(f"/api/report/{sid}").json()
    assert rep["task_detail"] == "multiclass_classification"
    assert rep["reliability_mode"] in {"model2", "confidence_fallback"}
    r = client.post(f"/api/predict/{sid}", json={"features": {"x1": 0.1, "x2": 0.2, "grp": "a"}})
    assert r.status_code == 200 and r.json()["prediction"] in {"high", "low", "mid"}


def test_sessions_list_and_delete(client):
    sid = _upload(client, pd.read_csv(LOANS).head(600)).json()["session_id"]
    ids = [s["session_id"] for s in client.get("/api/sessions").json()]
    assert sid in ids
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/datasets/{sid}/profile").status_code == 404


def test_monitoring_awaiting_only_lists_unlabeled(client, trained_session):
    sid, df = trained_session
    pid = client.post(f"/api/predict/{sid}", json={"features": _typical_record(df)}).json()["prediction_id"]
    rows = client.get(f"/api/monitoring/{sid}", params={"awaiting_only": True}).json()["recent_predictions"]
    assert rows and all(r["actual"] is None for r in rows)
    assert pid in {r["prediction_id"] for r in rows}
