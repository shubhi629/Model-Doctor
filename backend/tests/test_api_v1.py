"""API tests for /health and the versioned deployment API (/api/v1), plus the
end-to-end pipeline: upload -> train -> deploy -> predict -> monitoring."""

import io
import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import sessions
from app.core.config import settings
from app.main import app
from ml.inference import InferenceEngine

LOANS = Path(__file__).resolve().parents[2] / "sample_data" / "loans.csv"
DECISIONS = {"TRUST", "REVIEW", "REJECT"}
RISK = {"TRUST": "LOW", "REVIEW": "MEDIUM", "REJECT": "HIGH"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    original = settings.storage_dir, settings.model_session
    settings.storage_dir = tmp_path_factory.mktemp("storage_v1")
    settings.model_session = None
    with TestClient(app) as c:
        yield c
    settings.storage_dir, settings.model_session = original


def _upload(client, df, name="data.csv"):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/api/datasets/upload", files={"file": (name, buf, "text/csv")})
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


@pytest.fixture(scope="module")
def loans():
    return pd.read_csv(LOANS)


@pytest.fixture(scope="module")
def deployed(client, loans):
    """Train loans.csv once; the first completed model is activated automatically."""
    no_model = client.get("/health").json()
    sid = _upload(client, loans, "loans.csv")
    r = client.post(f"/api/train/{sid}", json={"target_column": "default", "fast": True})
    assert r.status_code == 202, r.text
    assert client.get(f"/api/train/{sid}/status").json()["status"] == "completed"
    return sid, no_model


def _json_safe(rec):
    """NaN -> None (JSON has no NaN), numpy scalars -> Python."""
    return {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in rec.items()}


def _features(loans, i=0):
    return _json_safe(loans.drop(columns=["default"]).iloc[i].to_dict())


def _assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["code"] == code
    assert body["request_id"] == r.headers["x-request-id"]
    assert "Traceback" not in r.text and 'File "' not in r.text


# ---------------------------------------------------------------------------
# Health and model info
# ---------------------------------------------------------------------------

def test_health_before_any_model(deployed):
    _, before = deployed
    assert before["status"] == "ok"
    assert before["model_loaded"] is False
    assert before["active_model"] is None


def test_health_with_deployed_model(client, deployed):
    body = client.get("/health").json()
    assert body == {**body, "status": "ok", "model_loaded": True, "active_model": deployed[0]}
    assert client.get("/api/health").json() == {"status": "ok"}


def test_model_info_reports_real_artifacts_and_test_metrics(client, deployed):
    sid = deployed[0]
    info = client.get("/api/v1/model/info").json()
    report = client.get(f"/api/report/{sid}").json()
    assert info["model_id"] == sid and info["pinned"] is False
    assert info["target_column"] == "default"
    assert info["selected_model"] == report["selected_model"]
    assert info["reliability_mode"] == report["reliability_mode"]
    assert info["thresholds"]["trust_threshold"] == report["thresholds"]["trust_threshold"]
    assert info["test_metrics"]["comparison"] == report["comparison"]
    assert info["test_metrics"]["overall_error_rate"] == report["overall_error_rate"]
    assert [f["name"] for f in info["features"]] == [f["name"] for f in client.get(f"/api/schema/{sid}").json()["features"]]


def test_openapi_documents_v1(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/api/v1/model/info", "/api/v1/model/active", "/api/v1/predict", "/api/v1/predict/batch"):
        assert p in paths


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

def test_predict_returns_full_contract(client, deployed, loans):
    r = client.post("/api/v1/predict", json={"features": _features(loans)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == r.headers["x-request-id"]
    assert body["model_id"] == deployed[0]
    assert body["decision"] in DECISIONS and body["risk_level"] == RISK[body["decision"]]
    assert 0 <= body["model_confidence"] <= 1 and 0 <= body["trustworthiness"] <= 1
    assert body["explanation"] and body["reasons"]
    assert set(body["class_probabilities"]) == {"0", "1"}
    assert body["prediction"] in {"0", "1"}


def test_v1_and_session_endpoint_and_engine_agree(client, deployed, loans):
    sid, feats = deployed[0], _features(loans, 3)
    v1 = client.post("/api/v1/predict", json={"features": feats}).json()
    session = client.post(f"/api/predict/{sid}", json={"features": feats}).json()
    direct = InferenceEngine.from_directory(settings.storage_dir / sid).predict_one(feats)
    for res in (session, direct):
        assert res["decision"] == v1["decision"]
        assert res["trustworthiness"] == v1["trustworthiness"]
        assert res["confidence"] == v1["model_confidence"]
        assert res["explanation"] == v1["explanation"]
    assert session["risk_level"] == v1["risk_level"] and session["reasons"] == v1["reasons"]


def test_client_request_id_is_honoured_and_unsafe_ones_replaced(client, deployed, loans):
    r = client.post("/api/v1/predict", json={"features": _features(loans)}, headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and r.json()["request_id"] == "abc-123"
    r = client.get("/health", headers={"X-Request-ID": "bad id\nwith newline" * 10})
    assert r.headers["x-request-id"] != "bad id" and len(r.headers["x-request-id"]) == 32


def test_duplicate_requests_give_identical_results(client, deployed, loans):
    a = client.post("/api/v1/predict", json={"features": _features(loans, 5)}).json()
    b = client.post("/api/v1/predict", json={"features": _features(loans, 5)}).json()
    assert a["prediction_id"] != b["prediction_id"] and a["request_id"] != b["request_id"]
    strip = lambda d: {k: v for k, v in d.items() if k not in ("prediction_id", "request_id")}  # noqa: E731
    assert strip(a) == strip(b)


def test_out_of_distribution_record_is_flagged(client, deployed, loans):
    feats = {**_features(loans), "annual_income": 1e12, "employment_type": "astronaut", "credit_score": None}
    body = client.post("/api/v1/predict", json={"features": feats}).json()
    cats = {w["category"] for w in body["warnings"]}
    assert body["decision"] != "TRUST"
    assert {"out_of_distribution", "missing_information"} <= cats


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate,fragment", [
    (lambda f: {k: v for k, v in f.items() if k != "age"}, "missing required field"),
    (lambda f: {**f, "unexpected": 1}, "unknown field"),
    (lambda f: {**f, "age": "old"}, "must be numeric"),
    (lambda f: {**f, "age": [1, 2]}, "features.age"),
    (lambda f: {**f, "loan_purpose": "x" * 5000}, "at most 1000 characters"),
    (lambda f: {}, "missing required field"),
])
def test_predict_validation_errors(client, deployed, loans, mutate, fragment):
    r = client.post("/api/v1/predict", json={"features": mutate(_features(loans))})
    _assert_error(r, 422, "validation_error")
    assert fragment in r.json()["detail"]


def test_infinity_in_json_is_rejected(client, deployed, loans):
    payload = json.dumps({"features": {**_features(loans), "age": 1}}).replace('"age": 1', '"age": Infinity')
    r = client.post("/api/v1/predict", content=payload, headers={"content-type": "application/json"})
    _assert_error(r, 422, "validation_error")
    assert "finite" in r.json()["detail"]


def test_unexpected_top_level_field_rejected(client, deployed, loans):
    r = client.post("/api/v1/predict", json={"features": _features(loans), "debug": True})
    _assert_error(r, 422, "validation_error")


@pytest.mark.parametrize("content", ["", "{not json", '{"features": {"age": 1,}}'])
def test_malformed_or_empty_body(client, deployed, content):
    r = client.post("/api/v1/predict", content=content, headers={"content-type": "application/json"})
    assert r.status_code in (400, 422), r.text
    assert r.json()["code"] in ("malformed_json", "validation_error")
    if content:
        _assert_error(r, 400, "malformed_json")


def test_oversized_body_rejected(client, deployed, loans, monkeypatch):
    monkeypatch.setattr(settings, "max_json_mb", 0.001)  # ~1 KB
    records = [_features(loans, i) for i in range(20)]
    r = client.post("/api/v1/predict/batch", json={"records": records})
    _assert_error(r, 413, "too_large")


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def test_batch_prediction_matches_single_predictions(client, deployed, loans):
    records = [_features(loans, i) for i in range(30)]
    r = client.post("/api/v1/predict/batch", json={"records": records})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 30 and sum(body["summary"].values()) == 30
    assert body["summary"] == {d: sum(1 for x in body["results"] if x["decision"] == d) for d in DECISIONS}
    for i in (0, 7, 29):
        single = client.post("/api/v1/predict", json={"features": records[i]}).json()
        assert single["decision"] == body["results"][i]["decision"]
        assert single["trustworthiness"] == body["results"][i]["trustworthiness"]


def test_batch_with_invalid_record_scores_nothing(client, deployed, loans):
    sid = deployed[0]
    before = len(sessions.read_jsonl(sid, "predictions.jsonl"))
    bad = _features(loans, 1)
    bad.pop("credit_score")
    r = client.post("/api/v1/predict/batch", json={"records": [_features(loans, 0), bad]})
    _assert_error(r, 422, "validation_error")
    assert "Record 1" in r.json()["detail"]
    assert len(sessions.read_jsonl(sid, "predictions.jsonl")) == before


def test_empty_batch_rejected(client, deployed):
    _assert_error(client.post("/api/v1/predict/batch", json={"records": []}), 422, "validation_error")


def test_batch_row_limit(client, deployed, loans, monkeypatch):
    monkeypatch.setattr(settings, "max_batch_rows", 5)
    r = client.post("/api/v1/predict/batch", json={"records": [_features(loans, i) for i in range(6)]})
    _assert_error(r, 413, "too_large")
    assert "limit is 5" in r.json()["detail"]


def test_large_batch_within_limit(client, deployed, loans):
    records = [_json_safe(r) for r in loans.drop(columns=["default"]).head(2000).to_dict(orient="records")]
    r = client.post("/api/v1/predict/batch", json={"records": records})
    assert r.status_code == 200 and r.json()["count"] == 2000


def test_concurrent_requests(client, deployed, loans):
    sid = deployed[0]
    before = len(sessions.read_jsonl(sid, "predictions.jsonl"))
    payloads = [{"features": _features(loans, i)} for i in range(24)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda p: client.post("/api/v1/predict", json=p), payloads))
    assert all(r.status_code == 200 for r in responses)
    assert len({r.json()["request_id"] for r in responses}) == 24
    assert len(sessions.read_jsonl(sid, "predictions.jsonl")) == before + 24


# ---------------------------------------------------------------------------
# Integration: predictions reach monitoring with the same decisions
# ---------------------------------------------------------------------------

def test_v1_predictions_feed_monitoring(client, deployed, loans):
    sid = deployed[0]
    body = client.post("/api/v1/predict/batch", json={"records": [_features(loans, i) for i in range(40, 50)]}).json()
    mon = client.get(f"/api/monitoring/{sid}", params={"recent": 1000}).json()
    logged = {p["prediction_id"]: p for p in mon["recent_predictions"]}
    for res in body["results"]:
        row = logged[res["prediction_id"]]
        assert row["decision"] == res["decision"] and row["trustworthiness"] == res["trustworthiness"]
        assert row["source"] == "api_batch" and row["request_id"] == body["request_id"]
    counts = {d["decision"]: d["count"] for d in mon["decision_distribution"]}
    assert sum(counts.values()) == mon["total_predictions"]
    assert sum(h["count"] for h in mon["trustworthiness_histogram"]) == mon["total_predictions"]
    for h in mon["confidence_histogram"]:
        assert h["count"] == h["TRUST"] + h["REVIEW"] + h["REJECT"]
    assert mon["mean_confidence"] is not None and mon["mean_trustworthiness"] is not None


# ---------------------------------------------------------------------------
# Activation, missing / corrupt models, internal errors
# ---------------------------------------------------------------------------

def test_activate_errors(client, deployed, loans, monkeypatch):
    _assert_error(client.put("/api/v1/model/active", json={"session_id": uuid.uuid4().hex}), 404, "not_found")
    _assert_error(client.put("/api/v1/model/active", json={"session_id": "../../etc"}), 422, "validation_error")
    untrained = _upload(client, loans.head(600))
    _assert_error(client.put("/api/v1/model/active", json={"session_id": untrained}), 409, "conflict")
    monkeypatch.setattr(settings, "model_session", deployed[0])
    _assert_error(client.put("/api/v1/model/active", json={"session_id": deployed[0]}), 409, "model_pinned")
    assert client.get("/api/v1/model/info").json()["pinned"] is True


@pytest.fixture
def corrupt_copy(client, deployed):
    """A copy of the trained session whose Model 2 file is corrupted, activated for v1."""
    src = settings.storage_dir / deployed[0]
    sid = uuid.uuid4().hex
    shutil.copytree(src, settings.storage_dir / sid)
    (settings.storage_dir / sid / "model2.joblib").write_bytes(b"corrupted")
    sessions.write_json_file(settings.storage_dir / "active_model.json", {"session_id": sid})
    yield sid
    sessions.write_json_file(settings.storage_dir / "active_model.json", {"session_id": deployed[0]})
    sessions.delete_session(sid)


def test_corrupted_model_returns_503_and_degraded_health(client, corrupt_copy, loans):
    r = client.post("/api/v1/predict", json={"features": _features(loans)})
    _assert_error(r, 503, "model_unavailable")
    assert "model2" in r.json()["detail"]
    health = client.get("/health").json()
    assert health["status"] == "degraded" and health["model_loaded"] is False
    _assert_error(client.post(f"/api/predict/{corrupt_copy}", json={"features": _features(loans)}),
                  503, "model_unavailable")


def test_deleted_active_model_returns_503(client, deployed, loans):
    sid = uuid.uuid4().hex
    sessions.write_json_file(settings.storage_dir / "active_model.json", {"session_id": sid})
    try:
        _assert_error(client.post("/api/v1/predict", json={"features": _features(loans)}), 503, "no_active_model")
        assert client.get("/health").json()["status"] == "ok"
    finally:
        sessions.write_json_file(settings.storage_dir / "active_model.json", {"session_id": deployed[0]})


def test_internal_error_hides_details(deployed, loans, monkeypatch):
    def boom(self, X):
        raise RuntimeError("secret internal state /etc/passwd")

    monkeypatch.setattr(InferenceEngine, "predict_validated", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/api/v1/predict", json={"features": _features(loans)})
    _assert_error(r, 500, "internal_error")
    assert "secret" not in r.text and "RuntimeError" not in r.text


def test_path_traversal_session_ids_are_rejected(client):
    for path in ("/api/schema/..%2F..%2Fetc", "/api/report/....", "/api/predict/../../etc"):
        r = client.get(path) if "predict" not in path else client.post(path, json={"features": {"a": 1}})
        assert r.status_code in (404, 405), (path, r.status_code)


def test_cors_allows_configured_origin_only(client):
    ok = client.options("/api/v1/predict", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/api/v1/predict", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert "x-request-id" in r.headers.get("access-control-expose-headers", "").lower()


def test_streamed_body_without_content_length_is_limited(client, deployed, loans, monkeypatch):
    monkeypatch.setattr(settings, "max_json_mb", 0.001)
    payload = json.dumps({"records": [_features(loans, i) for i in range(20)]}).encode()
    chunks = (payload[i:i + 256] for i in range(0, len(payload), 256))
    r = client.post("/api/v1/predict/batch", content=chunks, headers={"content-type": "application/json"})
    _assert_error(r, 413, "too_large")
