"""Pydantic request / response schemas for the ModelDoctor REST API.

Deeply nested chart / diagnostic payloads are typed as dicts to keep the contract
flexible; every key that the dashboard relies on is typed explicitly.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Decision = Literal["TRUST", "REVIEW", "REJECT"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
ReliabilityMode = Literal["model2", "confidence_fallback"]

# Input size guards (far above any realistic tabular record).
MAX_FIELDS_PER_RECORD = 1000
MAX_STRING_VALUE_LENGTH = 1000

FeatureValue = Union[float, int, Annotated[str, StringConstraints(max_length=MAX_STRING_VALUE_LENGTH)], bool, None]
FeatureRecord = Annotated[
    Dict[Annotated[str, StringConstraints(max_length=MAX_STRING_VALUE_LENGTH)], FeatureValue],
    Field(max_length=MAX_FIELDS_PER_RECORD),
]


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    code: str
    request_id: str


class HealthResponse(BaseModel):
    status: str


class ServiceHealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    active_model: Optional[str] = None
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ColumnSummary(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical"]
    missing_count: int
    unique_count: int
    is_target_candidate: bool


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: List[ColumnSummary]
    preview: List[Dict[str, Any]]
    suggested_target: Optional[str]


class ProfileResponse(BaseModel):
    """Output of ml.data.profiler.profile_dataset plus validation result."""
    model_config = ConfigDict(extra="allow")
    session_id: str
    valid: bool
    validation_error: Optional[str] = None
    row_count: int
    column_count: int
    target_column: str
    detected_task: Optional[str] = None
    class_distribution: Dict[str, int] = {}
    warnings: List[str] = []


class SessionSummary(BaseModel):
    session_id: str
    filename: Optional[str]
    created_at: Optional[str]
    target_column: Optional[str]
    rows: Optional[int]
    status: str
    stage: Optional[str]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    target_column: str = Field(..., min_length=1)
    target_risk: float = Field(0.05, gt=0.0, lt=1.0, description="Max error rate among TRUST predictions")
    min_coverage: float = Field(0.30, gt=0.0, le=1.0, description="Minimum share of predictions to TRUST")
    force_model: Optional[Literal["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "KNN"]] = None
    force_review_on_anomalies: bool = False
    fast: Optional[bool] = Field(None, description="Smaller ensembles for quick demos (server default if omitted)")


class TrainStartResponse(BaseModel):
    session_id: str
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    session_id: str
    status: Literal["uploaded", "pending", "running", "completed", "failed"]
    stage: Optional[str] = None
    stages: List[str]
    progress: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class LeaderboardRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    score: Optional[float]
    train_seconds: Optional[float]
    status: str


class ComparisonRow(BaseModel):
    method: str
    coverage: float
    selective_risk: float
    accepted: int
    total: int


class ReportResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    session_id: str
    task: str
    target_column: str
    classes: List[str]
    selected_model: str
    leaderboard: List[LeaderboardRow]
    reliability_mode: Literal["model2", "confidence_fallback"]
    overall_error_rate: float
    comparison: List[ComparisonRow]
    verdict: str
    auroc: Dict[str, Optional[float]]
    auprc: Dict[str, Optional[float]]
    risk_coverage_curve: Dict[str, List[Dict[str, float]]]
    calibration_curve: Dict[str, List[Dict[str, Any]]]
    decision_distribution: List[Dict[str, Any]]
    thresholds: Dict[str, Any]
    profile: Dict[str, Any]


# ---------------------------------------------------------------------------
# Schema / Predict
# ---------------------------------------------------------------------------

class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    type: Literal["numeric", "categorical"]
    missing_rate: float
    integer: Optional[bool] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    categories: Optional[List[str]] = None
    most_frequent: Optional[str] = None


class SchemaResponse(BaseModel):
    session_id: str
    target_column: str
    classes: List[str]
    features: List[FeatureSpec]


class PredictRequest(BaseModel):
    features: FeatureRecord = Field(
        ...,
        description="Raw feature values keyed by column name. Use null for an unknown value.",
        examples=[{"annual_income": 42000, "age": 35, "loan_amount": 15000, "employment_type": "self_employed"}],
    )


WarningCategory = Literal[
    "out_of_distribution", "anomaly", "missing_information", "model_disagreement",
    "uncertainty", "low_trustworthiness", "other",
]


class Warning_(BaseModel):
    code: str
    message: str
    category: WarningCategory = "other"


class PredictResponse(BaseModel):
    prediction_id: str
    prediction: str
    confidence: float
    trustworthiness: float
    decision: Decision
    risk_level: RiskLevel
    explanation: str
    reasons: List[str]
    warnings: List[Warning_]
    reliability_mode: ReliabilityMode
    class_probabilities: Dict[str, float]
    reliability_features: Dict[str, Any]


class BatchPredictResponse(BaseModel):
    session_id: str
    count: int
    ignored_columns: List[str]
    summary: Dict[str, int]
    results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Feedback / Monitoring
# ---------------------------------------------------------------------------

class FeedbackItem(BaseModel):
    prediction_id: str = Field(..., min_length=1)
    actual: Union[str, int, float, bool]


class FeedbackRequest(BaseModel):
    items: List[FeedbackItem] = Field(..., min_length=1)


class FeedbackResponse(BaseModel):
    accepted: int
    rejected: List[Dict[str, str]]
    labeled_total: int


class PredictionLogRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    prediction_id: str
    timestamp: str
    source: str
    prediction: str
    confidence: float
    trustworthiness: float
    decision: Decision
    actual: Optional[str] = None


class MonitoringResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    session_id: str
    total_predictions: int
    labeled_predictions: int
    coverage: Optional[float]
    decision_distribution: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    observed_vs_estimated: Dict[str, Any]
    alerts: List[Dict[str, str]]
    retraining_recommended: bool
    recent_predictions: List[PredictionLogRow]


# ---------------------------------------------------------------------------
# Versioned deployment API (/api/v1)
# ---------------------------------------------------------------------------

class V1PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: FeatureRecord = Field(
        ...,
        description="Raw feature values keyed by column name. Every trained feature must be present; "
                    "use null for an unknown value. Unknown fields are rejected.",
        examples=[{"age": 35, "annual_income": 42000, "loan_amount": 15000, "credit_score": 680,
                   "interest_rate": 14.5, "debt_to_income": 0.2, "employment_years": 6,
                   "home_ownership": "RENT", "loan_purpose": "debt_consolidation", "employment_type": "salaried"}],
    )


class V1BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[FeatureRecord] = Field(
        ..., min_length=1, description="Records to score (limit: MODELDOCTOR_MAX_BATCH_ROWS)",
    )


class V1Prediction(BaseModel):
    prediction_id: str
    prediction: str
    model_confidence: float = Field(..., description="Model 1's calibrated top-class probability")
    trustworthiness: float = Field(..., description="Estimated probability that this prediction is correct")
    decision: Decision
    risk_level: RiskLevel
    explanation: str
    reasons: List[str]
    warnings: List[Warning_]
    class_probabilities: Dict[str, float]


class V1PredictResponse(V1Prediction):
    request_id: str
    model_id: str
    reliability_mode: ReliabilityMode


class V1BatchResponse(BaseModel):
    request_id: str
    model_id: str
    reliability_mode: ReliabilityMode
    count: int
    summary: Dict[str, int]
    results: List[V1Prediction]


class V1ActivateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(..., pattern=r"^[a-f0-9]{32}$")


class V1ModelInfo(BaseModel):
    model_id: str
    pinned: bool
    target_column: str
    task: str
    classes: List[str]
    selected_model: str
    calibration_method: Optional[str] = None
    reliability_mode: ReliabilityMode
    reliability_model: Optional[str] = None
    thresholds: Dict[str, Any]
    features: List[FeatureSpec]
    test_metrics: Dict[str, Any]
    trained_at: Optional[str] = None
    max_batch_rows: int
