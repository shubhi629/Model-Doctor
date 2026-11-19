from typing import Any
from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    session_id: str
    filename: str
    columns: list[str]
    preview: list[dict[str, Any]]
    row_count: int


class DatasetProfileResponse(BaseModel):
    row_count: int
    column_count: int
    feature_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    columns: list[dict[str, Any]]
    target_column: str
    target_dtype: str
    detected_task: str
    class_distribution: dict[str, int]
    class_percentages: dict[str, float]
    imbalance_ratio: float
    warnings: list[str]