from typing import Any
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: list[dict[str, Any]]
