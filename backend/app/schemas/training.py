from typing import Literal, Optional
from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    target_column: str
    target_risk: float = Field(default=0.05, ge=0, le=1)
    min_coverage: float = Field(default=0.30, gt=0, le=1)


class TrainingStartResponse(BaseModel):
    session_id: str
    status: Literal["pending"]
    message: str


class TrainingStatusResponse(BaseModel):
    session_id: str
    status: Literal["pending", "running", "completed", "failed"]
    stage: Optional[str] = None
    progress: int = Field(default=0, ge=0, le=100)
    error: Optional[str] = None