from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.core.model_service import get_model_service

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        service = get_model_service()
        results = service.predict(request.records)

        return {"predictions": results}

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
