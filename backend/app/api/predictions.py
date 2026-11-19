from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.core.model_service import get_model_service
from app.core.session_prediction_service import predict_session

router = APIRouter(prefix="/api", tags=["Predictions"])


# Prediction using an uploaded and trained dataset
@router.post("/predict/{session_id}", response_model=PredictionResponse)
def predict_uploaded_session(
    session_id: str,
    request: PredictionRequest,
):
    try:
        results = predict_session(session_id, request.records)
        return {"predictions": results}

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# Existing prediction endpoint (keep it working)
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