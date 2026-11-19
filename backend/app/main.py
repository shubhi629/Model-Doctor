"""FastAPI entry point for ModelDoctor backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.training import router as training_router
from app.api.predictions import router as predictions_router
from app.api.datasets import router as datasets_router
from app.api.reports import router as reports_router


app = FastAPI(
    title="ModelDoctor API",
    description="Intelligent Detection and Mitigation of Unreliable ML Predictions",
    version="0.1.0",
)

# Enable CORS for frontend development
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(training_router)
app.include_router(predictions_router)
app.include_router(datasets_router)
app.include_router(reports_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}