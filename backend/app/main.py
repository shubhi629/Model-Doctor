"""FastAPI entry point for ModelDoctor backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/api/health")
async def health_check():
    """Health check endpoint confirming API service status."""
    return {"status": "ok"}

from app.api.predictions import router as predictions_router

app.include_router(predictions_router)
