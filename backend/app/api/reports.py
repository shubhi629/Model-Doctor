import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/report", tags=["Reports"])

STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage"


@router.get("/{session_id}")
def get_report(session_id: str):
    report_path = STORAGE_DIR / session_id / "report.json"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Training report not found."
        )

    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid report file."
        )