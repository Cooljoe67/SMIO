from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.ai.daily_summary import build_daily_summary, stored_daily_summary
from src.db.database import get_db

router = APIRouter(prefix="/summary", tags=["Summary"])

@router.get("/daily")
def daily_summary(summary_date: date | None = None, db: Session = Depends(get_db)):
    return stored_daily_summary(db, summary_date) or build_daily_summary(
        db,
        summary_date=summary_date,
    )
