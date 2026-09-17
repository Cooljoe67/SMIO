from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.ai.daily_summary import build_daily_summary
from src.db.database import get_db

router = APIRouter(prefix="/summary", tags=["Summary"])

@router.get("/daily")
def daily_summary(db: Session = Depends(get_db)):
    return build_daily_summary(db)
