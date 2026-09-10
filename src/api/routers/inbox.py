from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from src.db.database import get_db
from ..utils.inbox_processor import process_email_by_id

router = APIRouter(prefix="/inbox", tags=["Inbox"])


@router.post("/process_email/{email_id}")
def process_email(email_id: int, db: Session = Depends(get_db)):
    result = process_email_by_id(email_id, db)

    if result is None:
        raise HTTPException(status_code=404, detail="Email not found")

    return result
