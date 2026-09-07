# src/api/routers/model.py
from fastapi import APIRouter
from src.db.database import SessionLocal
from src.db.models import Email
from src.ai.model import classify_email

router = APIRouter(prefix="/model", tags=["AI"])

@router.post("/run")
def run_model():
    db = SessionLocal()
    emails = db.query(Email).filter(Email.processed == False).all()

    for email in emails:
        label, confidence = classify_email(email.text or "")
        email.classification = label
        email.confidence = str(confidence)
        email.processed = True

    db.commit()
    return {"processed": len(emails)}
