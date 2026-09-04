from fastapi import APIRouter
from ..schemas.email import EmailRequest

router = APIRouter(prefix="/model", tags=["Model"])

@router.post("/classify")
def classify_email(email: EmailRequest):
    return {
        "subject": email.subject,
        "body": email.body,
        "category": "unknown",
        "note": "Replace with BERT classifier output"
    }
