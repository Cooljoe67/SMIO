from fastapi import APIRouter
from ..utils.imap_client import fetch_inbox

router = APIRouter(prefix="/imap", tags=["IMAP"])

@router.get("/fetch")
def fetch_emails():
    emails = fetch_inbox()
    return {
        "count": len(emails),
        "emails": emails
    }
from fastapi import APIRouter
from ..utils.imap_client import fetch_inbox

router = APIRouter(prefix="/imap", tags=["IMAP"])

@router.get("/fetch")
def fetch_emails():
    emails = fetch_inbox()
    return {
        "count": len(emails),
        "emails": emails
    }
