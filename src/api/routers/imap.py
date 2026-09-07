from fastapi import APIRouter
from ..utils.imap_client import fetch_inbox

router = APIRouter(prefix="/imap", tags=["IMAP"])

@router.get("/fetch")
def fetch_sync():
    return fetch_inbox()
