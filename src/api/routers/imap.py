from fastapi import APIRouter
from ..utils.imap_client import fetch_inbox, sync_existing_mails

router = APIRouter(prefix="/imap", tags=["IMAP"])

@router.get("/fetch")
def fetch_sync():
    return fetch_inbox()


@router.get("/sync")
def sync_mail_folders():
    return sync_existing_mails()
