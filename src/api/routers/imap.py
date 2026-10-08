from fastapi import APIRouter

from src.config import resolve_account
from ..utils.imap_client import fetch_inbox, sync_existing_mails

router = APIRouter(prefix="/imap", tags=["IMAP"])

@router.get("/fetch")
def fetch_sync(account: str | None = None):
    return fetch_inbox(resolve_account(account))


@router.get("/sync")
def sync_mail_folders(account: str | None = None):
    return sync_existing_mails(resolve_account(account))
