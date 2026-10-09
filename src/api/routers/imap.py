from fastapi import APIRouter, Depends

from ..dependencies import requested_account
from ..utils.imap_client import fetch_inbox, sync_existing_mails

router = APIRouter(prefix="/imap", tags=["IMAP"])

@router.get("/fetch")
def fetch_sync(account=Depends(requested_account)):
    # The new rows are detached from their session here, so report the count.
    return {"fetched": len(fetch_inbox(account))}


@router.get("/sync")
def sync_mail_folders(account=Depends(requested_account)):
    return sync_existing_mails(account)
