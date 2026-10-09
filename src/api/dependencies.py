from fastapi import HTTPException

from src.config import resolve_account


def requested_account(account: str | None = None):
    """The ?account= of a manual endpoint, or the primary account; 404 if unknown."""
    try:
        return resolve_account(account)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
