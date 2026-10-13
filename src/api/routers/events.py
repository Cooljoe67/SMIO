"""Push notifications from mail providers."""

import base64
import json
import logging

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from src.scheduler.workflows import run_gmail_push

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["Mail events"])


@router.post("/gmail")
def gmail_push(envelope: dict = Body(...)):
    """A Gmail notification from the Pub/Sub push subscription: message.data is
    base64 JSON with emailAddress and historyId.

    Cloud Run lets only callers with run.invoker in, here the subscription's
    service account. A 2xx answer acknowledges the message; 503 makes Pub/Sub
    deliver it again later.
    """
    try:
        notification = json.loads(base64.b64decode(envelope["message"]["data"]))
        address = notification["emailAddress"]
        history_id = str(notification["historyId"])
    except (KeyError, TypeError, ValueError):
        # Delivering it again would not help.
        logger.warning("Ignoring a malformed Gmail push notification")
        return {"ignored": "malformed"}
    result = run_gmail_push(address, history_id)
    if result.get("reason") == "workflow_in_progress":
        return JSONResponse(status_code=503, content=result)
    return result
