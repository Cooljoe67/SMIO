"""Sends summary emails through the Gmail API."""

import base64
import logging
from email.mime.text import MIMEText
from email.utils import formataddr

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from .settings import gmail_settings

logger = logging.getLogger(__name__)

GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


def send_summary_email(body, subject="SMIO Daily Summary"):
    if not all((
        gmail_settings.client_id,
        gmail_settings.client_secret,
        gmail_settings.refresh_token,
        gmail_settings.to_address,
    )):
        logger.info("Gmail API not configured, skipping summary email")
        return False

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = formataddr((gmail_settings.from_name, gmail_settings.from_address))
    message["To"] = gmail_settings.to_address

    try:
        credentials = Credentials(
            token=None,
            refresh_token=gmail_settings.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=gmail_settings.client_id,
            client_secret=gmail_settings.client_secret,
            scopes=[GMAIL_SEND_SCOPE],
        )
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        service.users().messages().send(
            userId="me",
            body={"raw": encoded_message},
        ).execute()
        logger.info("Daily summary email sent to %s", gmail_settings.to_address)
        return True
    except Exception:
        logger.exception("Failed to send daily summary email with Gmail API")
        return False
