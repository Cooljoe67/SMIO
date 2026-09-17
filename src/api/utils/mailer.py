"""Sends the daily summary as a plain-text email via SMTP."""

import logging
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

from .settings import smtp_settings

logger = logging.getLogger(__name__)


def send_summary_email(body, subject="SMIO Daily Summary"):
    if not smtp_settings.host or not smtp_settings.to_address:
        logger.info("SMTP not configured, skipping summary email")
        return False

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = formataddr((smtp_settings.from_name, smtp_settings.user))
    message["To"] = smtp_settings.to_address

    try:
        with smtplib.SMTP_SSL(smtp_settings.host, smtp_settings.port) as server:
            server.login(smtp_settings.user, smtp_settings.password)
            server.send_message(message)
        logger.info("Daily summary email sent to %s", smtp_settings.to_address)
        return True
    except Exception:
        logger.exception("Failed to send daily summary email")
        return False
