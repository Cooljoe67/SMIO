"""Appends summary emails to the configured IMAP inbox."""

import logging
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.utils import formataddr

from imap_tools import MailBox

from .settings import email_settings

logger = logging.getLogger(__name__)


def append_summary_to_inbox(
    body,
    html_body=None,
    subject="SMIO Daily Summary",
    attachments=None,
):
    if not all((email_settings.host, email_settings.user, email_settings.password)):
        logger.info("IMAP not configured, skipping summary inbox append")
        return False

    if attachments:
        message = MIMEMultipart("mixed")
        if html_body:
            alternative = MIMEMultipart("alternative")
            alternative.attach(MIMEText(body, "plain", "utf-8"))
            alternative.attach(MIMEText(html_body, "html", "utf-8"))
            message.attach(alternative)
        else:
            message.attach(MIMEText(body, "plain", "utf-8"))
        for filename, content in attachments:
            attachment = MIMEApplication(content, _subtype="octet-stream")
            attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename=filename,
            )
            message.attach(attachment)
    elif html_body:
        message = MIMEMultipart("alternative")
        message.attach(MIMEText(body, "plain", "utf-8"))
        message.attach(MIMEText(html_body, "html", "utf-8"))
    else:
        message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = formataddr(("Smio, der Mail Organizer", email_settings.user))
    message["To"] = email_settings.user
    message["Reply-To"] = email_settings.user

    try:
        with MailBox(email_settings.host).login(
            email_settings.user,
            email_settings.password,
        ) as mailbox:
            mailbox.append(message.as_bytes(), folder="INBOX")
        logger.info("Daily summary appended to %s", email_settings.user)
        return True
    except Exception:
        logger.exception("Failed to append daily summary to the IMAP inbox")
        return False
