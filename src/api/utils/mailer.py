"""Appends summary emails to an account's own inbox."""

import logging
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid

from src.config import primary_account

from .folder_rules import inbox_folder
from .imap_client import open_mailbox

logger = logging.getLogger(__name__)


def append_summary_to_inbox(
    body,
    html_body=None,
    subject="SMIO Daily Summary",
    attachments=None,
    account=None,
):
    account = account or primary_account()

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
    message["From"] = formataddr(("Smio, der Mail Organizer", account.email))
    message["To"] = account.email
    message["Reply-To"] = account.email
    # Without these the mail could only be recognised by its UID.
    message["Message-ID"] = make_msgid(domain=account.email.rpartition("@")[2] or None)
    message["Date"] = formatdate(localtime=True)

    try:
        with open_mailbox(account) as mailbox:
            mailbox.append(message.as_bytes(), folder=inbox_folder(account))
        logger.info("Daily summary appended to %s", account.email)
        return True
    except Exception:
        logger.exception("Failed to append daily summary to the IMAP inbox")
        return False
