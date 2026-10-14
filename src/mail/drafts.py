"""Reply drafts for stored mails: the create_draft step of action rules."""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid, parseaddr

from src.ai.reply_draft import write_reply
from src.ai.reply_status import mail_text
from src.config import get_config
from src.db.models import Email

from . import open_mailbox

logger = logging.getLogger(__name__)

# Guards for mails SMIO itself sorts into a draft folder, so a backlog sorted
# at once (e.g. when a folder list's model arrives) gets no drafts to old or
# answered mail. A user's own move always gets a draft.
MAX_AGE_DAYS = int(os.getenv("SMIO_DRAFT_MAX_AGE_DAYS", "7"))
# Drafts per dispatch; the rest wait for the next run (src/mail/events.py).
MAX_PER_RUN = int(os.getenv("SMIO_DRAFT_MAX_PER_RUN", "10"))

_REPLY_PREFIX = re.compile(r"^\s*(re|aw|antw)\s*:", re.IGNORECASE)


def _quote(email):
    text = email.text if (email.text or "").strip() else mail_text(email, max_chars=20000)
    when = email.date.strftime("%d.%m.%Y um %H:%M") if email.date else "unbekanntem Datum"
    quoted = "\n".join(f"> {line}" if line else ">" for line in (text or "").splitlines())
    return f"Am {when} schrieb {email.sender or 'unbekannt'}:\n{quoted}"


def build_reply(email, account, body, event_key=None):
    """A reply to email in the same thread, with the original quoted below body."""
    subject = email.subject or ""
    message = EmailMessage()
    message["Subject"] = subject if _REPLY_PREFIX.match(subject) else f"Re: {subject}"
    message["From"] = account.email
    message["To"] = formataddr(parseaddr(email.sender or ""))
    if email.message_id:
        message["In-Reply-To"] = email.message_id
        references = (email.reference_ids or "").split()
        message["References"] = " ".join([*references, email.message_id])
    message["Message-ID"] = make_msgid(domain=account.email.rpartition("@")[2] or None)
    message["Date"] = formatdate(localtime=True)
    if event_key:
        # Which event made SMIO write this draft.
        message["X-SMIO-Event"] = event_key
    message.set_content(f"{body}\n\n{_quote(email)}\n")
    return message


def skip_reason(db, event, email):
    """Why the event gets no draft, or None. Only SMIO's own moves are checked."""
    if event.source != "system":
        return None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if email.date and email.date < now - timedelta(days=MAX_AGE_DAYS):
        return f"older than {MAX_AGE_DAYS} days"
    answered = (
        db.query(Email.id)
        .filter(Email.reply_to_id == email.id, Email.direction == "OUTBOUND")
        .first()
    )
    return "already answered" if answered else None


def create_reply_draft(db, event, options):
    """Write a reply to the event's mail with Claude Haiku and store it as a draft."""
    email = db.get(Email, event.email_id)
    if email is None or email.direction == "OUTBOUND" or not email.sender:
        logger.info("Event %s: no draft for a mail without an outside sender", event.event_key)
        return
    reason = skip_reason(db, event, email)
    if reason:
        logger.info("Event %s: no draft for email %s: %s", event.event_key, email.id, reason)
        return
    account = get_config().account(event.account)
    signer = account.user_first_name if account.user_first_name != "there" else None
    # Log in and find the drafts folder before Haiku is paid for the text.
    with open_mailbox(account) as mailbox:
        if mailbox.drafts_folder() is None:
            raise RuntimeError(f"Account {account.id!r} has no drafts folder")
        body = write_reply(email, signer=signer, instructions=options.instructions)
        raw = build_reply(email, account, body, event.event_key).as_bytes()
        mailbox.create_draft(raw)
    logger.info("Event %s: reply draft for email %s stored", event.event_key, email.id)
