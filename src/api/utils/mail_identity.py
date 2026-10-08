"""Recognising the same mail across folders, moves and sync runs.

A mail is identified, in this order, by the provider's own stable id (Gmail
message id, Exchange ImmutableId; IMAP has none), its Message-ID header, and a
fingerprint of headers that do not change when the mail is moved. IMAP UIDs are
not identities: they are per folder and change on every move.
"""

import hashlib
import re
from datetime import datetime
from email.utils import parseaddr

# imap_tools reports a missing Date header as this date.
MISSING_DATE = datetime(1900, 1, 1)


def normalized_date(value):
    """Wall-clock time as written in the Date header, the way the DB stores it."""
    if value is None:
        return ""
    value = value.replace(tzinfo=None, microsecond=0)
    return "" if value <= MISSING_DATE else value.isoformat()


def _normalized_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def fingerprint(account_id, sender, subject, date, message_id, text="", html=""):
    """Hash of the parts of a mail that a move leaves unchanged.

    The body is only included when there is no Message-ID: providers decode and
    convert bodies differently, so headers are the more stable choice.
    """
    parts = [
        account_id or "",
        parseaddr(sender or "")[1].casefold(),
        _normalized_text(subject),
        normalized_date(date),
        (message_id or "").strip(),
    ]
    if not parts[4]:
        parts.append(_normalized_text(text) or _normalized_text(html))
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def email_fingerprint(email):
    return fingerprint(
        email.account, email.sender, email.subject, email.date,
        email.message_id, email.text, email.html,
    )


def is_same_mail(email, message_id, subject, date):
    """Whether a server message still is this stored mail (header comparison)."""
    if email.message_id:
        return message_id == email.message_id
    return (
        _normalized_text(subject) == _normalized_text(email.subject)
        and normalized_date(date) == normalized_date(email.date)
    )
