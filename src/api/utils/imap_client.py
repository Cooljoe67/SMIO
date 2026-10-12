"""Commands SMIO runs on a mailbox for stored mails: sorting, undo, retention.

Each command writes the new state to the database at once and records a
"system" event (src/mail/events.py), so the next sync sees no change.
"""

import logging
from datetime import datetime, timedelta, timezone

from src.db.models import Email
from src.mail.events import record

from .mail_identity import is_same_mail
from .folder_rules import (
    folder_for_classification,
    folder_path,
    folder_rules,
    inbox_folder,
    trash_folder,
)

logger = logging.getLogger(__name__)


def _verify_uid(mailbox, email, folder):
    """Re-read the message at the stored UID and refuse if it is another mail now.

    UIDs change when a folder is renumbered (UIDVALIDITY), so a stale UID must
    never be moved or trashed blindly. Returns the message, or None if gone.
    """
    if not email.uid:
        raise ValueError(f"Email {email.id} has no IMAP UID")
    message = mailbox.get(folder, email.uid)
    if message is not None and not is_same_mail(
        email, message.message_id, message.subject, message.date
    ):
        raise ValueError(f"Email {email.id}: UID {email.uid} in {folder} now holds another mail")
    return message


def move_email_to_classification_folder(mailbox, email, account):
    destination = folder_for_classification(email.classification, account)
    if destination is None:
        return None
    source_folder = email.folder or inbox_folder(account)
    if source_folder == destination:
        # Already there; a MOVE within one folder would only renumber it.
        return None
    if _verify_uid(mailbox, email, source_folder) is None:
        # Moved or deleted since the sync; the next sync records which. Raising
        # would leave it unprocessed and fail again on every run.
        logger.warning("Email %s is no longer in %s; not moving it", email.id, source_folder)
        return None

    mailbox.move(source_folder, [email.uid], destination)
    if not email.provider_message_id:
        # IMAP: the destination assigns a new UID; the next sync records it.
        email.uid = None
    return destination


def move_email_to_inbox(mailbox, email, account):
    """Move a mail back to the inbox and return its inbox UID (None if unknown)."""
    inbox = inbox_folder(account)
    if not email.uid:
        raise ValueError(f"Email {email.id} has no IMAP UID")
    if not email.folder or email.folder == inbox:
        return email.uid

    source_folder = email.folder
    if _verify_uid(mailbox, email, source_folder) is None:
        raise ValueError(f"Email {email.id} is no longer in {source_folder}")
    mailbox.move(source_folder, [email.uid], inbox)

    # UIDs are per folder: the old one may now belong to another INBOX mail.
    if not email.message_id:
        return None
    uids = mailbox.find(inbox, email.message_id)
    return uids[-1] if uids else None


def _move_to_trash(mailbox, email, account):
    if not email.uid or not email.folder:
        raise ValueError(f"Email {email.id} has no folder or IMAP UID")
    mailbox.trash(email.folder, [email.uid])
    # The trash assigns its own UID; the old one must not be reused.
    email.uid = None


def cleanup_expired_mails(db, mailbox, account):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    moved = 0
    skipped_unread = 0
    errors = []

    for classification, rule in folder_rules(account).items():
        retention_days = rule.retention_days
        retention_days_unread = rule.retention_days_unread
        folder_name = folder_path(classification, account)
        active_periods = [
            days for days in (retention_days, retention_days_unread)
            if days is not None
        ]
        if not active_periods:
            continue

        candidate_cutoff = max(
            now - timedelta(days=days) for days in active_periods
        )
        candidates = db.query(Email).filter(
            Email.account == account.id,
            Email.folder == folder_name,
            Email.date <= candidate_cutoff,
            Email.is_training_data.is_(False),
        ).all()

        for email in candidates:
            if not email.uid:
                errors.append({"email_id": email.id, "error": "missing_imap_uid"})
                logger.warning(
                    "Skipping retention cleanup for email %s in %s: no IMAP UID",
                    email.id,
                    folder_name,
                )
                continue

            try:
                current_message = _verify_uid(mailbox, email, folder_name)
                if current_message is None:
                    continue

                is_seen = current_message.seen
                applicable_period = retention_days if is_seen else retention_days_unread
                if applicable_period is None or email.date > now - timedelta(days=applicable_period):
                    if not is_seen:
                        skipped_unread += 1
                    continue

                _move_to_trash(mailbox, email, account)
                record(
                    db, account, email, "removed",
                    from_folder=folder_name, to_folder=trash_folder(account), source="system",
                )
                email.folder = trash_folder(account)
                email.removed_at = now
                email.removal_reason = "retention"
                moved += 1
                logger.info(
                    "Email %s moved from %s to %s after %s retention days (%s)",
                    email.id,
                    folder_name,
                    trash_folder(account),
                    applicable_period,
                    "read" if is_seen else "unread",
                )
            except Exception as error:
                errors.append({"email_id": email.id, "error": str(error)})
                logger.exception("Could not move expired email %s to trash", email.id)

    return {
        "moved": moved,
        "skipped_unread": skipped_unread,
        "errors": errors,
    }
