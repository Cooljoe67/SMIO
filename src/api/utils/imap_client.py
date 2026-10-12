import logging
import os
import re
from datetime import date, datetime, timedelta, timezone

from src.db.database import SessionLocal
from src.db.models import Email, MailSyncState
from src.mail import CursorExpired, open_mailbox

from .mail_identity import fingerprint, is_same_mail, normalized_date
from .folder_rules import (
    classification_for_folder,
    folder_for_classification,
    folder_path,
    folder_rules,
    inbox_folder,
    trash_folder,
)

logger = logging.getLogger(__name__)

# Sent mails are only needed to recognise replies, so only recent ones are read.
SENT_SYNC_DAYS = int(os.getenv("SMIO_SENT_SYNC_DAYS", "30"))
# Between complete reads, a sync only reads what arrived in each folder. Only a
# complete read notices mails that left (deleted, or moved to an unwatched
# folder) and mails marked read.
FULL_SYNC_MINUTES = int(os.getenv("SMIO_FULL_SYNC_MINUTES", "60"))


def _message_ids(value):
    return re.findall(r"<[^<>\s]+>", value or "")


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
    # The destination assigns a new UID; the next sync records it.
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


def _folder_names(mailbox, account):
    """Classification folders first, the inbox last (see the copy rule in _sync_folders)."""
    # Disabled folders are still scanned so corrections and retention keep working.
    known_folders = {
        folder_path(classification, account) for classification in folder_rules(account)
    }
    return [
        *[folder for folder in mailbox.folders() if folder in known_folders],
        inbox_folder(account),
    ]


def _identity(account, msg):
    return msg.message_id, fingerprint(
        account.id, msg.sender, msg.subject, msg.date, msg.message_id, msg.text, msg.html
    )


def _find_email(db, account, message_id, mail_fingerprint, subject, date, provider_id=None):
    """Stored mail for a server message: by provider id, Message-ID, else fingerprint."""
    emails = db.query(Email).filter(Email.account == account.id)
    if provider_id:
        email_obj = emails.filter(Email.provider_message_id == provider_id).first()
        if email_obj is not None:
            return email_obj
    if message_id:
        for email_obj in emails.filter(Email.message_id == message_id).order_by(Email.id):
            # Some senders reuse Message-IDs; a different subject or a different
            # Date (which a move never changes) is another mail.
            if email_obj.fingerprint in (None, mail_fingerprint) or (
                email_obj.subject == subject
                and (
                    email_obj.date is None
                    or date is None
                    or normalized_date(email_obj.date) == normalized_date(date)
                )
            ):
                return email_obj
    return emails.filter(Email.fingerprint == mail_fingerprint).first()


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


def _sync_message(
    db, account, msg, folder_name, allow_new, seen_ids=(), new_emails=None, outbound=False
):
    message_id, mail_fingerprint = _identity(account, msg)
    email_obj = _find_email(
        db, account, message_id, mail_fingerprint, msg.subject, msg.date, msg.provider_id
    )
    if email_obj is not None and email_obj.id in seen_ids:
        # A copy of a mail already seen in this sync in a folder scanned earlier.
        return None
    new_email = email_obj is None
    if new_email and not allow_new:
        return None
    if new_email:
        email_obj = Email(account=account.id)
        if outbound:
            # Sent by the account: never classified, moved or counted as received.
            email_obj.direction = "OUTBOUND"
            email_obj.processed = 1
            email_obj.processing_batch = 0
    elif (
        email_obj.direction == "OUTBOUND"
        and folder_name == inbox_folder(account)
        and email_obj.classification is None
    ):
        # Sent to oneself, and the Sent copy was synced in an earlier run: the
        # inbox copy is received mail and is processed like any other.
        email_obj.direction = None
        email_obj.processed = 0
        email_obj.processing_batch = None
    email_obj.message_id = message_id
    email_obj.fingerprint = mail_fingerprint
    if msg.provider_id:
        email_obj.provider_message_id = msg.provider_id
    in_reply_to = _message_ids(msg.header("In-Reply-To"))
    email_obj.in_reply_to = in_reply_to[0] if in_reply_to else None
    email_obj.reference_ids = " ".join(_message_ids(msg.header("References"))) or None

    sync_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
    email_obj.last_seen_at = sync_timestamp
    if email_obj.removal_reason == "external":
        email_obj.removed_at = None
        email_obj.removal_reason = None
    previous_folder = email_obj.folder
    previous_classification = email_obj.classification
    email_obj.uid = msg.ref
    email_obj.folder = folder_name
    email_obj.subject = msg.subject
    email_obj.sender = msg.sender
    email_obj.date = msg.date
    email_obj.text = msg.text
    email_obj.html = msg.html
    if msg.seen and email_obj.read_at is None:
        email_obj.read_at = datetime.now(timezone.utc).replace(tzinfo=None)
        logger.info("Email %s first observed as read", email_obj.id)

    folder_classification = classification_for_folder(folder_name, account)
    previous_folder_classification = classification_for_folder(previous_folder or "", account)
    if not new_email and folder_classification and int(email_obj.processed or 0) <= 0:
        # Filed by the user before SMIO processed it: the folder is the label.
        # No processing batch, so undo leaves it where the user put it.
        email_obj.processed = 1
        email_obj.processed_at = sync_timestamp
        email_obj.classification = folder_classification
        email_obj.true_label = folder_classification
        email_obj.classification_source = "imap_folder"
        email_obj.folder_list = account.folder_list
    elif (
        not new_email
        and folder_classification
        and previous_classification
        and previous_classification != folder_classification
    ):
        email_obj.true_label = folder_classification
        email_obj.classification_source = "imap_folder"
        email_obj.folder_list = account.folder_list
    elif (
        not new_email
        and folder_name == inbox_folder(account)
        and previous_folder_classification is not None
        and email_obj.classification_source != "undo"
    ):
        email_obj.classification = "other"
        email_obj.true_label = "other"
        email_obj.classification_source = "imap_folder"
        email_obj.folder_list = account.folder_list

    if folder_classification and email_obj.classification != folder_classification:
        email_obj.classification = folder_classification
        email_obj.classification_source = "imap_folder"
        email_obj.folder_list = account.folder_list
    if new_email:
        db.add(email_obj)
        # Make the new row visible to the next lookup in this sync (autoflush is off).
        db.flush()
        if new_emails is not None:
            new_emails.append(email_obj)
    return email_obj


def _full_read_due(state, now):
    return (
        state is None
        or not state.cursor
        or state.full_sync_at is None
        or state.full_sync_at <= now - timedelta(minutes=FULL_SYNC_MINUTES)
    )


def _read(mailbox, folder, state, full, now, since=None):
    """A folder's messages and whether they were read in full rather than from
    the cursor on. A full read with since is complete only for that window."""
    if not full and not _full_read_due(state, now):
        try:
            return mailbox.read_changes(folder, state.cursor), False
        except CursorExpired:
            logger.info("Cursor of %s expired; reading it completely", folder)
    return mailbox.read_folder(folder, since=since), True


def _save_cursor(db, account, states, folder_state, read_in_full, sync_started):
    state = states.get(folder_state.folder)
    if state is None:
        state = MailSyncState(account=account.id, folder=folder_state.folder)
        db.add(state)
        states[folder_state.folder] = state
    state.cursor = folder_state.cursor
    state.updated_at = sync_started
    if read_in_full:
        state.full_sync_at = sync_started


def _sync_folders(account, allow_new, full=False):
    """Reconcile the inbox and classification folders; new mails only from the inbox.

    Each folder is read completely when due (FULL_SYNC_MINUTES, a new or
    expired cursor, or full=True), otherwise only from its cursor on. Mails
    stored in a completely read folder and not seen are marked removed.

    Copy rule: a mail present in several scanned folders keeps the first one
    seen. Classification folders are scanned before the inbox, so a copy left
    in the inbox neither moves the mail back nor counts as a correction.
    Sent mails come last, so a mail sent to oneself keeps its inbox copy; only
    the last SENT_SYNC_DAYS are read, as they only serve to recognise replies.
    """
    with open_mailbox(account) as mailbox:
        db = SessionLocal()
        try:
            sync_started = datetime.now(timezone.utc).replace(tzinfo=None)
            states = {
                state.folder: state
                for state in db.query(MailSyncState).filter(MailSyncState.account == account.id)
            }
            inbox = inbox_folder(account)
            checked = 0
            changed = 0
            new_emails = []
            seen_ids = set()
            complete_folders = set()

            def sync_folder(read, allow_new_here, outbound=False):
                nonlocal checked, changed
                folder_state, read_in_full = read
                new_count = len(new_emails)
                for msg in folder_state.messages:
                    checked += 0 if outbound else 1
                    email_obj = _sync_message(
                        db, account, msg, folder_state.folder, allow_new=allow_new_here,
                        seen_ids=seen_ids, new_emails=new_emails, outbound=outbound,
                    )
                    if email_obj is None:
                        continue
                    seen_ids.add(email_obj.id)
                    if not outbound and len(new_emails) == new_count:
                        changed += 1
                    new_count = len(new_emails)
                if folder_state.complete:
                    complete_folders.add(folder_state.folder)
                _save_cursor(db, account, states, folder_state, read_in_full, sync_started)

            for folder_name in _folder_names(mailbox, account):
                sync_folder(
                    _read(mailbox, folder_name, states.get(folder_name), full, sync_started),
                    allow_new and folder_name == inbox,
                )

            sent = mailbox.sent_folder()
            if sent not in mailbox.folders():
                sent = None
            new_before_sent = len(new_emails)
            if sent:
                since = date.today() - timedelta(days=SENT_SYNC_DAYS)
                sync_folder(
                    _read(mailbox, sent, states.get(sent), full, sync_started, since=since),
                    allow_new, outbound=True,
                )
            sent_result = {"folder": sent, "new": len(new_emails) - new_before_sent}
            del new_emails[new_before_sent:]

            # The stale check runs in SQL: write this sync's last_seen_at and
            # folders first (autoflush is off), or every mail seen counts as gone.
            db.flush()
            stale_emails = db.query(Email).filter(
                Email.account == account.id,
                Email.folder.in_(complete_folders),
                Email.is_training_data.is_(False),
                (Email.last_seen_at.is_(None) | (Email.last_seen_at < sync_started)),
            ).all()
            for email in stale_emails:
                email.removed_at = sync_started
                email.removal_reason = "external"
            db.commit()
            cleanup = cleanup_expired_mails(db, mailbox, account)
            db.commit()
            return {
                "checked": checked,
                "changed": changed,
                "complete_folders": sorted(complete_folders),
                "sent": sent_result,
                "cleanup": cleanup,
                "new": new_emails,
            }
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def sync_existing_mails(account):
    """Read every folder completely and reconcile the known messages."""
    result = _sync_folders(account, allow_new=False, full=True)
    del result["new"]
    return result


def fetch_inbox(account):
    """Sync all folders and store new inbox messages; returns the new mails."""
    return _sync_folders(account, allow_new=True)["new"]
