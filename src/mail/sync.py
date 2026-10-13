"""Change detector: compares what a provider's adapter reads with the stored
mails, updates the stored copy and records a normalized event for every change
(src/mail/events.py). The same code serves every provider; adapters only read.

What SMIO itself changes (moving a classified mail, undo, retention) is written
to the database by that command at once, together with a "system" event, so
the server's echo of it matches the stored state and is no change here.
"""

import logging
import os
import re
from datetime import date, datetime, timedelta, timezone

from src.api.utils.folder_rules import folder_path, folder_rules, inbox_folder
from src.api.utils.imap_client import cleanup_expired_mails
from src.api.utils.mail_identity import fingerprint, normalized_date
from src.api.utils.mailer import OWN_MAIL_HEADER
from src.db.database import SessionLocal
from src.db.models import Email, MailSyncState

from . import CursorExpired, open_mailbox
from .events import record

logger = logging.getLogger(__name__)

# Sent mails are only needed to recognise replies, so only recent ones are read.
SENT_SYNC_DAYS = int(os.getenv("SMIO_SENT_SYNC_DAYS", "30"))
# Between complete reads, a sync only reads what arrived in each folder. Only a
# complete read notices mails that left (deleted, or moved to an unwatched
# folder) and mails marked read.
FULL_SYNC_MINUTES = int(os.getenv("SMIO_FULL_SYNC_MINUTES", "60"))
# Providers that list stubs (Gmail) fetch the content of at most this many new
# mails per account and run, so a large first import stays within their rate
# limits; the rest follows in the next runs.
MAX_FETCHES = int(os.getenv("SMIO_SYNC_MAX_FETCHES", "500"))


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _message_ids(value):
    return re.findall(r"<[^<>\s]+>", value or "")


def _folder_names(mailbox, account):
    """Classification folders first, the inbox last (see the copy rule in sync_account)."""
    # Disabled and manual folders are scanned too, for corrections, retention and events.
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


def _find_by_provider_id(db, account, provider_id):
    if not provider_id:
        return None
    return db.query(Email).filter(
        Email.account == account.id, Email.provider_message_id == provider_id
    ).first()


def _find_email(db, account, message_id, mail_fingerprint, subject, date, provider_id=None):
    """Stored mail for a server message: by provider id, Message-ID, else fingerprint."""
    email_obj = _find_by_provider_id(db, account, provider_id)
    if email_obj is not None:
        return email_obj
    emails = db.query(Email).filter(Email.account == account.id)
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


def observe(
    db, account, msg, folder_name, allow_new, seen_ids=(), new_emails=None, outbound=False,
    initial=False, fetches=None,
):
    """Store a message read from folder_name and record what changed; initial
    for the folder's first read, whose events trigger no actions.

    Returns the stored mail, or None for a copy already seen in this sync, a
    new mail that may not be stored here (allow_new), or a stub beyond the
    run's fetches ({"left": n, "truncated": folders}).
    """
    email_obj = None
    if not msg.loaded:
        # A stub: its content is fetched only for a mail not stored yet.
        email_obj = _find_by_provider_id(db, account, msg.provider_id)
        if email_obj is None:
            if not (allow_new and msg.importable):
                return None
            if fetches is not None:
                if fetches["left"] <= 0:
                    fetches["truncated"].add(folder_name)
                    return None
                fetches["left"] -= 1
            msg = msg.load()
            if msg is None:
                return None
    if msg.loaded:
        message_id, mail_fingerprint = _identity(account, msg)
        email_obj = _find_email(
            db, account, message_id, mail_fingerprint, msg.subject, msg.date, msg.provider_id
        )
    if email_obj is not None and email_obj.id in seen_ids:
        # A copy of a mail already seen in this sync in a folder scanned earlier.
        return None
    new_email = email_obj is None
    if new_email and not (allow_new and msg.importable):
        return None
    received = new_email
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
        received = True
    if msg.provider_id:
        email_obj.provider_message_id = msg.provider_id
    if msg.loaded:
        email_obj.message_id = message_id
        email_obj.fingerprint = mail_fingerprint
        in_reply_to = _message_ids(msg.header("In-Reply-To"))
        email_obj.in_reply_to = in_reply_to[0] if in_reply_to else None
        email_obj.reference_ids = " ".join(_message_ids(msg.header("References"))) or None

    email_obj.last_seen_at = _utcnow()
    if email_obj.removal_reason == "external":
        # Marked removed by an earlier sync that missed it; it never left.
        email_obj.removed_at = None
        email_obj.removal_reason = None
    previous_folder = email_obj.folder
    email_obj.uid = msg.ref
    email_obj.folder = folder_name
    if msg.loaded:
        email_obj.subject = msg.subject
        email_obj.sender = msg.sender
        email_obj.date = msg.date
        email_obj.text = msg.text
        email_obj.html = msg.html
    first_read = msg.seen and email_obj.read_at is None
    if first_read:
        email_obj.read_at = _utcnow()
        logger.info("Email %s first observed as read", email_obj.id)

    if new_email and not outbound and msg.header(OWN_MAIL_HEADER):
        # Written by SMIO itself (summary, notice): never classified or moved.
        email_obj.processed = 1
        email_obj.processed_at = _utcnow()
        email_obj.classification = "other"
        email_obj.classification_source = "system"
        email_obj.folder_list = account.folder_list
    if new_email:
        db.add(email_obj)
        # Make the new row visible to the next lookup in this sync (autoflush is off).
        db.flush()
        if new_emails is not None:
            new_emails.append(email_obj)
    if received:
        record(
            db, account, email_obj, "sent" if outbound else "received",
            to_folder=folder_name, initial=initial,
        )
    elif previous_folder != folder_name:
        record(
            db, account, email_obj, "moved",
            from_folder=previous_folder, to_folder=folder_name, initial=initial,
        )
    if first_read:
        record(db, account, email_obj, "read", to_folder=folder_name, initial=initial)
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


def sync_account(account, allow_new, full=False):
    """Reconcile the inbox and the folder list's folders; new mails only from the inbox.

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
            sync_started = _utcnow()
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
            initial_folders = set()
            fetches = {"left": MAX_FETCHES, "truncated": set()}

            def sync_folder(read, allow_new_here, outbound=False):
                nonlocal checked, changed
                folder_state, read_in_full = read
                initial = folder_state.folder not in states
                if initial:
                    initial_folders.add(folder_state.folder)
                new_count = len(new_emails)
                for msg in folder_state.messages:
                    checked += 0 if outbound else 1
                    email_obj = observe(
                        db, account, msg, folder_state.folder, allow_new=allow_new_here,
                        seen_ids=seen_ids, new_emails=new_emails, outbound=outbound,
                        initial=initial, fetches=fetches,
                    )
                    if email_obj is None:
                        continue
                    seen_ids.add(email_obj.id)
                    if not outbound and len(new_emails) == new_count:
                        changed += 1
                    new_count = len(new_emails)
                if folder_state.complete:
                    complete_folders.add(folder_state.folder)
                if folder_state.folder not in fetches["truncated"]:
                    _save_cursor(db, account, states, folder_state, read_in_full, sync_started)
                elif not initial:
                    # Read it completely again next run to fetch the rest.
                    _save_cursor(db, account, states, folder_state, False, sync_started)
                    states[folder_state.folder].full_sync_at = None
                # A truncated first read gets no state: it stays a first read, whose
                # mail triggers no actions, until all of it is fetched.

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
                Email.removed_at.is_(None),
                (Email.last_seen_at.is_(None) | (Email.last_seen_at < sync_started)),
            ).all()
            for email in stale_emails:
                email.removed_at = sync_started
                email.removal_reason = "external"
                record(
                    db, account, email, "removed",
                    from_folder=email.folder, initial=email.folder in initial_folders,
                )
            db.commit()
            cleanup = cleanup_expired_mails(db, mailbox, account)
            db.commit()
            if fetches["truncated"]:
                logger.info(
                    "Account %s: %d new mails fetched; the rest of %s follows next run",
                    account.id, MAX_FETCHES, sorted(fetches["truncated"]),
                )
            return {
                "checked": checked,
                "changed": changed,
                "complete_folders": sorted(complete_folders),
                "fetch_pending": sorted(fetches["truncated"]),
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
    result = sync_account(account, allow_new=False, full=True)
    del result["new"]
    return result


def fetch_inbox(account):
    """Sync all folders and store new inbox messages; returns the new mails."""
    return sync_account(account, allow_new=True)["new"]
