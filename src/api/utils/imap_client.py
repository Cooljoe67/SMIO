import logging
import os
import re
from datetime import date, datetime, timedelta, timezone

from imap_tools import AND, H, MailBox, MailboxLoginError
from src.db.database import SessionLocal
from src.db.models import Email
from src.storage import secret_store

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


def open_mailbox(account):
    """Log in to an account's mailbox; use as a context manager."""
    if account.unavailable:
        raise RuntimeError(f"Account {account.id!r} is unavailable: {account.unavailable}")
    if account.provider != "imap":
        raise NotImplementedError(
            f"Account {account.id!r}: provider {account.provider!r} is not supported yet"
        )
    imap = account.imap
    if imap.password_secret is None:
        if not imap.password:
            raise RuntimeError(
                f"Account {account.id!r}: no IMAP password; set the variable its config refers to"
            )
        return MailBox(imap.host).login(imap.username, imap.password)

    password = secret_store.read_secret(imap.password_secret)
    try:
        return MailBox(imap.host).login(imap.username, password)
    except MailboxLoginError:
        # The password may have been rotated since it was cached.
        secret_store.invalidate(imap.password_secret)
        current = secret_store.read_secret(imap.password_secret)
        if current == password:
            raise
        return MailBox(imap.host).login(imap.username, current)


def _header(message, name):
    headers = getattr(message, "headers", {}) or {}
    value = headers.get(name) or ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip()


def _message_id(message):
    return _header(message, "Message-ID") or _header(message, "Message-Id")


def _message_ids(value):
    return re.findall(r"<[^<>\s]+>", value or "")


def _sent_folder(mailbox, account):
    """The account's sent folder: configured, else the one flagged \\Sent."""
    if account.mailbox.sent:
        return account.mailbox.sent
    for folder in mailbox.folder.list():
        if "\\Sent" in (getattr(folder, "flags", ()) or ()):
            return folder.name
    return None


def _is_seen(message):
    flags = getattr(message, "flags", ()) or ()
    return any(
        (flag.decode(errors="ignore") if isinstance(flag, bytes) else str(flag)).lower()
        == "\\seen"
        for flag in flags
    )


def _verify_uid(mailbox, email, folder):
    """Re-read the message at the stored UID and refuse if it is another mail now.

    UIDs change when a folder is renumbered (UIDVALIDITY), so a stale UID must
    never be moved or trashed blindly. Returns the message, or None if gone.
    """
    if not email.uid:
        raise ValueError(f"Email {email.id} has no IMAP UID")
    mailbox.folder.set(folder, readonly=True)
    message = next(
        mailbox.fetch(uid_list=[email.uid], mark_seen=False, headers_only=True),
        None,
    )
    if message is not None and not is_same_mail(
        email, _message_id(message), message.subject, message.date
    ):
        raise ValueError(f"Email {email.id}: UID {email.uid} in {folder} now holds another mail")
    return message


def move_email_to_classification_folder(mailbox, email, account):
    destination = folder_for_classification(email.classification, account)
    if destination is None:
        return None
    source_folder = email.folder or inbox_folder(account)
    if _verify_uid(mailbox, email, source_folder) is None:
        raise ValueError(f"Email {email.id} is no longer in {source_folder}")

    if not mailbox.folder.exists(destination):
        mailbox.folder.create(destination)
    mailbox.folder.set(source_folder, readonly=False)
    mailbox.move(email.uid, destination)
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
    if not mailbox.folder.exists(source_folder):
        raise ValueError(f"IMAP source folder does not exist: {source_folder}")
    if _verify_uid(mailbox, email, source_folder) is None:
        raise ValueError(f"Email {email.id} is no longer in {source_folder}")
    mailbox.folder.set(source_folder, readonly=False)
    mailbox.move([email.uid], inbox)
    mailbox.folder.set(inbox, readonly=False)

    # UIDs are per folder: the old one may now belong to another INBOX mail.
    if not email.message_id:
        return None
    uids = mailbox.uids(AND(header=H("Message-ID", email.message_id)))
    return uids[-1] if uids else None


def _folder_names(mailbox, account):
    """Classification folders first, the inbox last (see the copy rule in _sync_folders)."""
    # Disabled folders are still scanned so corrections and retention keep working.
    known_folders = {
        folder_path(classification, account) for classification in folder_rules(account)
    }
    return [
        *[
            folder.name
            for folder in mailbox.folder.list()
            if folder.name in known_folders
        ],
        inbox_folder(account),
    ]


def _identity(account, msg):
    message_id = _message_id(msg)
    return message_id, fingerprint(
        account.id, msg.from_, msg.subject, msg.date, message_id, msg.text, msg.html
    )


def _find_email(db, account, message_id, mail_fingerprint, subject, date):
    """Stored mail for a server message: by Message-ID, else by fingerprint."""
    emails = db.query(Email).filter(Email.account == account.id)
    if message_id:
        for email_obj in emails.filter(Email.message_id == message_id).order_by(Email.id):
            # Some senders reuse Message-IDs; a different subject *and* date is another mail.
            if (
                email_obj.fingerprint in (None, mail_fingerprint)
                or email_obj.subject == subject
                or normalized_date(email_obj.date) == normalized_date(date)
            ):
                return email_obj
    return emails.filter(Email.fingerprint == mail_fingerprint).first()


def _move_to_trash(mailbox, email, account):
    if not email.uid or not email.folder:
        raise ValueError(f"Email {email.id} has no folder or IMAP UID")
    trash = trash_folder(account)
    if not mailbox.folder.exists(trash):
        mailbox.folder.create(trash)
    mailbox.folder.set(email.folder, readonly=False)
    mailbox.move([email.uid], trash)
    mailbox.folder.set(inbox_folder(account), readonly=False)
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

                is_seen = _is_seen(current_message)
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
    email_obj = _find_email(db, account, message_id, mail_fingerprint, msg.subject, msg.date)
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
    email_obj.message_id = message_id
    email_obj.fingerprint = mail_fingerprint
    in_reply_to = _message_ids(_header(msg, "In-Reply-To"))
    email_obj.in_reply_to = in_reply_to[0] if in_reply_to else None
    email_obj.reference_ids = " ".join(_message_ids(_header(msg, "References"))) or None

    sync_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
    email_obj.last_seen_at = sync_timestamp
    if email_obj.removal_reason == "external":
        email_obj.removed_at = None
        email_obj.removal_reason = None
    previous_folder = email_obj.folder
    previous_classification = email_obj.classification
    email_obj.uid = msg.uid
    email_obj.folder = folder_name
    email_obj.subject = msg.subject
    email_obj.sender = msg.from_
    email_obj.date = msg.date
    email_obj.text = msg.text
    email_obj.html = msg.html
    if _is_seen(msg) and email_obj.read_at is None:
        email_obj.read_at = datetime.now(timezone.utc).replace(tzinfo=None)
        logger.info("Email %s first observed as read", email_obj.id)

    folder_classification = classification_for_folder(folder_name, account)
    previous_folder_classification = classification_for_folder(previous_folder or "", account)
    if (
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


def _sync_sent(db, mailbox, account, allow_new, seen_ids):
    """Store recent sent mails so replies can update the status of what they answer.

    Scanned after the inbox, so a mail sent to oneself keeps its inbox copy. Not
    part of the stale check: older sent mails are simply not read.
    """
    sent = _sent_folder(mailbox, account)
    if not sent or not mailbox.folder.exists(sent):
        return {"folder": None, "new": 0}
    mailbox.folder.set(sent, readonly=True)
    since = date.today() - timedelta(days=SENT_SYNC_DAYS)
    new_sent = []
    for msg in mailbox.fetch(AND(date_gte=since), mark_seen=False):
        email_obj = _sync_message(
            db, account, msg, sent, allow_new=allow_new,
            seen_ids=seen_ids, new_emails=new_sent, outbound=True,
        )
        if email_obj is not None:
            seen_ids.add(email_obj.id)
    return {"folder": sent, "new": len(new_sent)}


def _sync_folders(account, allow_new):
    """Reconcile the inbox and classification folders; new mails only from the inbox.

    Copy rule: a mail present in several scanned folders keeps the first one
    seen. Classification folders are scanned before the inbox, so a copy left
    in the inbox neither moves the mail back nor counts as a correction.
    """
    with open_mailbox(account) as mailbox:
        db = SessionLocal()
        try:
            sync_started = datetime.now(timezone.utc).replace(tzinfo=None)
            folders = _folder_names(mailbox, account)
            inbox = inbox_folder(account)
            active_folders = set(folders)
            checked = 0
            changed = 0
            new_emails = []
            seen_ids = set()
            for folder_name in folders:
                mailbox.folder.set(folder_name, readonly=True)
                for msg in mailbox.fetch(mark_seen=False):
                    checked += 1
                    new_count = len(new_emails)
                    email_obj = _sync_message(
                        db,
                        account,
                        msg,
                        folder_name,
                        allow_new=allow_new and folder_name == inbox,
                        seen_ids=seen_ids,
                        new_emails=new_emails,
                    )
                    if email_obj is None:
                        continue
                    seen_ids.add(email_obj.id)
                    if len(new_emails) == new_count:
                        changed += 1
            sent_result = _sync_sent(db, mailbox, account, allow_new, seen_ids)
            stale_emails = db.query(Email).filter(
                Email.account == account.id,
                Email.folder.in_(active_folders),
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
    """Reconcile known messages across the inbox and classification folders."""
    result = _sync_folders(account, allow_new=False)
    del result["new"]
    return result


def fetch_inbox(account):
    """Sync all folders and store new inbox messages; returns the new mails."""
    return _sync_folders(account, allow_new=True)["new"]
