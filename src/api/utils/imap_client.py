import logging
from datetime import datetime, timedelta

from imap_tools import MailBox
from src.db.database import SessionLocal
from src.db.models import Email

from .folder_rules import (
    CLASSIFICATION_FOLDERS,
    FOLDER_RULES,
    TRASH_FOLDER,
    classification_for_folder,
    folder_for_classification,
)
from .settings import email_settings

logger = logging.getLogger(__name__)

def _message_id(message):
    headers = getattr(message, "headers", {}) or {}
    value = headers.get("Message-ID") or headers.get("Message-Id") or ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip()


def _is_seen(message):
    flags = getattr(message, "flags", ()) or ()
    return any(
        (flag.decode(errors="ignore") if isinstance(flag, bytes) else str(flag)).lower()
        == "\\seen"
        for flag in flags
    )


def _classification_from_folder(folder):
    return classification_for_folder(folder)


def move_email_to_classification_folder(mailbox, email):
    if email.classification not in CLASSIFICATION_FOLDERS:
        return None
    if not email.uid:
        raise ValueError(f"Email {email.id} has no IMAP UID")

    destination = folder_for_classification(email.classification)
    if not mailbox.folder.exists(destination):
        mailbox.folder.create(destination)
    mailbox.move(email.uid, destination)
    return destination


def move_email_to_inbox(mailbox, email):
    if not email.uid:
        raise ValueError(f"Email {email.id} has no IMAP UID")
    if not email.folder or email.folder == "INBOX":
        return "INBOX"

    source_folder = email.folder
    if not mailbox.folder.exists(source_folder):
        raise ValueError(f"IMAP source folder does not exist: {source_folder}")
    mailbox.folder.set(source_folder, readonly=False)
    mailbox.move([email.uid], "INBOX")
    mailbox.folder.set("INBOX", readonly=False)
    return "INBOX"


def _folder_names(mailbox):
    known_folders = {
        folder_for_classification(classification)
        for classification in CLASSIFICATION_FOLDERS
    }
    return [
        "INBOX",
        *[
            folder.name
            for folder in mailbox.folder.list("INBOX")
            if folder.name in known_folders
        ],
    ]


def _find_email(db, message_id, uid, folder_name):
    email_obj = None
    if message_id:
        email_obj = db.query(Email).filter(Email.message_id == message_id).first()
    if email_obj is None:
        email_obj = db.query(Email).filter(
            Email.uid == uid,
            Email.folder == folder_name,
        ).first()
    return email_obj


def _move_to_trash(mailbox, email):
    if not email.uid or not email.folder:
        raise ValueError(f"Email {email.id} has no folder or IMAP UID")
    if not mailbox.folder.exists(TRASH_FOLDER):
        mailbox.folder.create(TRASH_FOLDER)
    mailbox.folder.set(email.folder, readonly=False)
    mailbox.move([email.uid], TRASH_FOLDER)
    mailbox.folder.set("INBOX", readonly=False)


def cleanup_expired_read_mails(db, mailbox):
    now = datetime.utcnow()
    moved = 0
    skipped_unread = 0
    errors = []

    for classification, rule in FOLDER_RULES.items():
        retention_days = rule.get("retention_days")
        folder_name = rule["folder"]
        if retention_days is None:
            continue

        cutoff = now - timedelta(days=retention_days)
        candidates = db.query(Email).filter(
            Email.folder == folder_name,
            Email.read_at.isnot(None),
            Email.read_at <= cutoff,
        ).all()

        for email in candidates:
            try:
                mailbox.folder.set(folder_name, readonly=True)
                current_message = next(
                    mailbox.fetch(
                        uid_list=[email.uid],
                        mark_seen=False,
                    ),
                    None,
                )
                if current_message is None or not _is_seen(current_message):
                    skipped_unread += 1
                    continue

                _move_to_trash(mailbox, email)
                email.folder = TRASH_FOLDER
                moved += 1
                logger.info(
                    "Email %s moved from %s to %s after %s retention days",
                    email.id,
                    folder_name,
                    TRASH_FOLDER,
                    retention_days,
                )
            except Exception as error:
                errors.append({"email_id": email.id, "error": str(error)})
                logger.exception("Could not move expired email %s to trash", email.id)

    return {
        "moved": moved,
        "skipped_unread": skipped_unread,
        "errors": errors,
    }


def _sync_message(db, msg, folder_name, allow_new):
    message_id = _message_id(msg)
    email_obj = _find_email(db, message_id, msg.uid, folder_name)
    new_email = email_obj is None
    if new_email and not allow_new:
        return None
    if new_email:
        email_obj = Email(message_id=message_id)

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
        email_obj.read_at = datetime.utcnow()
        logger.info("Email %s first observed as read", email_obj.id)

    folder_classification = _classification_from_folder(folder_name)
    previous_folder_classification = _classification_from_folder(previous_folder or "")
    if (
        not new_email
        and folder_classification
        and previous_classification
        and previous_classification != folder_classification
    ):
        email_obj.true_label = folder_classification
        email_obj.classification_source = "imap_folder"
    elif (
        not new_email
        and folder_name == "INBOX"
        and previous_folder_classification is not None
        and email_obj.classification_source != "undo"
    ):
        email_obj.classification = "other"
        email_obj.true_label = "other"
        email_obj.classification_source = "imap_folder"

    if folder_classification and email_obj.classification != folder_classification:
        email_obj.classification = folder_classification
        email_obj.classification_source = "imap_folder"
    if new_email:
        db.add(email_obj)
    return email_obj


def sync_existing_mails():
    """Reconcile known messages across INBOX and classification folders."""
    with MailBox(email_settings.host).login(email_settings.user, email_settings.password) as mailbox:
        db = SessionLocal()
        try:
            checked = 0
            changed = 0
            for folder_name in _folder_names(mailbox):
                mailbox.folder.set(folder_name, readonly=True)
                for msg in mailbox.fetch(mark_seen=False):
                    email_obj = _sync_message(db, msg, folder_name, allow_new=False)
                    checked += 1
                    if email_obj is not None:
                        changed += 1
            db.commit()
            cleanup = cleanup_expired_read_mails(db, mailbox)
            db.commit()
            return {"checked": checked, "changed": changed, "cleanup": cleanup}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def fetch_inbox():
    sync_existing_mails()
    with MailBox(email_settings.host).login(email_settings.user, email_settings.password) as mailbox:
        db = SessionLocal()
        try:
            new_emails = []
            mailbox.folder.set("INBOX", readonly=True)
            for msg in mailbox.fetch(mark_seen=False):
                message_id = _message_id(msg)
                was_known = _find_email(db, message_id, msg.uid, "INBOX") is not None
                email_obj = _sync_message(db, msg, "INBOX", allow_new=True)
                if email_obj is not None and not was_known:
                    new_emails.append(email_obj)
            db.commit()
            return new_emails
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
