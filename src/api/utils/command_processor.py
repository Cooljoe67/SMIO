"""Scans INBOX for self-sent instruction mails and executes the matching action.

Recognized subjects (case-insensitive, optional "SMIO:" prefix):
  UNDO           -> revert the last processing batch
  RETRAIN        -> force a retrain attempt regardless of pending-correction count
  RELOAD MODEL   -> reload the deployed classifier from disk
  SUMMARY        -> send an on-demand summary without resetting the scheduled period

Each matching mail is deleted (moved to trash) once handled, which doubles as
the confirmation that it was received and processed.
"""

import logging
import re

from imap_tools import MailBox

from src.ai import classifier
from src.ai.daily_summary import build_daily_summary
from src.ai.retrain import retrain_if_due
from src.db.database import SessionLocal

from .folder_rules import TRASH_FOLDER
from .inbox_processor import undo_last_processing
from .mailer import send_summary_email
from .settings import email_settings

logger = logging.getLogger(__name__)

# Only mails from the mailbox owner are honored; the From header is otherwise unauthenticated.
_COMMAND_PATTERN = re.compile(
    r"^\s*(?:smio\s*[:\-]?\s*)?(undo|retrain|reload model|summary)\s*$",
    re.IGNORECASE,
)


def _normalize_command(subject):
    match = _COMMAND_PATTERN.match(subject or "")
    return match.group(1).lower() if match else None


def _run_command(db, command):
    if command == "undo":
        batch_id, restored, errors = undo_last_processing(db)
        return {"command": command, "batch_id": batch_id, "restored": len(restored), "errors": errors}

    if command == "retrain":
        result = retrain_if_due(db, min_corrections=0)
        return {"command": command, **result}

    if command == "reload model":
        classifier.reload_model()
        return {"command": command, "reloaded": True}

    if command == "summary":
        summary = build_daily_summary(db, persist=False)
        sent = send_summary_email(summary["message"], subject="SMIO Summary (on demand)")
        return {"command": command, "sent": sent}

    return {"command": command, "error": "unknown command"}


def _delete_message(mailbox, uid):
    if not mailbox.folder.exists(TRASH_FOLDER):
        mailbox.folder.create(TRASH_FOLDER)
    mailbox.move([uid], TRASH_FOLDER)


def process_instruction_mails():
    """Find, execute, and delete self-sent instruction mails in INBOX."""
    executed = []
    errors = []
    owner = (email_settings.user or "").strip().lower()

    with MailBox(email_settings.host).login(email_settings.user, email_settings.password) as mailbox:
        mailbox.folder.set("INBOX", readonly=False)

        # Collect matches first: mutating folders while a fetch() generator is open is unsafe.
        matches = []
        for msg in mailbox.fetch(mark_seen=False):
            if (msg.from_ or "").strip().lower() != owner:
                continue
            command = _normalize_command(msg.subject)
            if command is not None:
                matches.append((msg.uid, command))

        db = SessionLocal()
        try:
            for uid, command in matches:
                logger.info("Instruction mail detected: %s (uid=%s)", command, uid)
                try:
                    executed.append(_run_command(db, command))
                except Exception as error:
                    errors.append({"command": command, "error": str(error)})
                    logger.exception("Failed to execute instruction command %s", command)
                finally:
                    # Delete regardless of outcome so a failing command isn't retried forever.
                    _delete_message(mailbox, uid)
        finally:
            db.close()

    return {"executed": executed, "errors": errors}
