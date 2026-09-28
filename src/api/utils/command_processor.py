"""Scans INBOX for self-sent summary replies containing an instruction.

Accepted subjects are summary replies (for example, "AW: SMIO Daily Summary")
or the fallback "SMIO Command". The command is the first non-empty body line,
optionally prefixed with "SMIO:".

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
_SUBJECT_REPLY_PREFIX = re.compile(r"^(?:aw|re|fw|fwd)\s*:\s*", re.IGNORECASE)
_COMMAND_SUBJECTS = {
    "smio daily summary",
    "smio summary (on demand)",
    "smio command",
}
_COMMAND_PATTERN = re.compile(
    r"^\s*(?:smio\s*:\s*)?(undo|retrain|reload\s+model|summary)(?=$|\s)",
    re.IGNORECASE,
)


def _normalize_command(subject, body):
    normalized_subject = (subject or "").strip()
    while True:
        without_prefix = _SUBJECT_REPLY_PREFIX.sub("", normalized_subject, count=1)
        if without_prefix == normalized_subject:
            break
        normalized_subject = without_prefix.strip()
    if normalized_subject.casefold() not in _COMMAND_SUBJECTS:
        return None

    first_line = next(
        (line.strip() for line in (body or "").splitlines() if line.strip()),
        "",
    )
    match = _COMMAND_PATTERN.match(first_line)
    return re.sub(r"\s+", " ", match.group(1)).casefold() if match else None


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
            command = _normalize_command(msg.subject, msg.text)
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
