"""Scans INBOX for self-sent summary replies containing an instruction.

Accepted subjects are summary replies (for example, "AW: SMIO Daily Summary")
or the fallback "SMIO Command". The command is the first non-empty body line,
optionally prefixed with "SMIO:".

Each matching mail is moved to trash before execution so a failed or interrupted
command is not retried. Caught failures generate a separate inbox notice.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from imap_tools import MailBox

from src.ai import classifier
from src.ai.daily_summary import build_daily_summary
from src.ai.retrain import restore_previous_model, retrain_if_due
from src.db.database import SessionLocal
from src.storage import gcs

from .folder_rules import TRASH_FOLDER
from .inbox_processor import undo_last_processing
from .mailer import append_summary_to_inbox
from .settings import email_settings

logger = logging.getLogger(__name__)
PENDING_COMMANDS_PREFIX = os.getenv(
    "SMIO_PENDING_COMMANDS_PREFIX",
    "state/pending_commands",
).rstrip("/")
LOCAL_PENDING_COMMANDS_DIR = Path(
    os.getenv("SMIO_PENDING_COMMANDS_DIR", "./tmp/pending_commands")
)
COMMAND_TIMEOUT_MINUTES = max(1, int(os.getenv("SMIO_COMMAND_TIMEOUT_MINUTES", "15")))

# Only mails from the mailbox owner are honored; the From header is otherwise unauthenticated.
_SUBJECT_REPLY_PREFIX = re.compile(r"^(?:aw|re|fw|fwd)\s*:\s*", re.IGNORECASE)
_COMMAND_SUBJECTS = {
    "smio daily summary",
    "smio summary (on demand)",
    "smio command",
}
_COMMAND_PATTERN = re.compile(
    r"^\s*(?:smio\s*:\s*)?(undo|retrain|reload\s+model|restore\s+model|summary|logs\s+\d+)(?=$|\s)",
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

    if command == "restore model":
        return {"command": command, **restore_previous_model()}

    if command.startswith("logs "):
        try:
            line_count = int(command.split(maxsplit=1)[1])
        except (IndexError, ValueError):
            return {"command": command, "error": "LOGS requires an integer from 1 to 2000."}
        if not 1 <= line_count <= 2000:
            return {"command": command, "error": "LOGS requires an integer from 1 to 2000."}

        try:
            log_text = _recent_cloud_run_logs(line_count)
        except Exception as error:
            logger.exception("Could not retrieve the latest Cloud Run logs")
            return {"command": command, "error": f"Could not retrieve logs: {error}"}

        actual_line_count = len(log_text.splitlines())
        sent = append_summary_to_inbox(
            f"Attached are {actual_line_count} Cloud Run API log lines "
            f"(requested up to {line_count}).",
            subject=f"SMIO API logs ({actual_line_count} lines)",
            attachments=[("smio-api-logs.txt", log_text.encode("utf-8"))],
        )
        return {"command": command, "sent": sent, "log_lines": actual_line_count}

    if command == "summary":
        summary = build_daily_summary(db, persist=False)
        sent = append_summary_to_inbox(
            summary["message"],
            html_body=summary.get("html_message"),
            subject="SMIO Summary (on demand)",
        )
        return {"command": command, "sent": sent}

    return {"command": command, "error": "unknown command"}


def _recent_cloud_run_logs(line_count):
    from google.cloud import logging_v2

    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID")
    )
    service_name = os.getenv("K_SERVICE") or os.getenv("SMIO_SERVICE_NAME", "smio")
    client = logging_v2.Client(project=project)
    entries = client.list_entries(
        filter_=(
            'resource.type="cloud_run_revision" '
            f'AND resource.labels.service_name="{service_name}"'
        ),
        order_by="timestamp desc",
        max_results=line_count,
    )

    lines = []
    for entry in entries:
        payload = entry.payload
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, ensure_ascii=False, default=str)
        payload = " ".join(str(payload or "").split())
        timestamp = entry.timestamp.isoformat() if entry.timestamp else "unknown-time"
        severity = entry.severity or "DEFAULT"
        lines.append(f"{timestamp} {severity} {payload}".rstrip())
    return "\n".join(lines) or "No Cloud Run API log entries were found."


def _delete_message(mailbox, uid):
    if not mailbox.folder.exists(TRASH_FOLDER):
        mailbox.folder.create(TRASH_FOLDER)
    mailbox.move([uid], TRASH_FOLDER)


def _command_failed(result):
    return bool(
        result.get("error")
        or result.get("errors")
        or result.get("retrained") is False
        or result.get("restored") is False
        or result.get("sent") is False
    )


def _attempt_id(uid):
    return hashlib.sha256(str(uid).encode("utf-8")).hexdigest()


def _attempt_object(attempt_id):
    return f"{PENDING_COMMANDS_PREFIX}/{attempt_id}/attempt.json"


def _save_command_attempt(attempt_id, attempt):
    contents = json.dumps(attempt)
    if not gcs.enabled():
        LOCAL_PENDING_COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
        (LOCAL_PENDING_COMMANDS_DIR / f"{attempt_id}.json").write_text(
            contents,
            encoding="utf-8",
        )
        return

    with tempfile.TemporaryDirectory(prefix="smio-command-state-") as temporary_dir:
        path = Path(temporary_dir) / "attempt.json"
        path.write_text(contents, encoding="utf-8")
        if not gcs.upload_file(path, _attempt_object(attempt_id)):
            raise RuntimeError("Could not persist command attempt to GCS")


def _load_command_attempts():
    if not gcs.enabled():
        LOCAL_PENDING_COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
        for path in LOCAL_PENDING_COMMANDS_DIR.glob("*.json"):
            yield path.stem, json.loads(path.read_text(encoding="utf-8"))
        return

    for attempt_id in gcs.list_directory_prefixes(PENDING_COMMANDS_PREFIX):
        with tempfile.TemporaryDirectory(prefix="smio-command-state-") as temporary_dir:
            path = Path(temporary_dir) / "attempt.json"
            if gcs.download_file(_attempt_object(attempt_id), path):
                yield attempt_id, json.loads(path.read_text(encoding="utf-8"))


def _delete_command_attempt(attempt_id):
    if gcs.enabled():
        gcs.delete_directory(f"{PENDING_COMMANDS_PREFIX}/{attempt_id}")
    else:
        (LOCAL_PENDING_COMMANDS_DIR / f"{attempt_id}.json").unlink(missing_ok=True)


def _notify_command_failure(command, reason):
    body = (
        f"The SMIO command '{command.upper()}' was not successful.\n\n"
        f"{reason}\n"
        "Check the Cloud Run logs for details."
    )
    return append_summary_to_inbox(body, subject="SMIO command unsuccessful")


def _record_command_failure(attempt_id, attempt, reason):
    attempt["status"] = "failed"
    attempt["failure_reason"] = reason
    attempt["failed_at"] = datetime.now(timezone.utc).isoformat()
    try:
        _save_command_attempt(attempt_id, attempt)
    except Exception:
        logger.exception("Could not persist failure state for command %s", attempt["command"])

    if _notify_command_failure(attempt["command"], reason):
        _delete_command_attempt(attempt_id)
    else:
        logger.error("Could not append failure notice for command %s", attempt["command"])


def _check_timed_out_commands():
    now = datetime.now(timezone.utc)
    timeout = timedelta(minutes=COMMAND_TIMEOUT_MINUTES)
    for attempt_id, attempt in _load_command_attempts():
        if attempt.get("status") == "completed":
            _delete_command_attempt(attempt_id)
            continue

        if attempt.get("status") == "failed":
            reason = attempt.get("failure_reason", "The command failed.")
        else:
            try:
                started_at = datetime.fromisoformat(attempt["started_at"])
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring malformed command attempt marker %s", attempt_id)
                continue
            if now - started_at < timeout:
                continue
            reason = (
                f"No completion was recorded within {COMMAND_TIMEOUT_MINUTES} minutes. "
                "The service may have restarted or terminated during execution."
            )
            attempt["status"] = "failed"
            attempt["failure_reason"] = reason
            try:
                _save_command_attempt(attempt_id, attempt)
            except Exception:
                logger.exception("Could not persist timeout state for command %s", attempt.get("command"))

        if _notify_command_failure(attempt["command"], reason):
            _delete_command_attempt(attempt_id)
        else:
            logger.error("Could not append failure notice for command %s", attempt.get("command"))


def process_instruction_mails():
    """Find commands, trash them before execution, and report caught failures."""
    executed = []
    errors = []
    try:
        _check_timed_out_commands()
    except Exception:
        logger.exception("Could not check pending command attempts")

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
                attempt_id = _attempt_id(uid)
                attempt = {
                    "uid": str(uid),
                    "command": command,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "status": "running",
                }
                try:
                    _save_command_attempt(attempt_id, attempt)
                    _delete_message(mailbox, uid)
                except Exception as error:
                    errors.append({"command": command, "error": str(error)})
                    logger.exception("Could not prepare instruction command %s", command)
                    _record_command_failure(
                        attempt_id,
                        attempt,
                        "The command could not be safely started or removed from the inbox.",
                    )
                    continue

                try:
                    result = _run_command(db, command)
                    executed.append(result)
                    if _command_failed(result):
                        reason = "The command returned an unsuccessful result."
                        errors.append({
                            "command": command,
                            "error": reason,
                        })
                        logger.error("Instruction command %s was not successful", command)
                        _record_command_failure(attempt_id, attempt, reason)
                    else:
                        attempt["status"] = "completed"
                        attempt["completed_at"] = datetime.now(timezone.utc).isoformat()
                        _save_command_attempt(attempt_id, attempt)
                        _delete_command_attempt(attempt_id)
                except Exception as error:
                    errors.append({"command": command, "error": str(error)})
                    logger.exception("Failed to execute instruction command %s", command)
                    _record_command_failure(
                        attempt_id,
                        attempt,
                        "The command raised an error during execution.",
                    )
        finally:
            db.close()

    return {"executed": executed, "errors": errors}
