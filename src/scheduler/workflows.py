"""Cloud Scheduler routines for the single-instance SMIO service."""

import logging
from threading import Lock

from src.ai.daily_summary import build_daily_summary, persist_summary_period_end
from src.ai.retrain import retrain_if_due
from src.api.utils.command_processor import process_instruction_mails
from src.api.utils.imap_client import fetch_inbox
from src.api.utils.inbox_processor import process_unprocessed_emails
from src.api.utils.mailer import send_summary_email
from src.db.database import SessionLocal

logger = logging.getLogger(__name__)
_workflow_lock = Lock()


def _run_exclusively(name, routine):
    if not _workflow_lock.acquire(blocking=False):
        logger.warning("%s skipped because another workflow is running", name)
        return {"skipped": True, "reason": "workflow_in_progress"}

    try:
        return routine()
    finally:
        _workflow_lock.release()


def run_five_minute_workflow():
    """Fetch, handle instructions, then classify newly received messages."""
    def routine():
        fetch_result = fetch_inbox()
        instruction_result = process_instruction_mails()

        db = SessionLocal()
        try:
            batch_id, results, errors = process_unprocessed_emails(db)
        finally:
            db.close()

        return {
            "fetch": fetch_result,
            "instructions": instruction_result,
            "batch_id": batch_id,
            "processed": len(results),
            "ner_processed": sum(result["ner_executed"] for result in results),
            "moved": sum("folder" in result for result in results),
            "failed": len(errors),
            "errors": errors,
        }

    return _run_exclusively("five-minute workflow", routine)


def run_daily_workflow():
    """Send the daily digest and retrain the classifier when corrections are due."""
    def routine():
        db = SessionLocal()
        try:
            summary = build_daily_summary(db, persist=False)
            sent = send_summary_email(summary["message"])
            if sent:
                persist_summary_period_end(summary["period_end"])
            retrain_result = retrain_if_due(db)
        finally:
            db.close()

        return {
            "summary": summary,
            "summary_sent": sent,
            "retrain": retrain_result,
        }

    return _run_exclusively("daily workflow", routine)