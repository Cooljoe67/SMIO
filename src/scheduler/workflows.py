"""Cloud Scheduler routines for the single-instance SMIO service."""

import logging
from threading import Lock

import json

from src.ai.daily_summary import (
    gather_daily_summary,
    mark_summary_sent,
    stored_daily_summary,
)
from src.ai.retrain import retrain_if_due
from src.api.utils.command_processor import process_instruction_mails
from src.api.utils.imap_client import fetch_inbox
from src.api.utils.inbox_processor import process_unprocessed_emails
from src.api.utils.mailer import send_summary_email
from src.db.database import SessionLocal
from src.db.models import SummarySnapshot

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


def _gather_summary(summary_date=None, force=False):
    db = SessionLocal()
    try:
        return gather_daily_summary(db, summary_date, force=force)
    finally:
        db.close()


def _send_summary(summary_date=None):
    db = SessionLocal()
    try:
        summary = stored_daily_summary(db, summary_date)
        if summary is None:
            raise ValueError("No stored summary exists for this date; gather it first")
        sent = send_summary_email(
            summary["message"],
            html_body=summary.get("html_message"),
        )
        stored = mark_summary_sent(db, summary["summary_date"], sent)
        return {"summary": stored, "summary_sent": sent}
    finally:
        db.close()


def _run_retraining(summary_date=None):
    db = SessionLocal()
    try:
        result = retrain_if_due(db)
        if summary_date is not None:
            snapshot = db.query(SummarySnapshot).filter(
                SummarySnapshot.summary_date == str(summary_date)
            ).one_or_none()
            if snapshot is not None:
                snapshot.retrain_run_id = result.get("run_id")
                snapshot.retrain_result_json = json.dumps(result, default=str)
                db.commit()
        return result
    finally:
        db.close()


def run_gather_summary(summary_date=None, force=False):
    return _run_exclusively(
        "summary gathering",
        lambda: _gather_summary(summary_date, force=force),
    )


def run_send_summary(summary_date=None):
    return _run_exclusively(
        "summary sending",
        lambda: _send_summary(summary_date),
    )


def run_retraining(summary_date=None):
    return _run_exclusively(
        "retraining",
        lambda: _run_retraining(summary_date),
    )


def run_daily_workflow(summary_date=None):
    """Gather, send, then retrain for one logical summary day."""
    def routine():
        summary = _gather_summary(summary_date)
        send_result = _send_summary(summary["summary_date"])
        retrain_result = _run_retraining(summary["summary_date"])

        return {
            "gather": summary,
            "send": send_result,
            "retrain": retrain_result,
        }

    return _run_exclusively("daily workflow", routine)