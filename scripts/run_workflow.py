"""Sequential cron/Task-Scheduler entry point: sync -> fetch -> process -> summary -> retrain.

Run as a single scheduled job on the one instance that owns the mailbox.
A lockfile prevents overlapping runs if a previous invocation is still active.
"""

import logging
from pathlib import Path

from src.ai.daily_summary import build_daily_summary
from src.ai.retrain import retrain_if_due
from src.api.utils.imap_client import fetch_inbox
from src.api.utils.inbox_processor import process_unprocessed_emails
from src.api.utils.mailer import send_summary_email
from src.db.database import SessionLocal, engine, ensure_email_columns
from src.db.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCK_FILE = Path("./tmp/workflow.lock")


def run_workflow():
    Base.metadata.create_all(bind=engine)
    ensure_email_columns()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        logger.warning("Workflow already running (lockfile present), skipping this run")
        return None

    LOCK_FILE.touch()
    try:
        logger.info("Step 1/4: fetch_inbox (includes folder sync)")
        fetch_inbox()

        db = SessionLocal()
        try:
            logger.info("Step 2/4: process_unprocessed_emails")
            batch_id, results, errors = process_unprocessed_emails(db)
            logger.info(
                "Batch %s processed %d emails (%d failed)",
                batch_id, len(results), len(errors),
            )

            logger.info("Step 3/4: daily summary email")
            summary = build_daily_summary(db)
            send_summary_email(summary["message"])

            logger.info("Step 4/4: retrain_if_due")
            retrain_result = retrain_if_due(db)
            logger.info("Retrain result: %s", retrain_result)
        finally:
            db.close()

        return retrain_result
    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    run_workflow()
