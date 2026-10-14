"""Cloud Scheduler routines for the single-instance SMIO service."""

import json
import logging
import os
import uuid
from threading import Lock

from src.ai.daily_summary import (
    gather_daily_summary,
    mark_summary_sent,
    stored_daily_summary,
)
from src.ai.retrain import retrain_if_due
from src.api.utils.command_processor import process_instruction_mails
from src.mail import PROVIDERS
from src.mail.events import dispatch as dispatch_events
from src.mail.gmail import address_key
from src.mail.sync import fetch_inbox
from src.api.utils.inbox_processor import process_unprocessed_emails
from src.api.utils.mailer import append_summary_to_inbox
from src.api.utils.reply_tracking import update_statuses_from_replies
from src.config import get_config, reload_config, require_stored_account_ids
from src.db.database import SessionLocal, persist_database, refresh_database
from src.db.models import MailSyncState, SummarySnapshot
from src.storage import gcs

logger = logging.getLogger(__name__)
_workflow_lock = Lock()

# Cloud Run runs one instance, but during a deploy the old and the new revision
# overlap, each with its own copy of the database. A lease in GCS lets only one
# of them run a workflow; the other skips (a Gmail push gets 503 and comes
# again). It expires after the longest request Cloud Run allows (--timeout), so
# a crashed holder blocks the workflows for at most that long.
LEASE_OBJECT = os.getenv("SMIO_WORKFLOW_LEASE_OBJECT", "locks/workflow.json")
LEASE_SECONDS = int(os.getenv("SMIO_WORKFLOW_LEASE_SECONDS", "3600"))
_HOLDER = f"{os.getenv('K_REVISION', 'local')}/{uuid.uuid4().hex[:8]}"


def publish_startup_changes():
    """Publish what the startup migrations changed, under the lease: during a
    deploy the old revision may still be running a workflow on the database."""
    if not gcs.syncs():
        return False
    lease = gcs.acquire_lease(LEASE_OBJECT, _HOLDER, LEASE_SECONDS)
    if lease is None:
        # The first workflow publishes them, after loading the other
        # instance's database and migrating it again.
        logger.warning("Startup changes stay local: another instance runs a workflow")
        return False
    try:
        return persist_database()
    finally:
        gcs.release_lease(LEASE_OBJECT, lease)


def _run_exclusively(name, routine):
    if not _workflow_lock.acquire(blocking=False):
        logger.warning("%s skipped because another workflow is running", name)
        return {"skipped": True, "reason": "workflow_in_progress"}

    lease = None
    try:
        if gcs.syncs():
            lease = gcs.acquire_lease(LEASE_OBJECT, _HOLDER, LEASE_SECONDS)
            if lease is None:
                logger.warning("%s skipped because another instance runs a workflow", name)
                return {"skipped": True, "reason": "workflow_in_progress"}
            # The other instance may have published since this copy was loaded.
            refresh_database()
        try:
            # Pick up folder/account changes made in GCS without a redeploy,
            # but never one that drops the account id of stored mails.
            reload_config(check=require_stored_account_ids)
            result = routine()
        except Exception:
            # Keep what was committed before the failure, but report the original error.
            try:
                persist_database()
            except Exception:
                logger.exception("Could not publish the database after %s failed", name)
            raise
        persist_database()
        return result
    finally:
        if lease is not None:
            try:
                gcs.release_lease(LEASE_OBJECT, lease)
            except Exception:
                # It expires on its own.
                logger.exception("Could not release the workflow lease")
        _workflow_lock.release()


def _process_account(account, is_primary):
    # Only the count: the result is returned as JSON and the mails are ORM objects.
    fetched = len(fetch_inbox(account))
    instruction_result = process_instruction_mails(account, is_primary=is_primary)

    db = SessionLocal()
    try:
        batch_id, results, errors = process_unprocessed_emails(db, account)
        status_result = update_statuses_from_replies(db, account)
        events_result = dispatch_events(db, account)
    finally:
        db.close()

    return {
        "fetched": fetched,
        "instructions": instruction_result,
        "batch_id": batch_id,
        "processed": len(results),
        "extracted": sum(result["extracted"] for result in results),
        "moved": sum("folder" in result for result in results),
        "failed": len(errors),
        "errors": errors,
        "reply_status": status_result,
        "events": events_result,
    }


def run_five_minute_workflow():
    """Per account: fetch, handle instructions, then classify new messages."""
    def routine():
        results = {}
        for position, account in enumerate(get_config().accounts):
            if account.provider not in PROVIDERS:
                results[account.id] = {"skipped": True, "reason": f"provider_{account.provider}_not_supported"}
                continue
            try:
                results[account.id] = _process_account(account, is_primary=position == 0)
            except Exception as error:
                # One broken mailbox must not stop the other accounts.
                logger.exception("Five-minute workflow failed for account %s", account.id)
                results[account.id] = {"error": str(error)}
        return {"accounts": results}

    return _run_exclusively("five-minute workflow", routine)


def _history_read(account_id, history_id):
    """Whether the account's sync already read its changes up to history_id."""
    db = SessionLocal()
    try:
        cursors = [
            state.cursor
            for state in db.query(MailSyncState).filter(MailSyncState.account == account_id)
        ]
    finally:
        db.close()
    return bool(cursors) and all(
        (cursor or "").isdigit() and int(cursor) >= int(history_id) for cursor in cursors
    )


def run_gmail_push(address, history_id):
    """Process the Gmail account a push notification is for, like the five-minute
    workflow does, unless its sync is already past the notification (e.g. the
    echo of SMIO's own change)."""
    wanted = address_key(address)
    account = next(
        (a for a in get_config().accounts if a.provider == "gmail" and address_key(a.email) == wanted),
        None,
    )
    if account is None:
        return {"ignored": "unknown_mailbox"}
    if str(history_id).isdigit() and _history_read(account.id, history_id):
        return {"skipped": True, "reason": "up_to_date"}

    def routine():
        for position, current in enumerate(get_config().accounts):
            if current.id != account.id:
                continue
            try:
                outcome = _process_account(current, is_primary=position == 0)
            except Exception as error:
                # Acknowledged anyway: the five-minute workflow retries the account.
                logger.exception("Gmail push failed for account %s", current.id)
                outcome = {"error": str(error)}
            return {"accounts": {current.id: outcome}}
        return {"ignored": "unknown_mailbox"}

    return _run_exclusively("Gmail push", routine)


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
        sent = append_summary_to_inbox(
            summary["message"],
            html_body=summary.get("html_message"),
            kind="summary",
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
                gather_daily_summary(db, summary_date, force=True)
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
    """Gather, append the summary to the inbox, then retrain."""
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