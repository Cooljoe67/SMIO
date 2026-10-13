import logging
from datetime import datetime, timezone

from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from src.ai import classifier
from src.ai import extraction
from src.db.models import Email
from src.api.utils.folder_rules import folder_rules, inbox_folder
from src.mail.commands import (
    move_email_to_classification_folder,
    move_email_to_inbox,
)
from src.mail import open_mailbox
from src.mail.events import record


logger = logging.getLogger(__name__)


def process_email_by_id(email_id: int, db: Session, account, batch_id: int = None, mailbox=None):
    """
    Full processing pipeline:
    - Load email
    - Classification
    - Detail extraction (if the folder asks for it)
    - DB update
    - Return result dict
    """

    # 1. Load email
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        logger.warning("Email %s not found", email_id)
        return None

    try:
        # SMIO's own mails never get here: the sync stores them as processed.
        # 2. Classification
        classification, confidence = classifier.predict_email(email, account.folder_list)
        logger.info(
            "Email %s classified as %s (confidence %.3f)",
            email_id,
            classification,
            confidence,
        )
        email.classification = classification
        email.classification_source = "model"
        email.folder_list = account.folder_list
        email.confidence = str(confidence)

        result = {
            "email_id": email_id,
            "classification": classification,
            "confidence": confidence,
            "extracted": False,
        }

        # 3. Read delivery and shipment details for folders that ask for them.
        rule = folder_rules(account).get(classification)
        if rule is not None and rule.extract:
            try:
                details = extraction.extract(email)
            except extraction.ExtractionError as error:
                logger.warning("Email %s: no details extracted: %s", email_id, error)
            else:
                result["details"] = extraction.apply(email, details)
                result["extracted"] = True

        source_folder = email.folder
        destination = move_email_to_classification_folder(mailbox, email, account)
        if destination:
            record(
                db, account, email, "moved",
                from_folder=source_folder, to_folder=destination, source="system",
            )
            email.folder = destination
            result["folder"] = destination
            logger.info("Email %s moved to %s", email_id, destination)

        email.processed = int(email.processed or 0) + 1
        email.processing_batch = batch_id
        email.processed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        logger.info(
            "Email %s processed successfully (count=%s, batch=%s)",
            email_id,
            email.processed,
            batch_id,
        )
        return result
    except Exception:
        db.rollback()
        logger.exception("Email %s processing failed", email_id)
        raise


def process_unprocessed_emails(db: Session, account):
    # Batch ids are global, so undo can address a batch without its account.
    last_batch_id = db.query(func.max(Email.processing_batch)).scalar() or 0
    batch_id = int(last_batch_id) + 1
    email_ids = [
        email_id
        for (email_id,) in db.query(Email.id)
        .filter(
            Email.account == account.id,
            Email.is_training_data.is_(False),
            # Mails the user filed or deleted before processing are not sorted again.
            Email.folder == inbox_folder(account),
            Email.removed_at.is_(None),
            or_(Email.processed <= 0, Email.processed.is_(None)),
        )
        .all()
    ]
    if email_ids and not classifier.model_available(account.folder_list):
        # Leave them unprocessed so they are classified once the model exists.
        logger.warning(
            "Account %s: no model for folder list %s yet; %d mails stay in the inbox",
            account.id,
            account.folder_list,
            len(email_ids),
        )
        return batch_id, [], [{"error": f"no_model_for_folder_list:{account.folder_list}"}]
    logger.info(
        "Account %s: starting batch %s for %d unprocessed emails",
        account.id,
        batch_id,
        len(email_ids),
    )

    results = []
    errors = []
    if not email_ids:
        return batch_id, results, errors
    with open_mailbox(account) as mailbox:
        for position, email_id in enumerate(email_ids, start=1):
            logger.info("Processing email %d/%d (id=%s)", position, len(email_ids), email_id)
            try:
                results.append(
                    process_email_by_id(
                        email_id,
                        db,
                        account,
                        batch_id=batch_id,
                        mailbox=mailbox,
                    )
                )
            except Exception as error:
                errors.append({"email_id": email_id, "error": str(error)})
                logger.error("Continuing after failure for email %s", email_id)

    logger.info(
        "Batch %s finished: %d processed, %d failed",
        batch_id,
        len(results),
        len(errors),
    )
    return batch_id, results, errors


MAX_UNDO_BATCHES = 20


def _latest_batch_ids(db: Session, count: int, account):
    """Most recent batches that moved at least one non-SMIO mail, newest first."""
    rows = (
        db.query(Email.processing_batch)
        .filter(
            Email.account == account.id,
            Email.processing_batch > 0,
            Email.removed_at.is_(None),
            or_(
                Email.classification_source.is_(None),
                Email.classification_source != "system",
            ),
        )
        .distinct()
        .order_by(Email.processing_batch.desc())
        .limit(count)
        .all()
    )
    return [batch_id for (batch_id,) in rows]


def undo_processing_batches(db: Session, account, count: int = 1):
    """Restore the mails of the account's last ``count`` processing batches to its inbox.

    Restored mails keep their processed count, so the next run does not file them
    away again. SMIO's own mails and mails already removed are left untouched.
    """
    if not 1 <= count <= MAX_UNDO_BATCHES:
        raise ValueError(f"count must be between 1 and {MAX_UNDO_BATCHES}")

    batch_ids = _latest_batch_ids(db, count, account)
    if not batch_ids:
        return [], [], []

    emails = (
        db.query(Email)
        .filter(
            Email.account == account.id,
            Email.processing_batch.in_(batch_ids),
            Email.removed_at.is_(None),
            or_(
                Email.classification_source.is_(None),
                Email.classification_source != "system",
            ),
        )
        .all()
    )
    restored = []
    errors = []

    with open_mailbox(account) as mailbox:
        for email in emails:
            try:
                source_folder = email.folder
                new_uid = move_email_to_inbox(mailbox, email, account)
                if source_folder != inbox_folder(account):
                    record(
                        db, account, email, "moved",
                        from_folder=source_folder, to_folder=inbox_folder(account),
                        source="system",
                    )
                # The old UID belonged to the source folder; never keep it.
                email.uid = new_uid
                if not new_uid:
                    logger.warning(
                        "Email %s: INBOX UID unknown after undo; next sync will resolve it",
                        email.id,
                    )
                email.folder = inbox_folder(account)
                email.processing_batch = None
                email.classification_source = "undo"
                db.commit()
                restored.append(email.id)
                logger.info("Email %s restored to INBOX (uid=%s)", email.id, email.uid)
            except Exception as error:
                db.rollback()
                errors.append({"email_id": email.id, "error": str(error)})
                logger.exception("Could not undo processing for email %s", email.id)

    logger.info(
        "Undo of batches %s finished: %d restored, %d failed",
        batch_ids,
        len(restored),
        len(errors),
    )
    return batch_ids, restored, errors
