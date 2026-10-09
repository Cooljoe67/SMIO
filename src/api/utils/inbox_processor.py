import logging
from datetime import datetime, timezone
from email.utils import parseaddr

from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from src.ai import classifier
from src.ai.email_metadata import extract_email_metadata, serialize_sources
from src.ai.ner import amazon_sender_status, serialize_entities
from src.db.models import Email
from src.api.utils.folder_rules import classification_folders, inbox_folder
from src.api.utils.imap_client import (
    move_email_to_classification_folder,
    move_email_to_inbox,
    open_mailbox,
)


logger = logging.getLogger(__name__)


def _is_smio_mail(sender, account):
    sender_address = parseaddr(sender or "")[1].strip().casefold()
    smio_address = parseaddr(account.email or "")[1].strip().casefold()
    return bool(smio_address and sender_address == smio_address)


def process_email_by_id(email_id: int, db: Session, account, batch_id: int = None, mailbox=None):
    """
    Full processing pipeline:
    - Load email
    - Classification
    - NER (if delivery)
    - DB update
    - Return result dict
    """

    # 1. Load email
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        logger.warning("Email %s not found", email_id)
        return None

    try:
        if _is_smio_mail(email.sender, account):
            email.classification = "other"
            email.classification_source = "system"
            email.folder_list = account.folder_list
            email.confidence = None
            email.processed = int(email.processed or 0) + 1
            email.processing_batch = batch_id
            email.processed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            db.commit()
            logger.info("Email %s from SMIO left in the inbox without classification", email_id)
            return {
                "email_id": email_id,
                "classification": "other",
                "confidence": None,
                "ner_executed": False,
                "classification_skipped": True,
            }

        # 2. Classification
        classification, confidence = classifier.predict_email(email, account.folder_list)
        amazon_status = amazon_sender_status(
            email.sender,
            email.subject,
            email.text,
        )
        if (
            classification != "delivery"
            and amazon_status is not None
            and "delivery" in classification_folders(account)
        ):
            logger.info(
                "Email %s: overriding classification %s to delivery from Amazon sender",
                email_id,
                classification,
            )
            classification = "delivery"
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
            "ner_executed": False,
        }

        # 3. Run NER only for delivery emails and persist all extracted fields.
        if classification == "delivery":
            fields, entities, sources = extract_email_metadata(
                email.sender,
                email.subject,
                email.text,
            )
            logger.info("Email %s: NER found %d entities", email_id, len(entities))
            if amazon_status is not None:
                logger.info(
                    "Email %s: Amazon sender %s provided status %s",
                    email_id,
                    email.sender,
                    amazon_status,
                )
            for field_name, value in fields.items():
                setattr(email, field_name, value)
            email.ner_entities = serialize_entities(entities)
            email.extraction_sources = serialize_sources(sources)
            result.update(fields)
            result["entities"] = entities
            result["sources"] = sources
            result["ner_executed"] = True

        destination = move_email_to_classification_folder(mailbox, email, account)
        if destination:
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
                new_uid = move_email_to_inbox(mailbox, email, account)
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
