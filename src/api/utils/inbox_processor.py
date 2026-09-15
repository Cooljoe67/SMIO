import logging

from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from imap_tools import MailBox
from src.ai.classifier import predict_text
from src.ai.email_metadata import extract_email_metadata, serialize_sources
from src.ai.ner import amazon_sender_status, serialize_entities
from src.db.models import Email
from src.api.utils.imap_client import (
    move_email_to_classification_folder,
    move_email_to_inbox,
)
from src.api.utils.settings import email_settings


logger = logging.getLogger(__name__)


def process_email_by_id(email_id: int, db: Session, batch_id: int = None, mailbox=None):
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
        # 2. Classification
        classification, confidence = predict_text(email.text)
        amazon_status = amazon_sender_status(
            email.sender,
            email.subject,
            email.text,
        )
        if classification != "delivery" and amazon_status is not None:
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

        destination = move_email_to_classification_folder(mailbox, email)
        if destination:
            email.folder = destination
            result["folder"] = destination
            logger.info("Email %s moved to %s", email_id, destination)

        email.processed = int(email.processed or 0) + 1
        email.processing_batch = batch_id
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


def process_unprocessed_emails(db: Session):
    last_batch_id = db.query(func.max(Email.processing_batch)).scalar() or 0
    batch_id = int(last_batch_id) + 1
    email_ids = [
        email_id
        for (email_id,) in db.query(Email.id)
        .filter(or_(Email.processed <= 0, Email.processed.is_(None)))
        .all()
    ]
    logger.info(
        "Starting batch %s for %d unprocessed emails",
        batch_id,
        len(email_ids),
    )

    results = []
    errors = []
    with MailBox(email_settings.host).login(
        email_settings.user,
        email_settings.password,
    ) as mailbox:
        for position, email_id in enumerate(email_ids, start=1):
            logger.info("Processing email %d/%d (id=%s)", position, len(email_ids), email_id)
            try:
                results.append(
                    process_email_by_id(
                        email_id,
                        db,
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


def undo_last_processing(db: Session):
    last_batch_id = (
        db.query(func.max(Email.processing_batch))
        .filter(Email.processing_batch > 0)
        .scalar()
    )
    if last_batch_id is None:
        return None, [], []

    emails = (
        db.query(Email)
        .filter(Email.processing_batch == last_batch_id)
        .all()
    )
    restored = []
    errors = []

    with MailBox(email_settings.host).login(
        email_settings.user,
        email_settings.password,
    ) as mailbox:
        for email in emails:
            try:
                move_email_to_inbox(mailbox, email)
                email.folder = "INBOX"
                email.processed = max(int(email.processed or 0) - 1, 0)
                email.processing_batch = None
                email.classification_source = "undo"
                restored.append(email.id)
                logger.info(
                    "Email %s restored to INBOX (processed count=%s)",
                    email.id,
                    email.processed,
                )
            except Exception as error:
                errors.append({"email_id": email.id, "error": str(error)})
                logger.exception("Could not undo processing for email %s", email.id)

    db.commit()
    logger.info(
        "Undo batch %s finished: %d restored, %d failed",
        last_batch_id,
        len(restored),
        len(errors),
    )
    return last_batch_id, restored, errors
