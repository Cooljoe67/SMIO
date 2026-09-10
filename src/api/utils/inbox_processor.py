from sqlalchemy.orm import Session
from ..models.email import Email
from ..services.classifier import classifier
from ..services.ner_model import ner_model
from ..services.update_db_from_ner import update_db_from_ner


def process_email_by_id(email_id: int, db: Session):
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
        return None

    # 2. Classification
    classification = classifier.predict(email.text)
    email.classification = classification

    # 3. If not delivery → no NER
    if classification != "delivery":
        db.commit()
        return {
            "email_id": email_id,
            "classification": classification,
            "ner_executed": False
        }

    # 4. Run NER
    ner_result = ner_model.extract(email.text)

    # 5. Update DB
    update_db_from_ner(db, email, ner_result)

    # 6. Return combined result
    return {
        "email_id": email_id,
        "classification": classification,
        "ner_executed": True,
        "delivery_status": email.delivery_status,
        "tracking_id": email.tracking_id,
        "order_id": email.order_id,
        "delivery_company": email.delivery_company,
        "sender_company": email.sender_company,
        "entities": email.ner_entities
    }
