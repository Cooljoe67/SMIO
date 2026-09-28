from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Email(Base):
    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, index=True)
    message_id = Column(String, index=True)
    folder = Column(String, default="INBOX")
    removed_at = Column(DateTime, nullable=True, index=True)
    removal_reason = Column(String, nullable=True)
    last_seen_at = Column(DateTime, nullable=True, index=True)

    subject = Column(String)
    sender = Column(String)
    date = Column(DateTime, default=_utcnow)
    read_at = Column(DateTime, nullable=True, index=True)

    text = Column(Text)
    html = Column(Text)


    # AI fields
    classification = Column(String, nullable=True)
    classification_source = Column(String, nullable=True)
    confidence = Column(String, nullable=True)
    embedding = Column(Text, nullable=True)
    order_id = Column(String, nullable=True)
    tracking_id = Column(String, nullable=True)
    entity_date = Column(String, nullable=True)
    entity_name = Column(String, nullable=True)
    entity_company = Column(String, nullable=True)
    delivery_status = Column(String, nullable=True)
    item_name = Column(String, nullable=True)
    delivery_date = Column(String, nullable=True)
    delivery_company = Column(String, nullable=True)
    ner_entities = Column(Text, nullable=True)
    extraction_sources = Column(Text, nullable=True)

    # Number of times this email has been processed.
    processed = Column(Integer, default=0, nullable=False)
    processing_batch = Column(Integer, nullable=True, index=True)
    processed_at = Column(DateTime, nullable=True, index=True)

    true_label = Column(String, nullable=True)

    # Set to the retrain run id once this correction has been used for fine-tuning.
    retrain_batch = Column(Integer, nullable=True, index=True)

    # Permanently reserved for evaluation; never used for training.
    is_eval_holdout = Column(Boolean, default=False, nullable=False)
    is_training_data = Column(Boolean, default=False, nullable=False)


class SummarySnapshot(Base):
    __tablename__ = "summary_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    summary_date = Column(String, nullable=False, unique=True, index=True)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    # When this snapshot's data was collected, not when the row was last touched.
    gathered_at = Column(DateTime, default=_utcnow, nullable=False)
    data_json = Column(Text, nullable=False)
    message = Column(Text, nullable=False)
    html_message = Column(Text, nullable=False)
    sent_at = Column(DateTime, nullable=True)
    send_status = Column(String, nullable=True)
    retrain_due = Column(Boolean, default=False, nullable=False)
    retrain_run_id = Column(Integer, nullable=True)
    retrain_result_json = Column(Text, nullable=True)

    # Flattened metrics, queryable directly without parsing data_json or joining emails.
    total_entries = Column(Integer, nullable=True)
    processed_entries = Column(Integer, nullable=True)
    correction_count = Column(Integer, nullable=True)
    pending_corrections = Column(Integer, nullable=True)
    correction_threshold = Column(Integer, nullable=True)
    new_count_total = Column(Integer, nullable=True)
    unread_count_total = Column(Integer, nullable=True)
    removed_count_total = Column(Integer, nullable=True)
    retrain_last_run_at = Column(String, nullable=True)
    retrain_last_promoted_at = Column(String, nullable=True)
    retrain_accuracy = Column(String, nullable=True)
    retrain_recall = Column(String, nullable=True)
    retrain_f1 = Column(String, nullable=True)

