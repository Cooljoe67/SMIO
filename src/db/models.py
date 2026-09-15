from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Email(Base):
    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, index=True)
    message_id = Column(String, index=True)
    folder = Column(String, default="INBOX")

    subject = Column(String)
    sender = Column(String)
    date = Column(DateTime, default=datetime.utcnow)
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

    true_label = Column(String, nullable=True)

