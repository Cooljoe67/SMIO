from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Email(Base):
    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, index=True)
    folder = Column(String, default="INBOX")

    subject = Column(String)
    sender = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

    text = Column(Text)
    html = Column(Text)


    # AI fields
    classification = Column(String, nullable=True)
    confidence = Column(String, nullable=True)
    embedding = Column(Text, nullable=True)

    processed = Column(Boolean, default=False)

    true_label = Column(String, nullable=True)

