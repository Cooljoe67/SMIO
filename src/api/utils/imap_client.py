from imap_tools import MailBox
from src.db.database import SessionLocal
from src.db.models import Email

from .settings import email_settings

def fetch_inbox():
    with MailBox(email_settings.host).login(email_settings.user, email_settings.password) as mailbox:
        messages = mailbox.fetch()
        db = SessionLocal()

        emails = []

        for msg in messages:
            email_obj = Email(
                uid=msg.uid,
                folder="INBOX",
                subject=msg.subject,
                sender=msg.from_,
                date=msg.date,
                text=msg.text,
                html=msg.html
            )

            db.add(email_obj)
            emails.append(email_obj)

        db.commit()
        return emails
