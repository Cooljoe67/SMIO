from imap_tools import MailBox
from .settings import email_settings

def fetch_inbox():
    with MailBox(email_settings.host).login(email_settings.user, email_settings.password) as mailbox:
        messages = mailbox.fetch()
        emails = []

        for msg in messages:
            emails.append({
                "uid": msg.uid,
                "subject": msg.subject,
                "from": msg.from_,
                "date": msg.date_str,
                "text": msg.text,
                "html": msg.html,
            })

        return emails
