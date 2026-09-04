import imaplib
from .settings import settings

def fetch_inbox():
    host = settings.IMAP_HOST
    user = settings.EMAIL_USER
    password = settings.EMAIL_PASS

    mail = imaplib.IMAP4_SSL(host)
    mail.login(user, password)
    mail.select("INBOX")

    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()

    results = []

    for eid in email_ids[-10:]:  # fetch last 10 emails
        status, msg_data = mail.fetch(eid, "(RFC822)")
        raw_email = msg_data[0][1].decode("utf-8", errors="ignore")
        results.append({"id": eid.decode(), "raw": raw_email})

    mail.close()
    mail.logout()

    return results
