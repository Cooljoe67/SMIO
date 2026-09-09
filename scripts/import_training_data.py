# scripts/import_training_data.py

from imap_tools import MailBox
from src.db.database import SessionLocal
from src.db.models import Email
from src.api.utils.settings import email_settings

CATEGORY_FOLDERS = {
    "INBOX/delivery": "delivery",
    "INBOX/trash": "trash",
    "INBOX/social": "social",
    "INBOX/other": "other",
    "INBOX/commercial": "commercial",
    "INBOX/tech": "tech",
}

def import_training_data():
    db = SessionLocal()


    with MailBox(email_settings.host).login(
        email_settings.user,
        email_settings.password
    ) as mailbox:
        for folder, label in CATEGORY_FOLDERS.items():

            # Prüfen, ob der Ordner existiert
            try:
                mailbox.folder.set(folder)
            except Exception:
                print(f"Ordner nicht gefunden: {folder}")
                continue

            print(f"Importiere aus {folder} → Label: {label}")

            for msg in mailbox.fetch():
                email = Email(
                    subject=msg.subject,
                    sender=msg.from_,
                    date=msg.date,
                    text=msg.text,
                    html=msg.html,
                    folder=folder,
                    classification=label,
                    confidence=1.0,  # manuell gelabelt
                )
                db.add(email)

        db.commit()
        print("Training data import completed.")


if __name__ == "__main__":
    import_training_data()
