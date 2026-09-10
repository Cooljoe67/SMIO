# scripts/import_testing_data.py

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

def import_testing_data():
    db = SessionLocal()

    with MailBox(email_settings.host).login(
        email_settings.user,
        email_settings.password
    ) as mailbox:
        for folder, label in CATEGORY_FOLDERS.items():

            # Check if folder exists
            try:
                mailbox.folder.set(folder)
            except Exception:
                print(f"Folder not found: {folder}")
                continue

            print(f"Importing from {folder} → true_label: {label}")

            for msg in mailbox.fetch():
                email = Email(
                    subject=msg.subject,
                    sender=msg.from_,
                    date=msg.date,
                    text=msg.text,
                    html=msg.html,
                    folder=folder,

                    # Ground truth label for testing
                    true_label=label,

                    # Model prediction will be added later
                    classification=None,
                    confidence=None,
                )
                db.add(email)

        db.commit()
        print("Testing data import completed.")


if __name__ == "__main__":
    import_testing_data()
