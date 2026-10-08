# scripts/import_training_data.py

from src.config import primary_account
from src.db.database import SessionLocal
from src.db.models import Email
from src.api.utils.folder_rules import folder_path, folder_rules
from src.api.utils.imap_client import open_mailbox

def import_training_data():
    db = SessionLocal()
    account = primary_account()

    with open_mailbox(account) as mailbox:
        for label in folder_rules(account):
            folder = folder_path(label, account)

            # Check if folder exists
            try:
                mailbox.folder.set(folder)
            except Exception:
                print(f"Folder not found: {folder}")
                continue

            print(f"Importing from {folder} → true_label: {label}")

            for msg in mailbox.fetch():
                email = Email(
                    account=account.id,
                    subject=msg.subject,
                    sender=msg.from_,
                    date=msg.date,
                    text=msg.text,
                    html=msg.html,
                    folder=folder,

                    # Ground truth label for testing
                    true_label=label,
                    is_training_data=True,

                    # Model prediction will be added later
                    classification=None,
                    confidence=None,
                )
                db.add(email)

        db.commit()
        print("training data import completed.")


if __name__ == "__main__":
    import_training_data()
