# scripts/import_training_data.py

from src.config import primary_account
from src.db.database import SessionLocal, engine, ensure_email_columns
from src.db.models import Base, Email
from src.api.utils.folder_rules import folder_path, folder_rules
from src.mail import open_mailbox

def import_training_data():
    # The insert names every mapped column, so an old database is migrated first.
    Base.metadata.create_all(bind=engine)
    ensure_email_columns()
    db = SessionLocal()
    account = primary_account()
    # Mails already stored (in the mailbox or by an earlier import) are skipped,
    # so the import can be repeated without training on duplicates.
    stored_message_ids = {
        message_id
        for (message_id,) in db.query(Email.message_id).filter(Email.message_id.isnot(None))
    }

    with open_mailbox(account) as mailbox:
        existing = set(mailbox.folders())
        for label in folder_rules(account):
            folder = folder_path(label, account)
            if folder not in existing:
                print(f"Folder not found: {folder}")
                continue

            print(f"Importing from {folder} → true_label: {label}")

            for msg in mailbox.read_folder(folder).messages:
                message_id = msg.message_id or None
                if message_id in stored_message_ids:
                    continue
                if message_id:
                    stored_message_ids.add(message_id)

                email = Email(
                    # Labelled data belongs to no mailbox, like the TSV import.
                    account=None,
                    folder_list=account.folder_list,
                    message_id=message_id,
                    subject=msg.subject,
                    sender=msg.sender,
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
