"""Import a labelled mail dataset (e.g. t1a-arbeitsblatt.tsv) as training data.

Rows are matched by their key (SCHLUESSEL), so the import can be repeated after
the file changes. Rows without a FOLDER label (the TEST split) are imported
without labels. Folder labels are mapped from the dataset's folder names to the
keys of a folder list in config/smio.yaml; "Sonstiges" is the implicit "other".

Usage: python -m scripts.import_labelled_tsv [path] [--folder-list logistics]
"""

import argparse
import csv

from src.config import get_config
from src.db.database import SessionLocal, ensure_email_columns
from src.db.models import Email

OTHER_FOLDER_NAME = "Sonstiges"
BOOLEANS = {"ja": True, "nein": False}


def _value(row, column):
    value = (row.get(column) or "").strip()
    return value or None


def _boolean(row, column):
    value = _value(row, column)
    if value is None:
        return None
    if value.lower() not in BOOLEANS:
        raise ValueError(f"{row['SCHLUESSEL']}: {column} must be ja/nein, not {value!r}")
    return BOOLEANS[value.lower()]


def folder_labels(folder_list):
    """Dataset folder name -> label of the folder list."""
    rules = get_config().folder_lists[folder_list]
    labels = {rule.folder or label: label for label, rule in rules.items()}
    labels[OTHER_FOLDER_NAME] = "other"
    return labels


def _label(row, column, labels):
    name = _value(row, column)
    if name is None:
        return None
    if name not in labels:
        raise ValueError(f"{row['SCHLUESSEL']}: folder {name!r} is not in the folder list")
    return labels[name]


def apply_row(email, row, folder_list, labels):
    email.dataset_key = row["SCHLUESSEL"]
    email.dataset_split = _value(row, "split")
    email.difficulty = _value(row, "DIFFICULTY")
    email.case_ref = _value(row, "VORGANG")
    email.direction = _value(row, "direction")
    email.sender_name = _value(row, "senderName")
    email.sender = _value(row, "senderAddress")
    email.recipients = _value(row, "recipientAddress")
    email.subject = _value(row, "subject")
    # The file stores line breaks as a literal backslash-n.
    email.text = (_value(row, "body") or "").replace("\\n", "\n")
    email.attachment_names = _value(row, "attachmentNames")
    email.is_own_domain_sender = _boolean(row, "isOwnDomainSender")
    email.is_reply = _boolean(row, "isReply")
    email.is_forward = _boolean(row, "isForward")

    email.folder_list = folder_list
    email.true_label = _label(row, "FOLDER", labels)
    email.true_status = _value(row, "STATUS")
    email.true_priority = _value(row, "PRIORITY")
    email.true_action = _value(row, "ACTION")
    email.true_mail_type_1 = _value(row, "MAIL_TYPE_1")
    email.true_mail_type_2 = _value(row, "MAIL_TYPE_2")
    email.reviewed = _boolean(row, "GEPRUEFT")
    email.note = _value(row, "NOTE")
    email.traits = _value(row, "TRAITS")

    email.classification = _label(row, "VORSCHLAG_FOLDER", labels)
    email.classification_source = "dataset" if email.classification else None
    email.status = _value(row, "VORSCHLAG_STATUS")
    email.priority = _value(row, "VORSCHLAG_PRIORITY")
    email.action = _value(row, "VORSCHLAG_ACTION")
    email.mail_type_1 = _value(row, "VORSCHLAG_MAIL_TYPE_1")
    email.mail_type_2 = _value(row, "VORSCHLAG_MAIL_TYPE_2")
    email.classification_reason = _value(row, "VORSCHLAG_BEGRUENDUNG")

    # Not a mailbox message (no account): never fetched, processed or moved.
    email.is_training_data = True
    email.processed = max(1, email.processed or 0)
    email.processing_batch = email.processing_batch or 0


def import_tsv(db, path, folder_list):
    labels = folder_labels(folder_list)
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    existing = {
        email.dataset_key: email
        for email in db.query(Email).filter(Email.dataset_key.isnot(None))
    }
    created = updated = 0
    for row in rows:
        email = existing.get(row["SCHLUESSEL"])
        if email is None:
            email = Email()
            db.add(email)
            created += 1
        else:
            updated += 1
        apply_row(email, row, folder_list, labels)
    db.commit()
    labelled = sum(1 for row in rows if _value(row, "FOLDER"))
    return {"created": created, "updated": updated, "labelled": labelled}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default="t1a-arbeitsblatt.tsv")
    parser.add_argument("--folder-list", default="logistics")
    args = parser.parse_args()

    ensure_email_columns()
    db = SessionLocal()
    try:
        print(import_tsv(db, args.path, args.folder_list))
    finally:
        db.close()


if __name__ == "__main__":
    main()
