"""Import a labelled mail dataset (e.g. t1a-arbeitsblatt.tsv) as training data.

Rows are matched by their key (SCHLUESSEL), so the import can be repeated after
the file changes. Rows without a FOLDER label (the TEST split) are imported
without labels. The label, a key of the logistics folder list in config/smio.yaml,
is derived from the dataset's topic folder, priority, action and mail type; the
first matching rule in dataset_label wins.

The dataset company's own domain and name are replaced with neutral ones, so a
model trained on it does not learn one company's addresses. The company is
recognised from the rows whose sender is in the own domain.

Usage: python -m scripts.import_labelled_tsv [path] [--folder-list logistics] [--keep-company]
"""

import argparse
import csv
import re
from collections import Counter

from src.config import get_config
from src.db.database import SessionLocal, ensure_email_columns
from src.db.models import Email

DATASET_FOLDERS = {
    "Transportanfragen", "Transportaufträge", "Transportabwicklung", "Disposition & Touren",
    "Dokumente & Frachtpapiere", "Rechnungen & Abrechnung", "Schäden & Ereignisse",
    "Stammdaten & Verträge", "Partner & Dienstleister", "Marketing & Newsletter", "Sonstiges",
}
# Actions answered by writing back; escalations, approvals and complaints stay in the inbox.
REPLY_ACTIONS = {
    "Antwort erforderlich", "Bestätigung erforderlich", "Information anfordern",
    "Termin abstimmen", "Transportänderung bestätigen",
}
BOOLEANS = {"ja": True, "nein": False}
NEUTRAL_DOMAIN = "firma.example"
NEUTRAL_NAME = "Firma"
NEUTRALIZED_COLUMNS = (
    "senderName", "senderAddress", "recipientAddress", "subject", "body",
    "attachmentNames", "NOTE", "VORSCHLAG_BEGRUENDUNG",
)


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
    """Labels of the folder list, including the implicit "other"."""
    return get_config().labels(folder_list)


def dataset_label(row, prefix=""):
    """Label of a row from its true ("") or suggested ("VORSCHLAG_") columns."""
    folder = _value(row, f"{prefix}FOLDER")
    if folder is None:
        return None
    if folder not in DATASET_FOLDERS:
        raise ValueError(f"{row['SCHLUESSEL']}: unknown dataset folder {folder!r}")
    action = _value(row, f"{prefix}ACTION")
    if _value(row, f"{prefix}PRIORITY") == "Dringend":
        return "dringend"
    if folder == "Marketing & Newsletter":
        return "marketing"
    if action in REPLY_ACTIONS:
        return "zu_beantworten"
    if folder == "Rechnungen & Abrechnung":
        return "rechnungen"
    if folder == "Dokumente & Frachtpapiere":
        return "transportdokumente"
    if _value(row, f"{prefix}MAIL_TYPE_1") == "Statusupdate":
        return "statusmeldung"
    if action == "Keine Aktion":
        return "zur_kenntnis"
    return "other"


def own_domain(rows):
    """Most common sender domain of the rows marked as sent from the own domain."""
    domains = Counter(
        row["senderAddress"].rpartition("@")[2].lower()
        for row in rows
        if (row.get("isOwnDomainSender") or "").strip().lower() == "ja" and "@" in row["senderAddress"]
    )
    return domains.most_common(1)[0][0] if domains else None


def neutralizer(domain):
    """Replace the domain and the company name derived from it (nord-spedition.de ->
    "Nord Spedition", "Nord-Spedition", ...). Single words of the name are left
    alone, since they are often places."""
    name_parts = re.split(r"[-_.]+", domain.rsplit(".", 1)[0])
    domain_pattern = re.compile(re.escape(domain), re.IGNORECASE)
    name_pattern = re.compile(
        r"\b" + r"[\s\-_]*".join(map(re.escape, name_parts)) + r"\b", re.IGNORECASE
    )

    def neutralize(value):
        value = domain_pattern.sub(NEUTRAL_DOMAIN, value or "")
        # A one-word name would also hit reference numbers and ordinary words.
        return name_pattern.sub(NEUTRAL_NAME, value) if len(name_parts) > 1 else value
    return neutralize


def _label(row, prefix, labels):
    label = dataset_label(row, prefix)
    if label is not None and label not in labels:
        raise ValueError(f"{row['SCHLUESSEL']}: label {label!r} is not in the folder list")
    return label


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
    email.text = _value(row, "body") or ""
    email.attachment_names = _value(row, "attachmentNames")
    email.is_own_domain_sender = _boolean(row, "isOwnDomainSender")
    email.is_reply = _boolean(row, "isReply")
    email.is_forward = _boolean(row, "isForward")

    email.folder_list = folder_list
    email.true_label = _label(row, "", labels)
    email.true_status = _value(row, "STATUS")
    email.true_priority = _value(row, "PRIORITY")
    email.true_action = _value(row, "ACTION")
    email.true_mail_type_1 = _value(row, "MAIL_TYPE_1")
    email.true_mail_type_2 = _value(row, "MAIL_TYPE_2")
    email.reviewed = _boolean(row, "GEPRUEFT")
    email.note = _value(row, "NOTE")
    email.traits = _value(row, "TRAITS")

    email.classification = _label(row, "VORSCHLAG_", labels)
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


def import_tsv(db, path, folder_list, keep_company=False):
    labels = folder_labels(folder_list)
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    # The file stores line breaks as a literal backslash-n; convert them first so
    # a name at the start of a line is still a separate word.
    rows = [{**row, "body": (row.get("body") or "").replace("\\n", "\n")} for row in rows]
    domain = None if keep_company else own_domain(rows)
    if domain:
        neutralize = neutralizer(domain)
        rows = [
            {**row, **{column: neutralize(row.get(column)) for column in NEUTRALIZED_COLUMNS}}
            for row in rows
        ]

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
    parser.add_argument(
        "--keep-company",
        action="store_true",
        help="keep the dataset company's own domain and name instead of neutral ones",
    )
    args = parser.parse_args()

    ensure_email_columns()
    db = SessionLocal()
    try:
        print(import_tsv(db, args.path, args.folder_list, keep_company=args.keep_company))
    finally:
        db.close()


if __name__ == "__main__":
    main()
