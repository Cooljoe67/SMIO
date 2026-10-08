"""Easy-to-edit rules for IMAP classification folders.

Change folder behavior here instead of scattering folder names through the code.

Each entry in FOLDER_LISTS is one folder list. Its name is also the name of the
classifier model trained for it, and its folder keys (plus the implicit "other",
which stays in INBOX) are that model's labels. Adding or removing a folder
therefore means training a new model under a new list name.

Set "enabled" to False to stop sorting mails into a folder without retraining:
the model still predicts the label, but those mails stay in INBOX.
"""

TRASH_FOLDER = "Papierkorb"

DEFAULT_FOLDER_LIST = "default"

FOLDER_LISTS = {
    "default": {
        "delivery": {
            "folder": "INBOX/delivery",
            "enabled": True,
            "retention_days": None,
            "retention_days_unread": 20,
        },
        "tech": {
            "folder": "INBOX/tech",
            "enabled": True,
            "retention_days": 1,
            "retention_days_unread": 20,
        },
        "social": {
            "folder": "INBOX/social",
            "enabled": True,
            "retention_days": 5,
            "retention_days_unread": 20,
        },
        "commercial": {
            "folder": "INBOX/commercial",
            "enabled": True,
            "retention_days": 10,
            "retention_days_unread": 20,
        },
    },
}


def folder_rules(list_name=DEFAULT_FOLDER_LIST):
    try:
        return FOLDER_LISTS[list_name]
    except KeyError:
        raise ValueError(f"Unknown folder list: {list_name!r}") from None


def classification_folders(list_name=DEFAULT_FOLDER_LIST):
    return set(folder_rules(list_name))


def folder_for_classification(classification, list_name=DEFAULT_FOLDER_LIST):
    """Folder to sort a mail into, or None if it stays in INBOX."""
    rule = folder_rules(list_name).get(classification)
    if not rule or not rule.get("enabled", True):
        return None
    return rule["folder"]


def classification_for_folder(folder, list_name=DEFAULT_FOLDER_LIST):
    """Label for an IMAP folder, regardless of whether sorting into it is enabled."""
    for classification, rule in folder_rules(list_name).items():
        if rule["folder"] == folder:
            return classification
    return None


# Rules of the default list, kept for code that is not yet account-aware.
FOLDER_RULES = folder_rules(DEFAULT_FOLDER_LIST)
CLASSIFICATION_FOLDERS = classification_folders(DEFAULT_FOLDER_LIST)
