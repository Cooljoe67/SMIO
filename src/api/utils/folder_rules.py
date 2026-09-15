"""Easy-to-edit rules for IMAP classification folders.

Change folder behavior here instead of scattering folder names through the code.
"""

TRASH_FOLDER = "Papierkorb"

FOLDER_RULES = {
    "delivery": {
        "folder": "INBOX/delivery",
        "retention_days": None,
    },
    "tech": {
        "folder": "INBOX/tech",
        "retention_days": 1,
    },
    "social": {
        "folder": "INBOX/social",
        "retention_days": 5,
    },
    "commercial": {
        "folder": "INBOX/commercial",
        "retention_days": 10,
    },
}

CLASSIFICATION_FOLDERS = set(FOLDER_RULES)
FOLDER_TO_CLASSIFICATION = {
    rule["folder"]: classification
    for classification, rule in FOLDER_RULES.items()
}


def folder_for_classification(classification):
    rule = FOLDER_RULES.get(classification)
    return rule["folder"] if rule else None


def classification_for_folder(folder):
    return FOLDER_TO_CLASSIFICATION.get(folder)
