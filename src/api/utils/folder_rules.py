"""Folder rules of a mail account, as configured in config/smio.yaml.

Each folder list is named after the classifier model trained for it; its keys
(plus the implicit "other", which stays in the inbox) are the model's labels.
A folder with "enabled: false" keeps its label, but its mails stay in the inbox.

Every helper takes an account and falls back to the primary account for code
that is not yet account-aware.
"""

from src.config import get_config, primary_account


def _account(account):
    return account or primary_account()


def folder_list_name(account=None):
    return _account(account).folder_list


def folder_rules(account=None):
    return get_config().folder_lists[folder_list_name(account)]


def classification_folders(account=None):
    """Labels the model predicts, without "other" and manual folders."""
    return {label for label, rule in folder_rules(account).items() if not rule.manual}


def folder_path(classification, account=None):
    """IMAP folder of a label, whether or not sorting into it is enabled."""
    account = _account(account)
    rule = folder_rules(account).get(classification)
    if rule is None:
        return None
    name = rule.folder or classification
    root = account.mailbox.folder_root
    return f"{root}/{name}" if root else name


def folder_for_classification(classification, account=None):
    """Folder to sort a mail into, or None if it stays in the inbox."""
    rule = folder_rules(account).get(classification)
    if rule is None or not rule.enabled or rule.manual:
        return None
    return folder_path(classification, account)


def classification_for_folder(folder, account=None):
    """Label of a folder, whether or not sorting into it is enabled; None for
    the inbox, manual folders and folders outside the list."""
    for classification in classification_folders(account):
        if folder_path(classification, account) == folder:
            return classification
    return None


def inbox_folder(account=None):
    return _account(account).mailbox.inbox


def trash_folder(account=None):
    return _account(account).mailbox.trash
