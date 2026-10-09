"""Folder lists and mail accounts, loaded from config/smio.yaml.

When GCS is enabled the file is downloaded from GCS_CONFIG_OBJECT on every
reload, so production changes (e.g. switching a folder off) apply on the next
workflow run without a redeploy. Secrets are referenced as ${ENV_VAR}.
"""

import logging
import os
import re
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]

# Before anything reads the environment, including the modules imported below.
load_dotenv(REPO_ROOT / ".env")

from src.storage import gcs  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.getenv("SMIO_CONFIG_PATH", REPO_ROOT / "config" / "smio.yaml"))
CONFIG_GCS_OBJECT = os.getenv("GCS_CONFIG_OBJECT", "config/smio.yaml")

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FolderRule(_Strict):
    # Folder name below the account's folder_root; defaults to the label.
    folder: Optional[str] = None
    enabled: bool = True
    retention_days: Optional[int] = None
    retention_days_unread: Optional[int] = None


class MailboxLayout(_Strict):
    inbox: str = "INBOX"
    folder_root: str = "INBOX"
    trash: str = "Papierkorb"
    # Read-only, for recognising replies; None: the folder flagged \Sent.
    sent: Optional[str] = None


class ImapConnection(_Strict):
    host: str
    username: str
    password: str


class GmailConnection(_Strict):
    client_id: str
    client_secret: str
    refresh_token: str


class ExchangeConnection(_Strict):
    tenant_id: str
    client_id: str
    client_secret: str


class Account(_Strict):
    id: str
    provider: Literal["imap", "gmail", "exchange"]
    email: str
    user_first_name: str = "there"
    folder_list: str
    mailbox: MailboxLayout = MailboxLayout()
    imap: Optional[ImapConnection] = None
    gmail: Optional[GmailConnection] = None
    exchange: Optional[ExchangeConnection] = None

    @model_validator(mode="after")
    def _connection_matches_provider(self):
        for provider in ("imap", "gmail", "exchange"):
            configured = getattr(self, provider) is not None
            if provider == self.provider and not configured:
                raise ValueError(f"account {self.id!r} needs a '{provider}:' section")
            if provider != self.provider and configured:
                raise ValueError(
                    f"account {self.id!r} uses provider {self.provider!r} "
                    f"but also has a '{provider}:' section"
                )
        return self


class SmioConfig(_Strict):
    folder_lists: dict[str, dict[str, FolderRule]]
    accounts: list[Account]

    @model_validator(mode="after")
    def _check_references(self):
        for list_name, rules in self.folder_lists.items():
            if not rules:
                raise ValueError(f"folder list {list_name!r} has no folders")
            if "other" in rules:
                raise ValueError(
                    f"folder list {list_name!r}: 'other' is implicit and stays in the inbox"
                )
        if not self.accounts:
            raise ValueError("at least one account is required")
        ids = [account.id for account in self.accounts]
        if len(ids) != len(set(ids)):
            raise ValueError("account ids must be unique")
        for account in self.accounts:
            if account.folder_list not in self.folder_lists:
                raise ValueError(
                    f"account {account.id!r} uses unknown folder list {account.folder_list!r}"
                )
        return self

    def account(self, account_id):
        for account in self.accounts:
            if account.id == account_id:
                return account
        raise ValueError(f"Unknown account: {account_id!r}")


def _expand_env(value):
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.groups()
        resolved = os.getenv(name)
        if resolved is None or resolved == "":
            if default is None:
                raise ValueError(f"Environment variable {name} is not set")
            return default
        return resolved

    return _ENV_REFERENCE.sub(replace, value)


def parse_config(text):
    return SmioConfig.model_validate(_expand_env(yaml.safe_load(text) or {}))


def _sync_from_gcs():
    if not gcs.enabled():
        return
    if not gcs.download_file(CONFIG_GCS_OBJECT, CONFIG_PATH):
        logger.info("No config in GCS at %s; uploading the local one", CONFIG_GCS_OBJECT)
        gcs.upload_file(CONFIG_PATH, CONFIG_GCS_OBJECT)


_config = None
_config_lock = Lock()


def reload_config():
    """Re-read the configuration (from GCS when enabled) and return it."""
    global _config
    with _config_lock:
        try:
            _sync_from_gcs()
            _config = parse_config(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            if _config is None:
                raise
            # A broken edit in production must not stop mail processing.
            logger.exception("Invalid SMIO config; keeping the previous one")
        return _config


def get_config():
    return _config or reload_config()


def primary_account():
    """The account single-account code paths still use: the first one listed."""
    return get_config().accounts[0]


def resolve_account(account_id=None):
    """The account with this id, or the primary account when no id is given."""
    return get_config().account(account_id) if account_id else primary_account()
