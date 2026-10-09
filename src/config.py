"""Folder lists and mail accounts, loaded from config/smio.yaml.

The deployed service reads GCS_CONFIG_OBJECT from GCS on every reload, so
production changes (e.g. switching a folder off) apply on the next workflow
run without a redeploy; the file in the image is only the initial upload and
the fallback. Scripts and local runs read the checked-out file. Secrets are
referenced as ${ENV_VAR} or, read at login, by Secret Manager name (*_secret).
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


def _one_source(connection, field):
    """A credential is given either itself or as <field>_secret, the name of a
    Secret Manager secret read at login."""
    if (getattr(connection, field) is None) == (getattr(connection, f"{field}_secret") is None):
        raise ValueError(f"set exactly one of '{field}' and '{field}_secret'")


class ImapConnection(_Strict):
    host: str
    username: str
    password: Optional[str] = None
    password_secret: Optional[str] = None

    @model_validator(mode="after")
    def _one_password(self):
        _one_source(self, "password")
        return self


class GmailConnection(_Strict):
    client_id: str
    client_secret: Optional[str] = None
    client_secret_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    refresh_token_secret: Optional[str] = None

    @model_validator(mode="after")
    def _one_source_each(self):
        _one_source(self, "client_secret")
        _one_source(self, "refresh_token")
        return self


class ExchangeConnection(_Strict):
    tenant_id: str
    client_id: str
    client_secret: Optional[str] = None
    client_secret_secret: Optional[str] = None

    @model_validator(mode="after")
    def _one_client_secret(self):
        _one_source(self, "client_secret")
        return self


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
    # Set by parse_config, not in the YAML: why the account cannot be used.
    unavailable: Optional[str] = None

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


def _expand_env(value, missing=None):
    """Replace ${VAR} references; unset ones raise, or go to `missing` as ""."""
    if isinstance(value, dict):
        return {key: _expand_env(item, missing) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, missing) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.groups()
        resolved = os.getenv(name)
        if resolved is None or resolved == "":
            if default is not None:
                return default
            if missing is None:
                raise ValueError(f"Environment variable {name} is not set")
            missing.append(name)
            return ""
        return resolved

    return _ENV_REFERENCE.sub(replace, value)


def _expand_account(account):
    """Expand one account; an unset variable makes only this account unavailable."""
    missing = []
    expanded = _expand_env(account, missing)
    if missing and isinstance(expanded, dict):
        expanded["unavailable"] = (
            f"environment variable(s) not set: {', '.join(dict.fromkeys(missing))}"
        )
        logger.error("Account %r is unavailable: %s", expanded.get("id"), expanded["unavailable"])
    return expanded


def parse_config(text):
    data = yaml.safe_load(text) or {}
    if isinstance(data, dict) and isinstance(data.get("accounts"), list):
        # Per account, so one unset variable leaves the others usable. An
        # unavailable account keeps its place: the primary stays the first one.
        accounts = [_expand_account(account) for account in data["accounts"]]
        data = {key: value for key, value in data.items() if key != "accounts"}
        return SmioConfig.model_validate({**_expand_env(data), "accounts": accounts})
    return SmioConfig.model_validate(_expand_env(data))


def _reads_from_gcs():
    """Only the deployed service reads GCS; scripts and local runs use their file."""
    return gcs.syncs()


def _gcs_config_text():
    """The service's config from GCS, or None to use the bundled file.

    Kept in memory: the bundled file is read-only in the image and stays the
    initial upload and the fallback.
    """
    text = gcs.download_text(CONFIG_GCS_OBJECT)
    if text is None:
        logger.info("No config in GCS at %s; uploading the bundled one", CONFIG_GCS_OBJECT)
        gcs.upload_file(CONFIG_PATH, CONFIG_GCS_OBJECT)
    return text


def _parse_and_check(text, check):
    config = parse_config(text)
    if check is not None:
        check(config)
    return config


def _load_config(startup, check):
    if _reads_from_gcs():
        try:
            # Downloaded inside the try: a copy that is not UTF-8 fails here.
            text = _gcs_config_text()
            if text is not None:
                return _parse_and_check(text, check)
        except (ValueError, yaml.YAMLError):
            if not startup:
                raise
            # Nothing to keep yet: run on the bundled config rather than not at all.
            logger.exception(
                "Invalid SMIO config in GCS at %s; using the bundled %s",
                CONFIG_GCS_OBJECT, CONFIG_PATH,
            )
    # At startup there is nothing to keep, so the file is used unchecked:
    # accounts it lacks, e.g. added to the GCS copy only, are left alone until
    # the GCS copy is fixed instead of every account stopping, and scripts that
    # need only the folder lists still run.
    text = CONFIG_PATH.read_text(encoding="utf-8")
    return _parse_and_check(text, None if startup else check)


_config = None
_config_lock = Lock()


def reload_config(check=None):
    """Re-read the configuration and return it.

    check(config) may raise ValueError to reject the new config. A broken or
    rejected config keeps the previous one; at startup a broken or rejected
    GCS copy falls back to the bundled file, which is used unchecked.
    """
    global _config
    with _config_lock:
        try:
            _config = _load_config(startup=_config is None, check=check)
        except Exception:
            if _config is None:
                raise
            # A broken edit in production must not stop mail processing.
            logger.exception("Invalid SMIO config; keeping the previous one")
        return _config


def get_config():
    # The service's GCS copy is checked like the workflow's reloads: a renamed
    # id accepted here would be kept by every later reload that rejects it.
    return _config or reload_config(check=require_stored_account_ids)


def require_account_ids(config, account_ids):
    """Reject a config without all of these account ids; a reload check.

    Stored mails are identified by their account id: a renamed id would import
    that inbox again as new mail and orphan the stored rows.
    """
    missing = set(account_ids) - {account.id for account in config.accounts}
    if missing:
        raise ValueError(
            f"Accounts {', '.join(sorted(missing))} have stored mails but are not "
            "configured; account ids must never change"
        )


def require_stored_account_ids(config):
    """require_account_ids for the mails in the database; a reload check."""
    # Imported here: src.db.database imports this module.
    from src.db.database import stored_account_ids

    require_account_ids(config, stored_account_ids())


def primary_account():
    """The account single-account code paths still use: the first one listed."""
    return get_config().accounts[0]


def resolve_account(account_id=None):
    """The account with this id, or the primary account when no id is given."""
    return get_config().account(account_id) if account_id else primary_account()
