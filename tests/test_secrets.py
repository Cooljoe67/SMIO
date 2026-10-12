import sys
import types
from types import SimpleNamespace

import pytest
from imap_tools import MailboxLoginError

from src import config
from src.api.utils import imap_client
from src.mail import imap as mail_imap
from src.storage import secret_store


def _account(provider, connection):
    return f"""
folder_lists:
  default:
    delivery: {{}}
accounts:
  - id: primary
    provider: {provider}
    email: owner@test
    folder_list: default
    {provider}: {{{connection}}}
"""


IMAP_WITH_SECRET = _account(
    "imap", "host: imap.test, username: owner@test, password_secret: imap-password"
)


@pytest.mark.parametrize(
    "provider, connection, error",
    [
        ("imap", "host: h, username: u", "'password' and 'password_secret'"),
        ("imap", "host: h, username: u, password: x, password_secret: s", "'password' and 'password_secret'"),
        ("gmail", "client_id: c, client_secret: x, refresh_token_secret: t", None),
        ("gmail", "client_id: c, client_secret_secret: s, refresh_token: t", None),
        ("gmail", "client_id: c, client_secret_secret: s", "'refresh_token' and 'refresh_token_secret'"),
        (
            "gmail",
            "client_id: c, client_secret: x, client_secret_secret: s, refresh_token: t",
            "'client_secret' and 'client_secret_secret'",
        ),
        ("exchange", "tenant_id: t, client_id: c, client_secret_secret: s", None),
        ("exchange", "tenant_id: t, client_id: c", "'client_secret' and 'client_secret_secret'"),
    ],
)
def test_each_credential_is_given_itself_or_as_a_secret_name(provider, connection, error):
    text = _account(provider, connection)
    if error is None:
        config.parse_config(text)
    else:
        with pytest.raises(ValueError, match=f"exactly one of {error}"):
            config.parse_config(text)


@pytest.fixture
def secret_manager(monkeypatch):
    """A fake google-cloud-secret-manager serving `values` and recording each access."""
    values = {}
    accessed = []

    class SecretManagerServiceClient:
        def access_secret_version(self, name):
            accessed.append(name)
            secret = name.split("/")[3]
            return SimpleNamespace(payload=SimpleNamespace(data=values[secret].encode("utf-8")))

    google = types.ModuleType("google")
    google.cloud = types.ModuleType("google.cloud")
    google.cloud.secretmanager = types.ModuleType("google.cloud.secretmanager")
    google.cloud.secretmanager.SecretManagerServiceClient = SecretManagerServiceClient
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", google.cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", google.cloud.secretmanager)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "smio-project")
    monkeypatch.setattr(secret_store, "_cache", {})
    return values, accessed


def test_secret_is_read_once_until_invalidated(secret_manager):
    values, accessed = secret_manager
    values["imap-second-password"] = "first"

    assert secret_store.read_secret("imap-second-password") == "first"
    values["imap-second-password"] = "rotated"
    assert secret_store.read_secret("imap-second-password") == "first"
    secret_store.invalidate("imap-second-password")
    assert secret_store.read_secret("imap-second-password") == "rotated"

    assert accessed == [
        "projects/smio-project/secrets/imap-second-password/versions/latest"
    ] * 2


def test_project_falls_back_to_gcp_project_id(secret_manager, monkeypatch):
    values, accessed = secret_manager
    values["imap-password"] = "secret"
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    monkeypatch.setenv("GCP_PROJECT_ID", "dotenv-project")

    secret_store.read_secret("imap-password")

    assert accessed == ["projects/dotenv-project/secrets/imap-password/versions/latest"]


class FakeServer:
    """Stands in for MailBox(host); accepts only its current password."""

    def __init__(self, password):
        self.password = password
        self.logins = []

    def __call__(self, host):
        return self

    def login(self, username, password):
        self.logins.append(password)
        if password != self.password:
            raise MailboxLoginError(("NO", [b"authentication failed"]), "OK")
        return self


def test_mailbox_logs_in_with_the_secret(monkeypatch):
    server = FakeServer("secret")
    monkeypatch.setattr(mail_imap, "MailBox", server)
    account = config.parse_config(IMAP_WITH_SECRET).accounts[0]

    assert mail_imap.connect(account) is server
    assert server.logins == ["secret"]


def test_empty_plain_password_is_refused(monkeypatch):
    monkeypatch.setattr(mail_imap, "MailBox", lambda host: pytest.fail("connected"))
    account = config.parse_config(
        _account("imap", "host: imap.test, username: owner@test, password: ''")
    ).accounts[0]

    with pytest.raises(RuntimeError, match="no IMAP password"):
        mail_imap.connect(account)


def test_rotated_password_is_read_again_after_a_failed_login(secret_manager, monkeypatch):
    values, accessed = secret_manager
    secret_store._cache["imap-password"] = "old"
    values["imap-password"] = "new"
    server = FakeServer("new")
    monkeypatch.setattr(mail_imap, "MailBox", server)
    account = config.parse_config(IMAP_WITH_SECRET).accounts[0]

    assert mail_imap.connect(account) is server
    assert server.logins == ["old", "new"]
    assert secret_store.read_secret("imap-password") == "new"


def test_wrong_password_is_not_retried_when_the_secret_is_unchanged(secret_manager, monkeypatch):
    values, accessed = secret_manager
    values["imap-password"] = "wrong"
    server = FakeServer("right")
    monkeypatch.setattr(mail_imap, "MailBox", server)
    account = config.parse_config(IMAP_WITH_SECRET).accounts[0]

    with pytest.raises(MailboxLoginError):
        mail_imap.connect(account)
    assert server.logins == ["wrong"]
    assert len(accessed) == 2
