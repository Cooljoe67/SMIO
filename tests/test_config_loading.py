import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from src import config
from src.api.utils import imap_client
from src.db import database
from src.scheduler import workflows
from src.storage import gcs

ROOT = Path(__file__).resolve().parents[1]


def _config_text(first_name):
    return f"""
folder_lists:
  default:
    delivery: {{}}
accounts:
  - id: primary
    provider: imap
    email: owner@test
    user_first_name: {first_name}
    folder_list: default
    imap: {{host: imap.test, username: owner@test, password: x}}
"""


BUNDLED = _config_text("Bundled")

TWO_ACCOUNTS = BUNDLED + """
  - id: second
    provider: imap
    email: second@test
    folder_list: default
    imap: {host: imap.second, username: "${SMIO_TEST_UNSET}", password: "${SMIO_TEST_UNSET_TOO}"}
"""


class FakeBlob:
    def __init__(self, objects, name):
        self.objects = objects
        self.name = name

    def exists(self):
        return self.name in self.objects

    def download_as_text(self, encoding=None):
        return self.objects[self.name]

    def download_to_filename(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            file.write(self.objects[self.name])

    def upload_from_filename(self, filename):
        self.objects[self.name] = Path(filename).read_text(encoding="utf-8")


class FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self.objects, name)


@pytest.fixture
def bucket(monkeypatch, tmp_path):
    """A fake GCS bucket and a read-only bundled config, as in the image."""
    fake = FakeBucket()
    monkeypatch.setattr(gcs, "enabled", lambda: True)
    monkeypatch.setattr(gcs, "_bucket", lambda: fake)
    bundled = tmp_path / "smio.yaml"
    bundled.write_text(BUNDLED, encoding="utf-8")
    bundled.chmod(0o444)
    monkeypatch.setattr(config, "CONFIG_PATH", bundled)
    monkeypatch.setattr(config, "_config", None)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("SMIO_CONFIG_SOURCE", raising=False)
    return fake.objects


@pytest.mark.parametrize("variable, value", [("K_SERVICE", "smio"), ("SMIO_CONFIG_SOURCE", "gcs")])
def test_service_reads_the_gcs_config_without_writing_the_bundled_file(
    bucket, monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)
    bucket[config.CONFIG_GCS_OBJECT] = _config_text("Remote")

    assert config.reload_config().accounts[0].user_first_name == "Remote"
    assert config.CONFIG_PATH.read_text(encoding="utf-8") == BUNDLED


def test_scripts_read_the_local_file_and_leave_gcs_alone(bucket):
    bucket[config.CONFIG_GCS_OBJECT] = _config_text("Remote")

    assert config.reload_config().accounts[0].user_first_name == "Bundled"
    assert config.CONFIG_PATH.read_text(encoding="utf-8") == BUNDLED
    assert bucket == {config.CONFIG_GCS_OBJECT: _config_text("Remote")}


def test_scripts_never_upload_their_config(bucket):
    config.reload_config()

    assert bucket == {}


def test_service_uploads_the_bundled_config_when_gcs_has_none(bucket, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "smio")

    assert config.reload_config().accounts[0].user_first_name == "Bundled"
    assert bucket == {config.CONFIG_GCS_OBJECT: BUNDLED}


@pytest.mark.parametrize(
    "broken",
    ["folder_lists: [", BUNDLED.replace("folder_list: default", "folder_lists: default")],
    ids=["yaml", "schema"],
)
def test_broken_gcs_config_at_startup_falls_back_to_the_bundled_one(
    bucket, monkeypatch, caplog, broken
):
    monkeypatch.setenv("K_SERVICE", "smio")
    bucket[config.CONFIG_GCS_OBJECT] = broken

    with caplog.at_level(logging.ERROR, logger="src.config"):
        assert config.reload_config().accounts[0].user_first_name == "Bundled"
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_broken_gcs_config_later_keeps_the_previous_one(bucket, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "smio")
    previous = config.parse_config(_config_text("Previous"))
    monkeypatch.setattr(config, "_config", previous)
    bucket[config.CONFIG_GCS_OBJECT] = "folder_lists: ["

    assert config.reload_config() is previous


def test_account_with_an_unset_variable_is_kept_but_unavailable(caplog):
    with caplog.at_level(logging.ERROR, logger="src.config"):
        parsed = config.parse_config(TWO_ACCOUNTS)

    assert [account.id for account in parsed.accounts] == ["primary", "second"]
    assert parsed.account("primary").unavailable is None
    assert parsed.account("second").unavailable == (
        "environment variable(s) not set: SMIO_TEST_UNSET, SMIO_TEST_UNSET_TOO"
    )
    assert "'second' is unavailable" in caplog.text


def test_unavailable_primary_account_stays_the_primary_one():
    parsed = config.parse_config(
        TWO_ACCOUNTS.replace("host: imap.test", 'host: "${SMIO_TEST_UNSET}"')
    )

    assert parsed.accounts[0].id == "primary"
    assert "SMIO_TEST_UNSET" in parsed.accounts[0].unavailable


def test_unavailable_account_is_refused_before_connecting(monkeypatch):
    account = config.parse_config(TWO_ACCOUNTS).account("second")
    monkeypatch.setattr(imap_client, "MailBox", lambda host: pytest.fail("connected"))

    with pytest.raises(RuntimeError, match="'second' is unavailable: environment variable"):
        imap_client.open_mailbox(account)


def test_five_minute_workflow_reports_the_unavailable_account(monkeypatch):
    monkeypatch.setattr(config, "_config", config.parse_config(TWO_ACCOUNTS))
    monkeypatch.setattr(workflows, "reload_config", lambda **kwargs: None)
    monkeypatch.setattr(workflows, "persist_database", lambda: False)
    monkeypatch.setattr(
        imap_client, "MailBox", lambda host: SimpleNamespace(login=lambda user, password: host)
    )

    def fetch_inbox(account):
        imap_client.open_mailbox(account)
        return []

    monkeypatch.setattr(workflows, "fetch_inbox", fetch_inbox)
    monkeypatch.setattr(workflows, "process_instruction_mails", lambda account, is_primary: {})
    monkeypatch.setattr(workflows, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(workflows, "process_unprocessed_emails", lambda db, account: (None, [], []))
    monkeypatch.setattr(workflows, "update_statuses_from_replies", lambda db, account: {})

    results = workflows.run_five_minute_workflow()["accounts"]

    assert results["primary"]["processed"] == 0
    assert results["second"] == {"error": (
        "Account 'second' is unavailable: "
        "environment variable(s) not set: SMIO_TEST_UNSET, SMIO_TEST_UNSET_TOO"
    )}


def test_schema_migration_reads_the_config_only_for_unassigned_mails(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'smio.db').as_posix()}")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE emails (id INTEGER PRIMARY KEY, processed INTEGER, classification TEXT, "
            "true_label TEXT, uid TEXT, account TEXT, is_training_data BOOLEAN)"
        ))
        connection.execute(text(
            "INSERT INTO emails (id, processed, uid, account, is_training_data) "
            "VALUES (1, 1, '5', 'primary', 0), (2, 0, NULL, NULL, 1)"
        ))
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "persist_database", lambda: False)

    def unreadable_config():
        raise ValueError("the config must not be needed")

    monkeypatch.setattr(database, "primary_account", unreadable_config)

    database.ensure_email_columns()

    with engine.connect() as connection:
        accounts = connection.execute(text("SELECT account FROM emails ORDER BY id")).scalars().all()
    assert accounts == ["primary", None]
    engine.dispose()


def test_gcs_bucket_is_read_when_used(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "smio-test")
    assert gcs.enabled()

    monkeypatch.setenv("GCS_BUCKET", "")
    assert not gcs.enabled()


def test_dotenv_is_loaded_before_the_environment_is_read():
    # A stand-in for load_dotenv that sets what a .env file would.
    script = (
        "import os, dotenv\n"
        "dotenv.load_dotenv = lambda *args, **kwargs: os.environ.update("
        "GCS_BUCKET='from-dotenv', GCS_CONFIG_OBJECT='from-dotenv.yaml')\n"
        "from src import config\n"
        "print(config.gcs.enabled(), config.CONFIG_GCS_OBJECT)\n"
    )
    env = {
        name: value for name, value in os.environ.items()
        if name not in ("GCS_BUCKET", "GCS_CONFIG_OBJECT")
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )

    assert result.stdout.split() == ["True", "from-dotenv.yaml"]
