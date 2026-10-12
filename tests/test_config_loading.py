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
from src.mail import imap as mail_imap
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
        content = self.objects[self.name]
        # Objects stored as bytes are decoded like GCS does.
        return content.decode(encoding) if isinstance(content, bytes) else content

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
    monkeypatch.delenv("SMIO_GCS", raising=False)
    return fake.objects


@pytest.mark.parametrize("variable, value", [("K_SERVICE", "smio"), ("SMIO_GCS", "1")])
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
    [
        "folder_lists: [",
        BUNDLED.replace("folder_list: default", "folder_lists: default"),
        # As saved by Windows PowerShell 5.1's Set-Content.
        BUNDLED.replace("delivery", "schäden").encode("cp1252"),
    ],
    ids=["yaml", "schema", "encoding"],
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
    monkeypatch.setattr(mail_imap, "MailBox", lambda host: pytest.fail("connected"))

    with pytest.raises(RuntimeError, match="'second' is unavailable: environment variable"):
        mail_imap.connect(account)


def test_five_minute_workflow_reports_the_unavailable_account(monkeypatch):
    monkeypatch.setattr(config, "_config", config.parse_config(TWO_ACCOUNTS))
    monkeypatch.setattr(workflows, "reload_config", lambda **kwargs: None)
    monkeypatch.setattr(workflows, "persist_database", lambda: False)
    monkeypatch.setattr(
        mail_imap, "MailBox", lambda host: SimpleNamespace(login=lambda user, password: host)
    )

    def fetch_inbox(account):
        mail_imap.connect(account)
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


def test_migration_can_be_the_first_config_load(tmp_path, bucket, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'smio.db').as_posix()}")
    with engine.begin() as connection:
        # A database from before multi-account support.
        connection.execute(text(
            "CREATE TABLE emails (id INTEGER PRIMARY KEY, processed INTEGER, classification TEXT, "
            "true_label TEXT, uid TEXT, is_training_data BOOLEAN)"
        ))
        connection.execute(text("INSERT INTO emails (processed, uid) VALUES (1, '5')"))
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "persist_database", lambda: False)
    # The service checks its first GCS copy against the stored account ids.
    monkeypatch.setenv("K_SERVICE", "smio")
    bucket[config.CONFIG_GCS_OBJECT] = _config_text("Remote")

    database.ensure_email_columns()

    with engine.connect() as connection:
        assert connection.execute(text("SELECT account FROM emails")).scalar() == "primary"
    engine.dispose()


RENAMED = BUNDLED.replace("id: primary", "id: marcus")


def _database_with_accounts(tmp_path, *accounts):
    engine = create_engine(f"sqlite:///{(tmp_path / 'accounts.db').as_posix()}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE emails (id INTEGER PRIMARY KEY, account TEXT)"))
        for account in accounts:
            connection.execute(text("INSERT INTO emails (account) VALUES (:account)"), {"account": account})
    return engine


def test_stored_account_ids_are_the_distinct_mailbox_accounts(tmp_path, monkeypatch):
    engine = _database_with_accounts(tmp_path, "primary", "second", "primary", None)
    monkeypatch.setattr(database, "engine", engine)

    assert database.stored_account_ids() == {"primary", "second"}
    engine.dispose()


@pytest.mark.parametrize("schema", [None, "CREATE TABLE emails (id INTEGER PRIMARY KEY)"])
def test_no_account_ids_are_stored_before_the_migration(tmp_path, monkeypatch, schema):
    engine = create_engine(f"sqlite:///{(tmp_path / 'new.db').as_posix()}")
    if schema:
        with engine.begin() as connection:
            connection.execute(text(schema))
    monkeypatch.setattr(database, "engine", engine)

    assert database.stored_account_ids() == set()
    engine.dispose()


def test_workflow_keeps_the_config_when_a_stored_account_id_disappears(tmp_path, monkeypatch, caplog):
    engine = _database_with_accounts(tmp_path, "primary", None)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(workflows, "persist_database", lambda: False)
    previous = config.parse_config(BUNDLED)
    monkeypatch.setattr(config, "_config", previous)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "renamed.yaml")
    config.CONFIG_PATH.write_text(RENAMED, encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="src.config"):
        result = workflows._run_exclusively("test", lambda: config.get_config())

    assert result is previous
    assert "primary have stored mails but are not configured" in caplog.text
    engine.dispose()


def test_first_load_outside_the_workflow_rejects_a_renamed_gcs_account(tmp_path, bucket, monkeypatch):
    # A request before the first workflow run after a cold start; later
    # reloads would keep whatever this load accepts.
    engine = _database_with_accounts(tmp_path, "primary")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setenv("K_SERVICE", "smio")
    bucket[config.CONFIG_GCS_OBJECT] = RENAMED

    assert config.resolve_account().id == "primary"
    engine.dispose()


def test_scripts_load_their_file_when_other_accounts_have_stored_mails(tmp_path, monkeypatch):
    # With GCS_BUCKET in .env a script opens the production database, which
    # also holds the mails of accounts added to the GCS copy only.
    engine = _database_with_accounts(tmp_path, "primary", "second")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(config, "_config", None)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "smio.yaml")
    config.CONFIG_PATH.write_text(BUNDLED, encoding="utf-8")

    assert list(config.get_config().folder_lists["default"]) == ["delivery"]
    engine.dispose()


def test_rejected_gcs_config_at_startup_falls_back_to_the_bundled_one(bucket, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "smio")
    bucket[config.CONFIG_GCS_OBJECT] = RENAMED

    loaded = config.reload_config(check=lambda new: config.require_account_ids(new, {"primary"}))

    assert [account.id for account in loaded.accounts] == ["primary"]


SECOND_IN_GCS = _config_text("Remote") + """
  - id: second
    provider: imap
    email: second@test
    folder_list: default
    imap: {host: imap.second, username: second@test, password_secret: imap-second-password}
"""


@pytest.mark.parametrize(
    "gcs_copy",
    [SECOND_IN_GCS + "  - {id: [\n", _config_text("Remote")],
    ids=["broken", "account-removed"],
)
def test_service_starts_on_the_bundled_file_without_accounts_only_in_gcs(
    tmp_path, bucket, monkeypatch, caplog, gcs_copy
):
    # 'second' was added to the GCS copy only, as add_account.ps1 says, and
    # has stored mails; the bundled file does not list it.
    engine = _database_with_accounts(tmp_path, "primary", "second")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(workflows, "persist_database", lambda: False)
    monkeypatch.setenv("K_SERVICE", "smio")
    bucket[config.CONFIG_GCS_OBJECT] = gcs_copy

    with caplog.at_level(logging.ERROR, logger="src.config"):
        started = workflows._run_exclusively("test", config.get_config)

    assert [account.id for account in started.accounts] == ["primary"]
    assert started.accounts[0].user_first_name == "Bundled"
    assert "Invalid SMIO config in GCS" in caplog.text

    # Fixing the GCS copy brings 'second' back on the next run.
    bucket[config.CONFIG_GCS_OBJECT] = SECOND_IN_GCS
    fixed = workflows._run_exclusively("test", config.get_config)

    assert [account.id for account in fixed.accounts] == ["primary", "second"]
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
