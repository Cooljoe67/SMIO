from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, inspect, text

from src import config
from src.api.utils import command_processor, imap_client, inbox_processor
from src.db import database
from src.db.models import Email
from src.scheduler import workflows

TWO_ACCOUNTS = """
folder_lists:
  default:
    delivery: {}
    tech: {}
  small:
    delivery: {}
accounts:
  - id: first
    provider: imap
    email: first@test
    folder_list: default
    imap: {host: imap.first, username: first@test, password: x}
  - id: second
    provider: imap
    email: second@test
    folder_list: small
    mailbox: {inbox: Inbox, folder_root: Inbox, trash: Deleted Items}
    imap: {host: imap.second, username: second@test, password: x}
"""


@pytest.fixture
def accounts(monkeypatch):
    parsed = config.parse_config(TWO_ACCOUNTS)
    monkeypatch.setattr(config, "_config", parsed)
    return parsed.account("first"), parsed.account("second")


class FakeMailBox:
    def __init__(self):
        self.folder = SimpleNamespace(
            exists=lambda name: True,
            create=lambda name: None,
            set=lambda name, readonly=False: None,
        )
        self.moves = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def move(self, uids, destination):
        self.moves.append((uids, destination))

    def uids(self, criteria):
        return ["900"]


@pytest.fixture
def mailboxes(monkeypatch):
    opened = {}

    def open_mailbox(account):
        return opened.setdefault(account.id, FakeMailBox())

    for module in (imap_client, inbox_processor):
        monkeypatch.setattr(module, "open_mailbox", open_mailbox)
    return opened


def _message(uid, message_id):
    return SimpleNamespace(
        uid=uid,
        headers={"Message-ID": message_id},
        flags=(),
        subject="Hello",
        from_="sender@test",
        date=datetime(2026, 10, 8),
        text="body",
        html="",
    )


def _email(db, account, batch, folder, **fields):
    email = Email(
        account=account.id,
        uid=fields.pop("uid", str(batch)),
        message_id=fields.pop("message_id", f"<{account.id}-{batch}@test>"),
        folder=folder,
        classification=fields.pop("classification", "delivery"),
        classification_source="model",
        processed=fields.pop("processed", 1),
        processing_batch=batch,
        **fields,
    )
    db.add(email)
    db.commit()
    return email


def test_same_message_in_two_accounts_is_stored_twice(db, accounts):
    first, second = accounts
    message = _message("1", "<shared@test>")

    imap_client._sync_message(db, first, message, "INBOX", allow_new=True)
    imap_client._sync_message(db, second, message, "Inbox", allow_new=True)
    db.commit()

    stored = db.query(Email).order_by(Email.account).all()
    assert [(email.account, email.folder) for email in stored] == [
        ("first", "INBOX"),
        ("second", "Inbox"),
    ]


def test_folder_move_is_a_correction_in_that_accounts_folder_layout(db, accounts):
    _, second = accounts
    email = _email(db, second, 1, "Inbox", classification="other")

    imap_client._sync_message(
        db, second, _message("1", email.message_id), "Inbox/delivery", allow_new=False
    )

    assert email.true_label == "delivery"
    assert email.classification_source == "imap_folder"
    assert email.folder_list == "small"


def test_undo_only_restores_the_accounts_own_batches(db, accounts, mailboxes):
    first, second = accounts
    first_mail = _email(db, first, 1, "INBOX/delivery")
    second_mail = _email(db, second, 2, "Inbox/delivery")

    batch_ids, restored, errors = inbox_processor.undo_processing_batches(db, first, 1)

    assert (batch_ids, restored, errors) == ([1], [first_mail.id], [])
    db.refresh(second_mail)
    assert second_mail.folder == "Inbox/delivery"
    assert mailboxes["first"].moves == [(["1"], "INBOX")]
    assert "second" not in mailboxes


def test_processing_uses_the_accounts_folder_list_and_layout(db, accounts, mailboxes, monkeypatch):
    first, second = accounts
    calls = []
    monkeypatch.setattr(
        inbox_processor.classifier,
        "predict_text",
        lambda text, folder_list: calls.append(folder_list) or ("delivery", 0.9),
    )
    _email(db, first, None, "INBOX", processed=0)
    mail = _email(db, second, None, "Inbox", processed=0, uid="7")

    _, results, errors = inbox_processor.process_unprocessed_emails(db, second)

    assert errors == []
    assert [result["email_id"] for result in results] == [mail.id]
    assert calls == ["small"]
    assert mailboxes["second"].moves == [("7", "Inbox/delivery")]


def test_account_without_model_leaves_mails_unprocessed(db, accounts, monkeypatch):
    _, second = accounts
    monkeypatch.setattr(inbox_processor.classifier, "model_available", lambda folder_list: False)
    mail = _email(db, second, None, "Inbox", processed=0)

    _, results, errors = inbox_processor.process_unprocessed_emails(db, second)

    assert results == []
    assert errors == [{"error": "no_model_for_folder_list:small"}]
    db.refresh(mail)
    assert mail.processed == 0


def test_failing_account_does_not_stop_the_others(accounts, monkeypatch):
    def process(account, is_primary):
        if account.id == "first":
            raise RuntimeError("login failed")
        return {"processed": 1, "primary": is_primary}

    monkeypatch.setattr(workflows, "_process_account", process)
    monkeypatch.setattr(workflows, "reload_config", lambda: None)
    monkeypatch.setattr(workflows, "persist_database", lambda: False)

    assert workflows.run_five_minute_workflow() == {
        "accounts": {
            "first": {"error": "login failed"},
            "second": {"processed": 1, "primary": False},
        }
    }


@pytest.mark.parametrize(
    ("attempt_account", "account_index", "is_primary", "handled"),
    [
        ("second", 1, False, True),
        ("second", 0, True, False),
        (None, 0, True, True),
        (None, 1, False, False),
    ],
)
def test_pending_commands_are_reported_to_their_own_account(
    accounts, monkeypatch, attempt_account, account_index, is_primary, handled
):
    attempt = {"command": "retrain", "status": "failed", "failure_reason": "boom"}
    if attempt_account:
        attempt["account"] = attempt_account
    notified = []
    monkeypatch.setattr(command_processor, "_load_command_attempts", lambda: [("a1", attempt)])
    monkeypatch.setattr(command_processor, "_delete_command_attempt", lambda attempt_id: None)
    monkeypatch.setattr(
        command_processor,
        "_notify_command_failure",
        lambda command, reason, account: notified.append(account.id) or True,
    )

    command_processor._check_timed_out_commands(accounts[account_index], is_primary)

    assert notified == ([accounts[account_index].id] if handled else [])


def test_retrain_is_refused_for_other_folder_lists(db, accounts):
    _, second = accounts

    result = command_processor._run_command(db, "retrain", second)

    assert "small" in result["error"]
    assert command_processor._command_failed(result)


def test_existing_mails_get_the_primary_account_and_original_folder_list(tmp_path, monkeypatch, accounts):
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE emails (id INTEGER PRIMARY KEY, processed INTEGER, "
            "classification TEXT, true_label TEXT, uid TEXT)"
        ))
        connection.execute(text(
            "INSERT INTO emails (id, processed, classification) VALUES (1, 1, 'tech'), (2, 0, NULL)"
        ))
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "persist_database", lambda: False)

    database.ensure_email_columns()

    with engine.connect() as connection:
        rows = connection.execute(text("SELECT account, folder_list FROM emails ORDER BY id")).all()
    # Only labelled mails get the original folder list; unclassified ones get theirs later.
    assert rows == [("first", "default"), ("first", None)]
    assert "ix_emails_account" in {index["name"] for index in inspect(engine).get_indexes("emails")}
    engine.dispose()
