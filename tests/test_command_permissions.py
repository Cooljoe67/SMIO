from types import SimpleNamespace

import pytest

from src import config
from src.api.utils import command_processor
from src.mail.imap import ImapMailbox

# Both accounts share the default model, so only the account decides.
SHARED_MODEL = """
folder_lists:
  default:
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
    folder_list: default
    imap: {host: imap.second, username: second@test, password: x}
"""

SERVICE_COMMANDS = ["retrain", "reload model", "restore model", "logs 10"]


@pytest.fixture
def accounts(monkeypatch):
    parsed = config.parse_config(SHARED_MODEL)
    monkeypatch.setattr(config, "_config", parsed)
    return parsed.accounts


@pytest.fixture
def service_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        command_processor, "retrain_if_due", lambda db, **kwargs: calls.append("retrain") or {}
    )
    monkeypatch.setattr(
        command_processor.classifier, "reload_models", lambda: calls.append("reload") or ["default"]
    )
    monkeypatch.setattr(
        command_processor, "restore_previous_model", lambda: calls.append("restore") or {}
    )
    monkeypatch.setattr(
        command_processor, "_recent_cloud_run_logs", lambda count: calls.append("logs") or "line"
    )
    monkeypatch.setattr(command_processor, "append_summary_to_inbox", lambda *args, **kwargs: True)
    return calls


@pytest.mark.parametrize("command", SERVICE_COMMANDS)
def test_service_commands_are_refused_for_other_accounts(db, accounts, service_calls, command):
    _, second = accounts

    result = command_processor._run_command(db, command, second, is_primary=False)

    assert "not permitted for this account" in result["error"]
    assert command_processor._command_failed(result)
    assert service_calls == []


@pytest.mark.parametrize("command", SERVICE_COMMANDS)
def test_service_commands_run_for_the_primary_account(db, accounts, service_calls, command):
    first, _ = accounts

    result = command_processor._run_command(db, command, first, is_primary=True)

    assert "error" not in result
    assert len(service_calls) == 1


def test_undo_stays_available_to_other_accounts(db, accounts, monkeypatch):
    _, second = accounts
    calls = []
    monkeypatch.setattr(
        command_processor,
        "undo_processing_batches",
        lambda db, account, count: calls.append(account.id) or ([3], [7], []),
    )

    result = command_processor._run_command(db, "undo", second)

    assert calls == ["second"]
    assert not command_processor._command_failed(result)


class FakeMailBox:
    def __init__(self, messages):
        self.messages = messages
        self.folder = SimpleNamespace(
            exists=lambda name: True,
            status=lambda name, options: {"UIDVALIDITY": 1, "UIDNEXT": 1},
            create=lambda name: None,
            set=lambda name, readonly=False: None,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def fetch(self, criteria="ALL", mark_seen=False):
        return iter(self.messages)

    def move(self, uids, destination):
        pass

    def logout(self):
        pass


@pytest.mark.parametrize(("account_index", "is_primary", "restored"), [(0, True, True), (1, False, False)])
def test_command_mails_pass_on_whether_the_account_is_primary(
    db, accounts, service_calls, monkeypatch, account_index, is_primary, restored
):
    account = accounts[account_index]
    message = SimpleNamespace(
        uid="5", from_=account.email, subject="SMIO Command", text="restore model",
        html="", date=None,
    )
    notices = []
    monkeypatch.setattr(
        command_processor,
        "open_mailbox",
        lambda account: ImapMailbox(account, connection=FakeMailBox([message])),
    )
    monkeypatch.setattr(command_processor, "SessionLocal", lambda: db)
    monkeypatch.setattr(command_processor, "_load_command_attempts", lambda: [])
    monkeypatch.setattr(command_processor, "_save_command_attempt", lambda attempt_id, attempt: None)
    monkeypatch.setattr(command_processor, "_delete_command_attempt", lambda attempt_id: None)
    monkeypatch.setattr(
        command_processor,
        "_notify_command_failure",
        lambda command, reason, account: notices.append((account.id, reason)) or True,
    )

    result = command_processor.process_instruction_mails(account, is_primary=is_primary)

    assert (service_calls == ["restore"]) is restored
    if restored:
        assert result["errors"] == [] and notices == []
    else:
        [(notified_account, reason)] = notices
        assert notified_account == "second"
        assert "not permitted for this account" in reason
