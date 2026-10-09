import json
from types import SimpleNamespace

import pytest

from src import config
from src.db.models import Email
from src.scheduler import workflows

ACCOUNTS = """
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
  - id: mail
    provider: gmail
    email: mail@test
    folder_list: default
    gmail: {client_id: a, client_secret: b, refresh_token: c}
"""


@pytest.fixture
def accounts(monkeypatch):
    parsed = config.parse_config(ACCOUNTS)
    monkeypatch.setattr(config, "_config", parsed)
    monkeypatch.setattr(workflows, "reload_config", lambda: None)
    return parsed.accounts


def test_account_result_is_json_safe(accounts, monkeypatch):
    monkeypatch.setattr(workflows, "fetch_inbox", lambda account: [Email(id=1, subject="New")])
    monkeypatch.setattr(
        workflows, "process_instruction_mails", lambda account, is_primary: {"executed": [], "errors": []}
    )
    monkeypatch.setattr(workflows, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(workflows, "process_unprocessed_emails", lambda db, account: (4, [], []))
    monkeypatch.setattr(workflows, "update_statuses_from_replies", lambda db, account: {})

    result = workflows._process_account(accounts[0], is_primary=True)

    assert json.loads(json.dumps(result))["fetched"] == 1

