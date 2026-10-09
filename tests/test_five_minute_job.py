import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config
from src.api.routers import jobs
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
    monkeypatch.setattr(workflows, "reload_config", lambda **kwargs: None)
    return parsed.accounts


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(jobs.router)
    return TestClient(app)


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


def test_failed_account_answers_500_after_all_accounts_ran(accounts, client, monkeypatch):
    events = []

    def process(account, is_primary):
        events.append(account.id)
        if account.id == "first":
            raise RuntimeError("login failed")
        return {"processed": 1}

    monkeypatch.setattr(workflows, "_process_account", process)
    monkeypatch.setattr(workflows, "persist_database", lambda: events.append("persisted"))

    response = client.post("/jobs/five-minute")

    assert response.status_code == 500
    assert response.json() == {
        "accounts": {
            "first": {"error": "login failed"},
            "second": {"processed": 1},
            "mail": {"skipped": True, "reason": "provider_gmail_not_supported"},
        }
    }
    assert events == ["first", "second", "persisted"]


def test_skipped_accounts_and_runs_are_not_failures(accounts, client, monkeypatch):
    monkeypatch.setattr(workflows, "_process_account", lambda account, is_primary: {"processed": 0})
    monkeypatch.setattr(workflows, "persist_database", lambda: False)

    response = client.post("/jobs/five-minute")

    assert response.status_code == 200
    assert response.json()["accounts"]["mail"]["skipped"] is True

    # A run skipped because another workflow holds the lock has no account results.
    assert workflows._workflow_lock.acquire(blocking=False)
    try:
        response = client.post("/jobs/five-minute")
    finally:
        workflows._workflow_lock.release()
    assert response.status_code == 200
    assert response.json() == {"skipped": True, "reason": "workflow_in_progress"}
