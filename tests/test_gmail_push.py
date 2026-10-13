import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config
from src.api.routers import events as events_router
from src.db.models import MailSyncState
from src.scheduler import workflows
from tests.test_mail_gmail import GMAIL


def _envelope(address="owner@gmail.test", history_id="200"):
    data = json.dumps({"emailAddress": address, "historyId": int(history_id)}).encode()
    return {
        "message": {"data": base64.b64encode(data).decode(), "messageId": "1"},
        "subscription": "projects/p/subscriptions/smio-gmail-push",
    }


@pytest.fixture
def client(monkeypatch, db):
    monkeypatch.setattr(config, "_config", config.parse_config(GMAIL))
    monkeypatch.setattr(workflows, "reload_config", lambda **kwargs: None)
    monkeypatch.setattr(workflows, "persist_database", lambda: False)
    # The endpoint runs in another thread, which the in-memory test database
    # does not reach; _history_read has its own test.
    monkeypatch.setattr(workflows, "_history_read", lambda account_id, history_id: int(history_id) <= 150)
    processed = []
    monkeypatch.setattr(
        workflows, "_process_account",
        lambda account, is_primary: processed.append((account.id, is_primary)) or {"processed": 0},
    )
    app = FastAPI()
    app.include_router(events_router.router)
    test_client = TestClient(app)
    test_client.processed = processed
    return test_client


def test_a_notification_processes_its_account(client):
    response = client.post("/events/gmail", json=_envelope("OWNER@gmail.test", "200"))

    assert response.status_code == 200
    assert response.json() == {"accounts": {"mail": {"processed": 0}}}
    assert client.processed == [("mail", True)]


def test_a_notification_the_sync_already_read_is_skipped(client):
    response = client.post("/events/gmail", json=_envelope(history_id="120"))

    assert response.json() == {"skipped": True, "reason": "up_to_date"}
    assert client.processed == []


@pytest.mark.parametrize("envelope", [_envelope("someone@else.test"), {"message": {"data": "!!"}}])
def test_unknown_or_malformed_notifications_are_acknowledged(client, envelope):
    response = client.post("/events/gmail", json=envelope)

    assert response.status_code == 200 and "ignored" in response.json()
    assert client.processed == []


def test_a_running_workflow_makes_pub_sub_deliver_again(client):
    workflows._workflow_lock.acquire()
    try:
        response = client.post("/events/gmail", json=_envelope())
    finally:
        workflows._workflow_lock.release()

    assert response.status_code == 503
    assert client.processed == []


def test_the_sync_has_read_a_notification_once_every_cursor_is_past_it(db, monkeypatch):
    monkeypatch.setattr(workflows, "SessionLocal", lambda: db)
    assert not workflows._history_read("mail", "100")
    db.add_all([
        MailSyncState(account="mail", folder="INBOX", cursor="150"),
        MailSyncState(account="mail", folder="SENT", cursor="140"),
    ])
    db.commit()

    assert workflows._history_read("mail", "140")
    assert not workflows._history_read("mail", "145")
