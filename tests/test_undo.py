from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import inbox as inbox_router
from src.api.utils import command_processor, inbox_processor
from src.db.database import get_db
from src.db.models import Email


class FakeFolders:
    def __init__(self, mailbox):
        self.mailbox = mailbox

    def exists(self, name):
        return True

    def create(self, name):
        pass

    def set(self, name, readonly=False):
        self.mailbox.current = name


class FakeMailBox:
    """Records moves and hands out fresh INBOX UIDs, like a real IMAP server."""

    def __init__(self):
        self.folder = FakeFolders(self)
        self.current = "INBOX"
        self.moves = []
        self.inbox_uids = {}
        self.next_uid = 1000

    def login(self, user, password):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def move(self, uids, destination):
        self.moves.append((self.current, list(uids), destination))

    def uids(self, criteria):
        message_id = str(criteria).split('"')[-2]
        if message_id not in self.inbox_uids:
            self.next_uid += 1
            self.inbox_uids[message_id] = str(self.next_uid)
        return [self.inbox_uids[message_id]]


@pytest.fixture
def mailbox(monkeypatch):
    fake = FakeMailBox()
    monkeypatch.setattr(inbox_processor, "MailBox", lambda host: fake)
    return fake


def _email(db, batch, folder, source="model", **fields):
    email = Email(
        uid=fields.pop("uid", str(batch * 10)),
        message_id=fields.pop("message_id", f"<{batch}-{folder}@test>"),
        folder=folder,
        classification=fields.pop("classification", "delivery"),
        classification_source=source,
        processed=1,
        processing_batch=batch,
        processed_at=datetime(2026, 10, 7),
        **fields,
    )
    db.add(email)
    db.commit()
    return email


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("UNDO", "undo"),
        ("undo 3", "undo 3"),
        ("SMIO:  Undo   5\nthanks", "undo 5"),
        ("undo3", None),
    ],
)
def test_undo_command_accepts_optional_count(body, expected):
    assert command_processor._normalize_command("AW: SMIO Daily Summary", body) == expected


@pytest.mark.parametrize("command", ["undo 0", "undo 21"])
def test_undo_command_rejects_count_out_of_range(db, command):
    result = command_processor._run_command(db, command)
    assert "error" in result


def test_undo_command_passes_count(db, monkeypatch):
    calls = []
    monkeypatch.setattr(
        command_processor,
        "undo_processing_batches",
        lambda db, count: calls.append(count) or ([5, 4], [1, 2], []),
    )

    result = command_processor._run_command(db, "undo 2")

    assert calls == [2]
    assert result["batch_ids"] == [5, 4]
    assert result["restored"] == 2
    assert not command_processor._command_failed(result)


def test_undo_restores_last_n_batches_skipping_smio_only_batches(db, mailbox):
    oldest = _email(db, 1, "INBOX/Delivery")
    first = _email(db, 2, "INBOX/Delivery")
    second = _email(db, 3, "INBOX/Commercial", classification="commercial")
    summary = _email(db, 4, "INBOX", source="system", classification="other")

    batch_ids, restored, errors = inbox_processor.undo_processing_batches(db, 2)

    assert batch_ids == [3, 2]
    assert sorted(restored) == sorted([first.id, second.id])
    assert errors == []
    for email in (first, second):
        db.refresh(email)
        assert email.folder == "INBOX"
        assert email.processing_batch is None
        assert email.classification_source == "undo"
        assert int(email.uid) > 1000
    db.refresh(oldest)
    db.refresh(summary)
    assert oldest.processing_batch == 1
    assert summary.processing_batch == 4
    assert {(src, dest) for src, _, dest in mailbox.moves} == {
        ("INBOX/Delivery", "INBOX"),
        ("INBOX/Commercial", "INBOX"),
    }


def test_undone_mails_are_not_processed_again(db, mailbox):
    _email(db, 1, "INBOX/Delivery")
    inbox_processor.undo_processing_batches(db, 1)

    _, results, errors = inbox_processor.process_unprocessed_emails(db)

    assert results == []
    assert errors == []


def test_undo_skips_removed_mails(db, mailbox):
    removed = _email(db, 1, "Papierkorb", removed_at=datetime(2026, 10, 7))

    batch_ids, restored, _ = inbox_processor.undo_processing_batches(db, 1)

    assert batch_ids == []
    assert restored == []
    db.refresh(removed)
    assert removed.folder == "Papierkorb"
    assert mailbox.moves == []


def test_undo_rejects_invalid_count(db):
    with pytest.raises(ValueError):
        inbox_processor.undo_processing_batches(db, 0)


def test_undo_endpoint_accepts_count(db, monkeypatch):
    calls = []
    monkeypatch.setattr(
        inbox_router,
        "undo_processing_batches",
        lambda db, count: calls.append(count) or ([2, 1], [7], []),
    )
    app = FastAPI()
    app.include_router(inbox_router.router)
    app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app)

    response = client.post("/inbox/undo_last_processing?count=2")
    assert response.status_code == 200
    assert response.json() == {"batch_ids": [2, 1], "restored": 1, "failed": 0, "errors": []}
    assert calls == [2]

    assert client.post("/inbox/undo_last_processing?count=21").status_code == 422
