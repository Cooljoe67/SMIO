import base64
import json as jsonlib
from datetime import datetime, timedelta, timezone
from email import message_from_bytes
from email.utils import format_datetime

import pytest

from src import config
from src.db.models import Email, MailEvent
from src.mail import CursorExpired
from src.mail import gmail as gmail_module
from src.mail import sync as mail_sync
from src.mail.gmail import API, GmailMailbox

NOW = datetime.now(timezone.utc).replace(microsecond=0)
LABEL = "1: Dringend"


class Response:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self.content = b"" if body is None else jsonlib.dumps(body).encode()
        self.text = self.content.decode()

    def json(self):
        return jsonlib.loads(self.content)


class FakeGmail:
    """The parts of the Gmail API the adapter uses, two results per page."""

    def __init__(self):
        self.labels = {name: name for name in ("INBOX", "SENT", "UNREAD", "TRASH", "DRAFT")}
        self.messages = {}
        self.history = []
        self.history_id = 100
        self.oldest_history = 0
        self.drafts = []
        self.calls = []

    def add(self, message_id, *labels, days_old=0, subject="Lieferung", unread=True):
        date = NOW - timedelta(days=days_old)
        raw = (
            f"From: Sender <sender@test>\r\nSubject: {subject}\r\n"
            f"Message-ID: <{message_id}@test>\r\nDate: {format_datetime(date)}\r\n\r\nHallo"
        ).encode()
        self.messages[message_id] = {
            "labels": set(self._ids(labels)) | ({"UNREAD"} if unread else set()),
            "raw": raw, "thread": f"t-{message_id}", "date": date.timestamp(),
        }

    def _ids(self, labels):
        """Label ids, like Gmail stores them; tests may name user labels."""
        return [self.labels.get(label, label) for label in labels]

    def _record(self, **change):
        self.history_id += 1
        self.history.append({"id": str(self.history_id), **change})

    def deliver(self, message_id, *labels, **kwargs):
        self.add(message_id, *labels, **kwargs)
        self._record(messagesAdded=[{"message": {"id": message_id, "labelIds": self._ids(labels)}}])

    def relabel(self, message_id, add=(), remove=()):
        add, remove = self._ids(add), self._ids(remove)
        labels = self.messages[message_id]["labels"]
        labels.difference_update(remove)
        labels.update(add)
        if add:
            self._record(labelsAdded=[{
                "message": {"id": message_id, "labelIds": sorted(labels)}, "labelIds": add,
            }])

    def _page(self, items, key, params, extra=None):
        start = int(params.get("pageToken") or 0)
        body = {key: items[start:start + 2], **(extra or {})}
        if start + 2 < len(items):
            body["nextPageToken"] = str(start + 2)
        return body

    def _matches(self, message_id, message, query):
        for term in (query or "").split():
            name, _, value = term.partition(":")
            if name == "after" and message["date"] < int(value):
                return False
            if name == "rfc822msgid" and value != f"{message_id}@test":
                return False
            if name == "from" and value not in message["raw"].decode():
                return False
        return True

    def request(self, method, url, params=None, json=None, timeout=None):
        path = url.removeprefix(API + "/")
        params = params or {}
        self.calls.append((method, path, params.get("format")))
        if (method, path) == ("GET", "labels"):
            return Response(200, {"labels": [{"id": i, "name": n} for n, i in self.labels.items()]})
        if (method, path) == ("POST", "labels"):
            self.labels[json["name"]] = f"Label_{len(self.labels)}"
            return Response(200, {"id": self.labels[json["name"]], "name": json["name"]})
        if (method, path) == ("GET", "profile"):
            return Response(200, {"historyId": str(self.history_id)})
        if (method, path) == ("GET", "messages"):
            wanted = params.get("labelIds", [])
            ids = [
                message_id for message_id, message in sorted(
                    self.messages.items(), key=lambda item: -item[1]["date"]
                )
                if set(wanted) <= message["labels"]
                and self._matches(message_id, message, params.get("q"))
            ]
            return Response(200, self._page([{"id": i} for i in ids], "messages", params))
        if (method, path) == ("GET", "history"):
            if int(params["startHistoryId"]) < self.oldest_history:
                return Response(404, {"error": {"message": "Requested entity was not found."}})
            records = [r for r in self.history if int(r["id"]) > int(params["startHistoryId"])]
            return Response(200, self._page(records, "history", params, {"historyId": str(self.history_id)}))
        if (method, path) == ("POST", "messages/batchModify"):
            for message_id in json["ids"]:
                self.relabel(message_id, json["addLabelIds"], json["removeLabelIds"])
            return Response(204)
        if (method, path) == ("POST", "messages"):
            message_id = f"m{len(self.messages) + 1}"
            self.messages[message_id] = {
                "labels": set(json["labelIds"]), "raw": base64.urlsafe_b64decode(json["raw"]),
                "thread": f"t-{message_id}", "date": NOW.timestamp(),
            }
            return Response(200, {"id": message_id})
        if (method, path) == ("POST", "drafts"):
            self.drafts.append(json["message"])
            return Response(200, {"id": "d1"})
        if method == "POST" and path.endswith("/trash"):
            self.messages[path.split("/")[1]]["labels"] = {"TRASH"}
            return Response(200, {})
        if method == "GET" and path.startswith("messages/"):
            message = self.messages.get(path.split("/")[1])
            if message is None:
                return Response(404, {"error": {"message": "Not Found"}})
            body = {"id": path.split("/")[1], "threadId": message["thread"],
                    "labelIds": sorted(message["labels"])}
            if params.get("format") == "raw":
                body["raw"] = base64.urlsafe_b64encode(message["raw"]).decode().rstrip("=")
            return Response(200, body)
        raise AssertionError(f"unexpected request {method} {path}")

    def close(self):
        pass

    def content_reads(self):
        return sum(1 for call in self.calls if call[2] == "raw")


GMAIL = f"""
folder_lists:
  logistics:
    dringend: {{folder: "{LABEL}"}}
accounts:
  - id: mail
    provider: gmail
    email: owner@gmail.test
    folder_list: logistics
    mailbox: {{inbox: INBOX, folder_root: "", trash: TRASH}}
    gmail: {{client_id: cid, client_secret: cs, refresh_token: rt}}
"""


@pytest.fixture
def account(monkeypatch):
    monkeypatch.setattr(config, "_config", config.parse_config(GMAIL))
    return config.get_config().account("mail")


@pytest.fixture
def api(account):
    fake = FakeGmail()
    fake.labels[LABEL] = "Label_1"
    return fake


def _mailbox(account, api):
    return GmailMailbox(account, session=api)


def test_a_complete_read_lists_stubs_with_read_flags_and_import_window(account, api):
    api.add("a", "INBOX")
    api.add("b", "INBOX", unread=False)
    api.add("c", "INBOX", days_old=60)

    state = _mailbox(account, api).read_folder("INBOX")

    assert (state.complete, state.cursor) == (True, "100")
    assert [(m.ref, m.provider_id, m.loaded, m.seen, m.importable) for m in state.messages] == [
        ("a", "a", False, False, True), ("b", "b", False, True, True), ("c", "c", False, False, False),
    ]
    assert api.content_reads() == 0


def test_a_stub_loads_its_content(account, api):
    api.add("a", "INBOX")

    [stub] = _mailbox(account, api).read_folder("INBOX").messages
    message = stub.load()

    assert (message.subject, message.sender, message.message_id) == ("Lieferung", "sender@test", "<a@test>")
    assert (message.ref, message.provider_id, message.loaded, message.text.strip()) == ("a", "a", True, "Hallo")


def test_changes_are_the_mails_that_got_the_label(account, api):
    api.add("a", "INBOX")
    mailbox = _mailbox(account, api)
    cursor = mailbox.read_folder("INBOX").cursor
    api.deliver("b", "INBOX")
    api.relabel("a", add=[LABEL], remove=["INBOX"])

    inbox = mailbox.read_changes("INBOX", cursor)
    filed = mailbox.read_changes(LABEL, cursor)

    assert [m.ref for m in inbox.messages] == ["b"] and [m.ref for m in filed.messages] == ["a"]
    assert inbox.cursor == filed.cursor == "102" and not inbox.complete
    assert sum(1 for call in api.calls if call[1] == "history") == 1


def test_an_expired_history_id_needs_a_complete_read(account, api):
    api.oldest_history = 200

    with pytest.raises(CursorExpired):
        _mailbox(account, api).read_changes("INBOX", "100")


def test_a_move_replaces_the_label_and_creates_a_missing_one(account, api):
    api.add("a", "INBOX")

    _mailbox(account, api).move("INBOX", ["a"], "2: Zu beantworten")

    assert api.messages["a"]["labels"] == {api.labels["2: Zu beantworten"], "UNREAD"}


def test_a_draft_joins_the_thread_of_the_mail_it_answers(account, api):
    api.add("a", "INBOX")
    raw = b"Subject: Re: Lieferung\r\nIn-Reply-To: <a@test>\r\n\r\nGuten Tag"

    _mailbox(account, api).create_draft(raw)

    [draft] = api.drafts
    assert draft["threadId"] == "t-a"
    assert message_from_bytes(base64.urlsafe_b64decode(draft["raw"]))["Subject"] == "Re: Lieferung"


def test_append_stores_an_unread_mail_in_the_folder(account, api):
    _mailbox(account, api).append(b"Subject: SMIO\r\n\r\nx", "INBOX")

    [message] = api.messages.values()
    assert message["labels"] == {"INBOX", "UNREAD"}


def test_the_refresh_token_and_client_secret_come_from_secret_manager(account, monkeypatch):
    secrets = {"gmail-client-secret": "cs", "gmail-mail-refresh-token": "rt"}
    monkeypatch.setattr(gmail_module.secret_store, "read_secret", lambda name: secrets[name])
    stored = account.model_copy(update={"gmail": account.gmail.model_copy(update={
        "client_secret": None, "client_secret_secret": "gmail-client-secret",
        "refresh_token": None, "refresh_token_secret": "gmail-mail-refresh-token",
    })})

    credentials = gmail_module.connect(stored).credentials

    assert (credentials.refresh_token, credentials.client_secret) == ("rt", "cs")
    assert credentials.scopes == ["https://www.googleapis.com/auth/gmail.modify"]


@pytest.fixture
def synced(monkeypatch, db, account, api):
    monkeypatch.setattr(mail_sync, "open_mailbox", lambda account: GmailMailbox(account, session=api))
    monkeypatch.setattr(mail_sync, "SessionLocal", lambda: db)
    monkeypatch.setattr(mail_sync, "FULL_SYNC_MINUTES", 60)
    return lambda: mail_sync.fetch_inbox(account)


def test_a_sync_imports_recent_mail_and_then_follows_the_history(db, api, synced):
    api.add("a", "INBOX")
    api.add("old", "INBOX", days_old=60)

    synced()
    api.relabel("a", add=[LABEL], remove=["INBOX"])
    api.deliver("b", "INBOX")
    synced()

    stored = {email.provider_message_id: email.folder for email in db.query(Email)}
    assert stored == {"a": LABEL, "b": "INBOX"}
    # Content only for the two new mails, not for the move or the old mail.
    assert api.content_reads() == 2
    moved = db.query(MailEvent).filter(MailEvent.event_type == "moved").one()
    assert (moved.from_folder, moved.to_folder, moved.source, moved.initial) == (
        "INBOX", LABEL, "user", False
    )


def test_an_archived_mail_is_removed_by_the_next_complete_read(db, api, account, synced):
    api.add("a", "INBOX")
    synced()
    api.relabel("a", remove=["INBOX"])

    synced()
    assert db.query(Email).one().removed_at is None
    mail_sync.sync_existing_mails(account)

    assert db.query(Email).one().removal_reason == "external"
