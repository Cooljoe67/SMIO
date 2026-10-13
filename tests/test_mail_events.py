import pytest

from src import config
from src.api.utils import inbox_processor
from src.api.utils.folder_rules import classification_folders, classification_for_folder
from src.config import primary_account
from src.db.models import Email, MailEvent, MailEventAction
from src.mail import events
from tests.test_mail_identity import _message, _predict, _sync, server  # noqa: F401


def _events(db):
    return [
        (e.event_type, e.source, e.from_folder, e.to_folder)
        for e in db.query(MailEvent).order_by(MailEvent.id)
    ]


def test_a_new_mail_is_received_and_a_read_one_also_read(db, server):
    server(INBOX=[_message("1"), _message("2", message_id="<b@test>", seen=True)])

    _sync()

    assert _events(db) == [
        ("received", "user", None, "INBOX"),
        ("received", "user", None, "INBOX"),
        ("read", "user", None, "INBOX"),
    ]
    assert [e.event_key for e in db.query(MailEvent).order_by(MailEvent.id)] == [
        "primary:1:1", "primary:2:1", "primary:2:2",
    ]


def test_a_user_move_is_one_moved_event_and_a_correction(db, server):
    mailbox = server(**{"INBOX": [_message("1")], "INBOX/delivery": []})
    _sync()
    mailbox.current = "INBOX"
    mailbox.move(["1"], "INBOX/delivery")

    _sync()
    _sync()

    assert _events(db)[1:] == [("moved", "user", "INBOX", "INBOX/delivery")]
    assert db.query(Email).one().true_label == "delivery"


def test_smios_own_move_is_a_system_event_and_its_echo_no_event(db, server, monkeypatch):
    mailbox = server(**{"INBOX": [_message("1")], "INBOX/tech": []})
    _sync()
    _predict(monkeypatch, mailbox, "tech")

    inbox_processor.process_unprocessed_emails(db, primary_account())
    _sync()

    assert _events(db)[1:] == [("moved", "system", "INBOX", "INBOX/tech")]
    email = db.query(Email).one()
    assert (email.folder, email.true_label) == ("INBOX/tech", None)


def test_a_vanished_mail_is_removed_once(db, server):
    mailbox = server(INBOX=[_message("1"), _message("2", message_id="<b@test>")])
    _sync()
    mailbox.folders["INBOX"].pop()

    _sync()
    removed_at = db.query(Email).filter(Email.message_id == "<b@test>").one().removed_at
    _sync()

    assert _events(db)[2:] == [("removed", "user", "INBOX", None)]
    assert db.query(Email).filter(Email.message_id == "<b@test>").one().removed_at == removed_at


def test_mail_from_the_sent_folder_is_sent(db, server):
    server(flags={"Sent": ("\\Sent",)}, INBOX=[], Sent=[_message("7")])

    _sync()

    assert _events(db) == [("sent", "user", None, "Sent")]


MANUAL = """
folder_lists:
  default:
    delivery: {}
    pending: {folder: "5: Antwort ausstehend", manual: true}
accounts:
  - id: primary
    provider: imap
    email: owner@test
    folder_list: default
    imap: {host: imap.test, username: owner@test, password: x}
"""


@pytest.fixture
def manual_folder(monkeypatch):
    monkeypatch.setattr(config, "_config", config.parse_config(MANUAL))


def test_a_manual_folder_is_watched_but_no_label(db, server, manual_folder):
    mailbox = server(**{"INBOX": [_message("1")], "INBOX/delivery": [], "INBOX/5: Antwort ausstehend": []})
    _sync()
    mailbox.current = "INBOX"
    mailbox.move(["1"], "INBOX/5: Antwort ausstehend")

    _sync()

    assert _events(db)[1:] == [("moved", "user", "INBOX", "INBOX/5: Antwort ausstehend")]
    email = db.query(Email).one()
    assert (email.classification, email.true_label, email.processed) == (None, None, 0)
    assert config.get_config().labels("default") == {"delivery", "other"}
    assert classification_folders() == {"delivery"}
    assert classification_for_folder("INBOX/5: Antwort ausstehend") is None


def test_a_list_of_manual_folders_only_is_refused():
    with pytest.raises(ValueError, match="no folders the model predicts"):
        config.parse_config(MANUAL.replace("    delivery: {}\n", ""))


@pytest.fixture
def actions(monkeypatch):
    registered = {}
    monkeypatch.setattr(events, "_ACTIONS", registered)
    return registered


def _stored_event(db, event_type="moved"):
    email = Email(account="primary", message_id="<a@test>", folder="INBOX/delivery")
    db.add(email)
    db.flush()
    event = events.record(
        db, primary_account(), email, event_type,
        from_folder="INBOX", to_folder="INBOX/delivery", source="system",
    )
    db.commit()
    return event


def test_an_action_runs_once_per_event(db, actions):
    runs = []
    events.action("note", applies=lambda db, e: e.event_type == "moved")(
        lambda db, e: runs.append(e.event_key)
    )
    event = _stored_event(db)
    _stored_event(db, "read")

    assert events.dispatch(db) == {"events": 2, "done": 1, "skipped": 0, "failed": 0}
    assert events.dispatch(db)["events"] == 0
    assert runs == [event.event_key]


def test_a_failing_action_is_retried_until_it_gives_up(db, actions):
    def fail(db, event):
        raise RuntimeError("ntfy down")

    events.action("notify", applies=lambda db, e: True)(fail)
    event = _stored_event(db)

    for _ in range(events.MAX_ATTEMPTS):
        assert events.dispatch(db)["failed"] == 1
    assert events.dispatch(db)["events"] == 0
    run = db.query(MailEventAction).one()
    assert (run.status, run.attempts, run.error) == ("failed", 3, "ntfy down")
    assert db.get(MailEvent, event.id).dispatched_at is not None


def test_an_interrupted_action_is_never_started_again(db, actions):
    runs = []
    events.action("draft", applies=lambda db, e: True)(lambda db, e: runs.append(e.id))
    event = _stored_event(db)
    # The process died after claiming the run.
    db.add(MailEventAction(
        event_id=event.id, email_id=event.email_id, action="draft", status="running", attempts=1
    ))
    db.commit()

    events.dispatch(db)

    assert runs == []
    assert db.get(MailEvent, event.id).dispatched_at is not None


def test_a_retry_never_repeats_an_action_another_event_of_the_mail_completed(db, actions):
    calls = []

    def draft(db, event):
        calls.append(event.id)
        if len(calls) == 1:
            raise RuntimeError("overloaded")

    events.action("draft", applies=lambda db, e: True)(draft)
    first = _stored_event(db, "received")
    second = events.record(
        db, primary_account(), db.get(Email, first.email_id), "moved",
        from_folder="INBOX", to_folder="INBOX/delivery", source="system",
    )
    db.commit()

    events.dispatch(db)  # the first event fails; the second waits for the next call
    events.dispatch(db)  # the retry succeeds
    events.dispatch(db)

    assert calls == [first.id, first.id]
    run = db.query(MailEventAction).one()
    assert (run.status, run.event_id) == ("done", first.id)
    assert db.get(MailEvent, second.id).dispatched_at is not None


def test_events_have_a_provider_neutral_format(db):
    event = _stored_event(db)

    assert events.as_dict(event) | {"occurredAt": None} == {
        "eventKey": "primary:1:1",
        "eventType": "moved",
        "provider": "imap",
        "account": "primary",
        "emailId": 1,
        "messageId": "<a@test>",
        "fromFolder": "INBOX",
        "toFolder": "INBOX/delivery",
        "source": "system",
        "occurredAt": None,
    }
