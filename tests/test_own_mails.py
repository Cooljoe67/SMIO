from src.api.utils import command_processor, inbox_processor, mailer
from src.config import primary_account
from src.db import database
from src.db.models import Email
from src.mail.imap import ImapMailbox
from tests.test_mail_identity import FakeServer, _message, _predict, _sync, server  # noqa: F401


def test_smio_marks_the_mails_it_writes(monkeypatch):
    appended = []

    class Mailbox(FakeServer):
        def append(self, message, folder):
            appended.append(message)

    monkeypatch.setattr(mailer, "open_mailbox", lambda account: Mailbox())

    assert mailer.append_summary_to_inbox("Hallo", kind="summary")
    assert mailer.append_summary_to_inbox("Fehler")
    assert b"\nX-SMIO-Mail: summary" in appended[0] and b"\nX-SMIO-Mail: notice" in appended[1]


def test_only_marked_mails_count_as_smios_and_the_owners_others_are_sorted(db, server, monkeypatch):
    owner = primary_account().email
    mailbox = server(**{"INBOX": [
        _message("1", message_id="<summary@test>", subject="SMIO Daily Summary", sender=owner,
                 **{"X-SMIO-Mail": "summary"}),
        _message("2", message_id="<fritz@test>", subject="Änderungsnotiz - Heimnetz", sender=owner),
    ], "INBOX/tech": []})
    _sync()
    _predict(monkeypatch, mailbox, "tech")

    inbox_processor.process_unprocessed_emails(db, primary_account())

    stored = {
        email.message_id: (email.folder, email.classification, email.classification_source)
        for email in db.query(Email)
    }
    assert stored == {
        "<summary@test>": ("INBOX", "other", "system"),
        "<fritz@test>": ("INBOX/tech", "tech", "model"),
    }


def test_a_command_mail_is_recorded_as_removed_and_never_classified(db, server, monkeypatch):
    owner = primary_account().email
    mailbox = server(INBOX=[
        _message("1", message_id="<cmd@test>", subject="SMIO Command", text="SUMMARY", sender=owner)
    ])
    _sync()
    monkeypatch.setattr(
        command_processor, "open_mailbox", lambda account: ImapMailbox(account, connection=mailbox)
    )
    monkeypatch.setattr(command_processor, "SessionLocal", lambda: db)
    monkeypatch.setattr(command_processor, "_load_command_attempts", lambda: [])
    monkeypatch.setattr(command_processor, "_save_command_attempt", lambda attempt_id, attempt: None)
    monkeypatch.setattr(command_processor, "_delete_command_attempt", lambda attempt_id: None)
    monkeypatch.setattr(command_processor, "_run_command", lambda *args: {"command": "summary"})

    command_processor.process_instruction_mails(primary_account(), is_primary=True)

    email = db.query(Email).one()
    assert (email.folder, email.uid, email.removal_reason, email.processed) == (
        "Papierkorb", None, "command", 1
    )
    assert inbox_processor.process_unprocessed_emails(db, primary_account())[1:] == ([], [])


def test_the_owners_mails_taken_for_smios_are_queued_again_once(db, monkeypatch):
    owner = primary_account().email
    taken = dict(account="primary", folder="INBOX", sender=owner, classification="other",
                 classification_source="system", processed=1)
    db.add_all([
        Email(subject="Änderungsnotiz - Heimnetz", **taken),
        Email(subject="RE: SMIO Daily Summary", **taken),
        Email(subject="FRITZ!Box-Info", **{**taken, "folder": "INBOX/tech"}),
        Email(subject="Newsletter", **{**taken, "sender": "news@test"}),
    ])
    db.commit()
    monkeypatch.setattr(database, "SessionLocal", lambda: db)

    assert database.requeue_own_address_mails() == 1
    assert database.requeue_own_address_mails() == 0

    queued = db.query(Email).filter(Email.processed == 0).one()
    assert (queued.subject, queued.classification, queued.classification_source) == (
        "Änderungsnotiz - Heimnetz", None, None
    )
