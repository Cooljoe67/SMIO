from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.utils import imap_client, inbox_processor, mailer
from src.api.utils.mail_identity import email_fingerprint, fingerprint, normalized_date
from src.config import primary_account
from src.db import database
from src.db.models import Base, Email

SENT = datetime(2026, 10, 7, 9, 30, tzinfo=timezone(timedelta(hours=2)))


def _message(
    uid, message_id="<a@test>", subject="Lieferung", date=SENT, text="Hallo", seen=False,
    sender="Sender <sender@test>", **headers
):
    if message_id:
        headers["Message-ID"] = message_id
    return SimpleNamespace(
        uid=uid,
        headers=headers,
        flags=("\\Seen",) if seen else (),
        subject=subject,
        from_=sender,
        date=date,
        text=text,
        html="",
    )


class FakeServer:
    """Folders of messages; a move gives the message a new UID, like IMAP."""

    def __init__(self, flags=None, **folders):
        self.folders = {name: list(messages) for name, messages in folders.items()}
        self.flags = flags or {}
        self.current = "INBOX"
        self.next_uid = 500
        self.moves = []
        self.criteria = {}
        self.folder = SimpleNamespace(
            list=lambda: [
                SimpleNamespace(name=name, flags=self.flags.get(name, ()))
                for name in self.folders
            ],
            set=self._select,
            exists=lambda name: name in self.folders,
            create=lambda name: self.folders.setdefault(name, []),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _select(self, name, readonly=False):
        self.current = name

    def fetch(self, criteria="ALL", mark_seen=False, uid_list=None, headers_only=False):
        self.criteria[self.current] = str(criteria)
        messages = self.folders.get(self.current, [])
        return iter([m for m in messages if uid_list is None or m.uid in uid_list])

    def move(self, uids, destination):
        uids = [uids] if isinstance(uids, str) else list(uids)
        for message in [m for m in self.folders[self.current] if m.uid in uids]:
            self.folders[self.current].remove(message)
            self.next_uid += 1
            message.uid = str(self.next_uid)
            self.folders.setdefault(destination, []).append(message)
            self.moves.append((self.current, destination))

    def renumber(self, folder):
        """What a server does when UIDVALIDITY changes."""
        for message in self.folders[folder]:
            self.next_uid += 1
            message.uid = str(self.next_uid)


@pytest.fixture
def server(monkeypatch, db):
    holder = {}

    def use(flags=None, **folders):
        holder["server"] = FakeServer(flags, **folders)
        return holder["server"]

    monkeypatch.setattr(imap_client, "open_mailbox", lambda account: holder["server"])
    monkeypatch.setattr(imap_client, "SessionLocal", lambda: db)
    return use


def _sync(new=True):
    account = primary_account()
    if new:
        return imap_client.fetch_inbox(account)
    return imap_client.sync_existing_mails(account)


def test_fingerprint_ignores_the_folder_and_uses_the_body_only_without_message_id():
    base = ("primary", "Sender <SENDER@test>", "Lieferung  heute", SENT)

    assert fingerprint(*base, "<a@test>", "text") == fingerprint(*base, "<a@test>", "other text")
    assert fingerprint(*base, "", "text") != fingerprint(*base, "", "other text")
    assert fingerprint(*base, "<a@test>") != fingerprint(*base, "<b@test>")


def test_dates_compare_as_stored():
    assert normalized_date(SENT) == normalized_date(SENT.replace(tzinfo=None)) == "2026-10-07T09:30:00"
    assert normalized_date(datetime(1900, 1, 1)) == ""


def test_backfilled_fingerprint_matches_a_later_sync(tmp_path, monkeypatch, server):
    """The DB round trip (naive date, stored text) must give the same fingerprint."""
    engine = create_engine(f"sqlite:///{(tmp_path / 'smio.db').as_posix()}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(database, "SessionLocal", Session)
    message = _message("1", message_id="")
    with Session() as session:
        session.add(Email(
            account="primary", uid="1", folder="INBOX", message_id="",
            subject=message.subject, sender=message.from_, date=message.date, text=message.text,
        ))
        session.commit()

    assert database.backfill_fingerprints() == 1
    with Session() as session:
        stored = session.query(Email).one()
        assert stored.fingerprint == imap_client._identity(primary_account(), message)[1]
        assert stored.fingerprint == email_fingerprint(stored)
    engine.dispose()


def test_move_without_message_id_is_recognised_as_a_correction(db, server):
    mail = _message("1", message_id="")
    mailbox = server(**{"INBOX": [mail], "INBOX/delivery": []})
    assert len(_sync()) == 1
    db.query(Email).one().classification = "tech"
    db.commit()

    mailbox.move(["1"], "INBOX/delivery")
    _sync()

    email = db.query(Email).one()
    assert email.folder == "INBOX/delivery"
    assert email.uid == mail.uid
    assert (email.true_label, email.classification_source) == ("delivery", "imap_folder")
    assert email.removed_at is None


def _predict(monkeypatch, mailbox, classification):
    monkeypatch.setattr(inbox_processor, "open_mailbox", lambda account: mailbox)
    monkeypatch.setattr(
        inbox_processor.classifier, "predict_text", lambda text, folder_list: (classification, 0.9)
    )


def test_mail_filed_by_the_user_before_processing_keeps_the_users_label(db, server, monkeypatch):
    mailbox = server(**{"INBOX": [_message("1")], "INBOX/delivery": [], "INBOX/tech": []})
    _sync()  # Stays unprocessed, e.g. while the account has no model yet.
    mailbox.move(["1"], "INBOX/delivery")
    _sync()
    _predict(monkeypatch, mailbox, "tech")

    _, results, errors = inbox_processor.process_unprocessed_emails(db, primary_account())

    email = db.query(Email).one()
    assert (results, errors) == ([], [])
    assert mailbox.moves == [("INBOX", "INBOX/delivery")]
    assert (email.folder, email.classification, email.true_label, email.classification_source) == (
        "INBOX/delivery", "delivery", "delivery", "imap_folder"
    )
    assert (email.processed, email.processing_batch) == (1, None)
    assert email.processed_at is not None


def test_processing_only_takes_mails_still_in_the_inbox(db, server, monkeypatch):
    mailbox = server(**{
        "INBOX": [_message("1")],
        "INBOX/delivery": [_message("3", message_id="<filed@test>")],
        "INBOX/tech": [],
    })
    inbox_mail = Email(account="primary", uid="1", message_id="<a@test>", folder="INBOX")
    db.add_all([
        inbox_mail,
        Email(account="primary", uid="2", message_id="<gone@test>", folder="INBOX", removed_at=datetime(2026, 10, 8)),
        Email(account="primary", uid="3", message_id="<filed@test>", folder="INBOX/delivery"),
    ])
    db.commit()
    _predict(monkeypatch, mailbox, "tech")

    _, results, errors = inbox_processor.process_unprocessed_emails(db, primary_account())

    assert [result["email_id"] for result in results] == [inbox_mail.id]
    assert errors == []
    assert mailbox.moves == [("INBOX", "INBOX/tech")]


def test_mail_already_in_its_folder_is_not_moved(db, server):
    mailbox = server(**{"INBOX": [], "INBOX/tech": [_message("1")]})
    email = Email(id=1, account="primary", uid="1", message_id="<a@test>", folder="INBOX/tech", classification="tech")

    assert imap_client.move_email_to_classification_folder(mailbox, email, primary_account()) is None
    assert mailbox.moves == []
    assert email.uid == "1"


def test_copy_in_inbox_and_folder_keeps_the_folder(db, server):
    server(**{
        "INBOX": [_message("1")],
        "INBOX/delivery": [_message("9")],
    })

    _sync()
    _sync()

    email = db.query(Email).one()
    assert email.folder == "INBOX/delivery"
    # Filed before SMIO processed it: the folder is the label, not "other" from the inbox copy.
    assert email.true_label == "delivery"
    assert email.removed_at is None


def test_sync_without_new_mail_marks_only_vanished_mails_removed(db, server):
    mailbox = server(INBOX=[_message("1"), _message("2", message_id="<b@test>")])
    assert len(_sync()) == 2
    mailbox.folders["INBOX"].pop()

    assert _sync() == []

    kept, gone = db.query(Email).order_by(Email.id).all()
    assert (kept.removed_at, kept.removal_reason) == (None, None)
    assert gone.removed_at is not None and gone.removal_reason == "external"


def test_mail_wrongly_marked_external_is_restored_when_seen_again(db, server):
    server(**{"INBOX": [], "INBOX/tech": [_message("1")]})
    db.add(Email(
        account="primary", uid="1", folder="INBOX/tech", message_id="<a@test>",
        subject="Lieferung", date=SENT, classification="tech", processed=1, processing_batch=1,
        removed_at=datetime(2026, 10, 8), removal_reason="external", last_seen_at=datetime(2026, 10, 8),
    ))
    db.commit()

    _sync()

    email = db.query(Email).one()
    assert (email.removed_at, email.removal_reason) == (None, None)


def test_reused_message_id_with_other_content_is_a_separate_mail(db, server):
    server(INBOX=[
        _message("1", message_id="<same@test>"),
        _message("2", message_id="<same@test>", subject="Andere Mail", date=SENT + timedelta(days=3)),
    ])

    assert len(_sync()) == 2
    assert db.query(Email).count() == 2


def test_the_same_message_twice_in_the_inbox_is_stored_once(db, server):
    server(INBOX=[_message("1"), _message("2")])

    assert len(_sync()) == 1


def test_retention_does_not_trash_a_renumbered_uid(db, server, monkeypatch):
    old = _message("1", subject="Alt", date=datetime(2026, 1, 1), seen=True)
    mailbox = server(**{"INBOX": [], "INBOX/tech": [old]})
    db.add(Email(
        account="primary", uid="1", folder="INBOX/tech", message_id="<a@test>",
        subject="Alt", sender=old.from_, date=old.date, classification="tech",
    ))
    db.commit()
    email = db.query(Email).one()
    # After renumbering, UID 1 belongs to a different, newer mail.
    mailbox.renumber("INBOX/tech")
    mailbox.folders["INBOX/tech"].append(_message("1", message_id="<new@test>", seen=True))

    result = imap_client.cleanup_expired_mails(db, mailbox, primary_account())

    assert result["moved"] == 0
    assert "another mail" in result["errors"][0]["error"]
    assert mailbox.moves == []
    assert email.removed_at is None


def test_retention_trashes_a_verified_mail_and_forgets_its_uid(db, server):
    old = _message("1", date=datetime(2026, 1, 1), seen=True)
    mailbox = server(**{"INBOX": [], "INBOX/tech": [old]})
    db.add(Email(
        account="primary", uid="1", folder="INBOX/tech", message_id="<a@test>",
        subject=old.subject, sender=old.from_, date=old.date, classification="tech",
    ))
    db.commit()

    result = imap_client.cleanup_expired_mails(db, mailbox, primary_account())

    email = db.query(Email).one()
    assert result["moved"] == 1
    assert (email.folder, email.uid, email.removal_reason) == ("Papierkorb", None, "retention")


def test_smio_mails_get_a_message_id_and_date(monkeypatch):
    appended = []

    class Mailbox(FakeServer):
        def append(self, message, folder):
            appended.append(message)

    monkeypatch.setattr(mailer, "open_mailbox", lambda account: Mailbox())

    assert mailer.append_summary_to_inbox("Hallo")
    raw = appended[0].decode()
    assert "Message-ID: <" in raw and "@test>" in raw
    assert "\nDate: " in raw
