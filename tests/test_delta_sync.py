import pytest

from src.api.utils import imap_client
from src.db.models import Email, MailSyncState
from tests.test_mail_identity import _message, _sync, server  # noqa: F401


@pytest.fixture
def deltas(monkeypatch):
    monkeypatch.setattr(imap_client, "FULL_SYNC_MINUTES", 60)


def test_between_full_reads_only_new_mail_is_read(db, server, deltas):
    mailbox = server(INBOX=[_message("1")])
    _sync()
    mailbox.deliver("INBOX", _message(None, message_id="<b@test>"))

    assert len(_sync()) == 1

    assert db.query(Email.message_id).order_by(Email.id).all() == [("<a@test>",), ("<b@test>",)]
    assert mailbox.criteria["INBOX"].startswith("(UID ")
    assert {state.folder for state in db.query(MailSyncState)} >= {"INBOX"}


def test_only_a_full_read_notices_a_vanished_mail(db, server, deltas):
    mailbox = server(INBOX=[_message("1"), _message("2", message_id="<b@test>")])
    _sync()
    mailbox.folders["INBOX"].pop()

    _sync()
    assert db.query(Email).filter(Email.removed_at.isnot(None)).count() == 0

    result = _sync(new=False)  # sync_existing_mails always reads in full.
    removed = db.query(Email).filter(Email.removed_at.isnot(None)).one()
    assert removed.message_id == "<b@test>" and "INBOX" in result["complete_folders"]


def test_a_user_move_is_seen_as_an_arrival_in_the_destination(db, server, deltas):
    mailbox = server(**{"INBOX": [_message("1")], "INBOX/delivery": []})
    _sync()
    mailbox.current = "INBOX"
    mailbox.move(["1"], "INBOX/delivery")

    result = _sync()

    email = db.query(Email).one()
    assert result == [] and mailbox.criteria["INBOX/delivery"].startswith("(UID ")
    assert (email.folder, email.true_label, email.classification_source) == (
        "INBOX/delivery", "delivery", "imap_folder"
    )
    assert email.removed_at is None


def test_a_renumbered_folder_is_read_completely(db, server, deltas):
    mailbox = server(INBOX=[_message("1"), _message("2", message_id="<b@test>")])
    _sync()
    mailbox.folders["INBOX"].pop()
    mailbox.renumber("INBOX")

    _sync()

    assert mailbox.criteria["INBOX"] == "ALL"
    removed = db.query(Email).filter(Email.removed_at.isnot(None)).one()
    assert removed.message_id == "<b@test>"
    assert db.query(Email).filter(Email.message_id == "<a@test>").one().uid == mailbox.folders["INBOX"][0].uid
