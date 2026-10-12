from datetime import date, datetime
from types import SimpleNamespace

import pytest

from src.config import primary_account
from src.mail import CursorExpired, open_mailbox
from src.mail.imap import ImapMailbox


class Connection:
    """The parts of an imap_tools MailBox the adapter uses."""

    def __init__(self, flags=None, **folders):
        self.folders = {name: list(messages) for name, messages in folders.items()}
        self.flags = flags or {}
        self.current = None
        self.readonly = None
        self.criteria = None
        self.appended = []
        self.logged_out = False
        self.uid_next = 9
        self.folder = SimpleNamespace(
            list=lambda: [
                SimpleNamespace(name=name, flags=self.flags.get(name, ()))
                for name in self.folders
            ],
            set=self._select,
            exists=lambda name: name in self.folders,
            status=lambda name, options: {"UIDVALIDITY": 3, "UIDNEXT": self.uid_next},
            create=lambda name: self.folders.setdefault(name, []),
        )

    def _select(self, name, readonly=False):
        self.current, self.readonly = name, readonly

    def fetch(self, criteria="ALL", mark_seen=True, uid_list=None, headers_only=False):
        assert mark_seen is False
        self.criteria = str(criteria)
        return iter(
            m for m in self.folders[self.current] if uid_list is None or m.uid in uid_list
        )

    def move(self, uids, destination):
        assert self.readonly is False
        moved = [m for m in self.folders[self.current] if m.uid in uids]
        for message in moved:
            self.folders[self.current].remove(message)
            self.folders[destination].append(message)

    def append(self, raw, folder="INBOX", dt=None, flag_set=None):
        self.appended.append((raw, folder, list(flag_set or ())))

    def logout(self):
        self.logged_out = True


def _msg(uid, seen=False):
    return SimpleNamespace(
        uid=uid,
        headers={"Message-ID": ("<m@test>",), "In-Reply-To": ("<x@test>",)},
        flags=(b"\\Seen",) if seen else (),
        subject="Lieferung",
        from_="sender@test",
        date=datetime(2026, 10, 7, 9, 30),
        text="Hallo",
        html="",
    )


def _mailbox(**kwargs):
    return ImapMailbox(primary_account(), connection=Connection(**kwargs))


def test_read_folder_maps_messages_and_marks_a_full_read_complete():
    mailbox = _mailbox(INBOX=[_msg("7", seen=True)])

    state = mailbox.read_folder("INBOX")

    assert state.complete and mailbox.connection.readonly is True
    [message] = state.messages
    assert (message.folder, message.ref, message.provider_id) == ("INBOX", "7", None)
    assert (message.message_id, message.sender, message.seen) == ("<m@test>", "sender@test", True)
    assert message.header("In-Reply-To") == "<x@test>"


def test_a_date_window_is_not_a_complete_folder():
    mailbox = _mailbox(Sent=[_msg("1")])

    state = mailbox.read_folder("Sent", since=date(2026, 10, 1))

    assert not state.complete
    assert "SINCE" in mailbox.connection.criteria


def test_changes_are_the_mails_from_the_cursor_on():
    mailbox = _mailbox(INBOX=[_msg("8"), _msg("9"), _msg("10")])
    mailbox.connection.uid_next = 11

    state = mailbox.read_changes("INBOX", "3:9")

    assert [m.ref for m in state.messages] == ["9", "10"]
    assert (state.complete, state.cursor) == (False, "3:11")
    assert mailbox.connection.criteria == "(UID 9:*)"


def test_without_new_uids_nothing_is_fetched():
    mailbox = _mailbox(INBOX=[_msg("8")])

    state = mailbox.read_changes("INBOX", "3:9")

    assert (state.messages, state.cursor, mailbox.connection.criteria) == ([], "3:9", None)


def test_a_renumbered_folder_expires_the_cursor():
    mailbox = _mailbox(INBOX=[_msg("8")])

    with pytest.raises(CursorExpired):
        mailbox.read_changes("INBOX", "2:9")


def test_get_returns_none_for_a_vanished_uid():
    mailbox = _mailbox(INBOX=[_msg("7")])

    assert mailbox.get("INBOX", "7").ref == "7"
    assert mailbox.get("INBOX", "8") is None


def test_trash_creates_the_trash_folder_and_moves_writable():
    mailbox = _mailbox(INBOX=[_msg("7")])

    mailbox.trash("INBOX", ["7"])

    assert mailbox.connection.folders["INBOX"] == []
    assert [m.uid for m in mailbox.connection.folders["Papierkorb"]] == ["7"]


def test_drafts_go_to_the_folder_flagged_drafts():
    mailbox = _mailbox(flags={"Entwürfe": ("\\Drafts",)}, INBOX=[], Entwürfe=[])

    mailbox.create_draft(b"raw")

    assert mailbox.connection.appended == [(b"raw", "Entwürfe", ["\\Draft", "\\Seen"])]


def test_without_a_drafts_folder_no_draft_is_written():
    mailbox = _mailbox(INBOX=[])

    with pytest.raises(RuntimeError, match="no drafts folder"):
        mailbox.create_draft(b"raw")


def test_closing_logs_out():
    with _mailbox(INBOX=[]) as mailbox:
        pass

    assert mailbox.connection.logged_out


def test_providers_without_an_adapter_are_refused():
    account = primary_account().model_copy(update={"provider": "gmail"})

    with pytest.raises(NotImplementedError, match="gmail"):
        open_mailbox(account)
