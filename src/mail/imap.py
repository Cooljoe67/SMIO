"""IMAP adapter (imap_tools)."""

from imap_tools import AND, H, U, MailBox, MailboxLoginError, MailMessageFlags

from src.storage import secret_store

from .base import CursorExpired, FolderState, Mailbox, MailMessage


def connect(account):
    """Log in to an account's IMAP server; returns the imap_tools MailBox."""
    if account.unavailable:
        raise RuntimeError(f"Account {account.id!r} is unavailable: {account.unavailable}")
    if account.provider != "imap":
        raise NotImplementedError(
            f"Account {account.id!r}: provider {account.provider!r} is not supported yet"
        )
    imap = account.imap
    if imap.password_secret is None:
        if not imap.password:
            raise RuntimeError(
                f"Account {account.id!r}: no IMAP password; set the variable its config refers to"
            )
        return MailBox(imap.host).login(imap.username, imap.password)

    password = secret_store.read_secret(imap.password_secret)
    try:
        return MailBox(imap.host).login(imap.username, password)
    except MailboxLoginError:
        # The password may have been rotated since it was cached.
        secret_store.invalidate(imap.password_secret)
        current = secret_store.read_secret(imap.password_secret)
        if current == password:
            raise
        return MailBox(imap.host).login(imap.username, current)


def _first(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "").strip()


def _flag_names(flags):
    return {
        (flag.decode(errors="ignore") if isinstance(flag, bytes) else str(flag)).lower()
        for flag in flags or ()
    }


def to_mail_message(msg, folder):
    headers = {
        str(name).lower(): _first(value)
        for name, value in (getattr(msg, "headers", {}) or {}).items()
    }
    return MailMessage(
        folder=folder,
        ref=str(msg.uid),
        provider_id=None,
        message_id=headers.get("message-id", ""),
        subject=msg.subject,
        sender=msg.from_,
        date=msg.date,
        text=msg.text,
        html=msg.html,
        seen="\\seen" in _flag_names(getattr(msg, "flags", ())),
        headers=headers,
    )


class ImapMailbox(Mailbox):
    def __init__(self, account, connection=None):
        super().__init__(account)
        self.connection = connection if connection is not None else connect(account)

    def close(self):
        self.connection.logout()

    def _select(self, folder, readonly):
        self.connection.folder.set(folder, readonly=readonly)

    def _flagged(self, flag):
        for folder in self.connection.folder.list():
            if flag in (getattr(folder, "flags", ()) or ()):
                return folder.name
        return None

    def folders(self):
        return [folder.name for folder in self.connection.folder.list()]

    def sent_folder(self):
        return self.account.mailbox.sent or self._flagged("\\Sent")

    def drafts_folder(self):
        return self.account.mailbox.drafts or self._flagged("\\Drafts")

    def _cursor(self, folder):
        """UIDVALIDITY and UIDNEXT: every later arrival gets a UID >= UIDNEXT.

        Taken before fetching, so a mail arriving meanwhile is read again next
        time rather than missed.
        """
        status = self.connection.folder.status(folder, ["UIDVALIDITY", "UIDNEXT"])
        return f"{status['UIDVALIDITY']}:{status['UIDNEXT']}"

    def read_folder(self, folder, since=None):
        cursor = self._cursor(folder)
        self._select(folder, readonly=True)
        criteria = AND(date_gte=since) if since else "ALL"
        messages = [
            to_mail_message(msg, folder)
            for msg in self.connection.fetch(criteria, mark_seen=False)
        ]
        return FolderState(
            folder=folder, messages=messages, complete=since is None, cursor=cursor
        )

    def read_changes(self, folder, cursor):
        validity, _, next_uid = (cursor or "").partition(":")
        current = self._cursor(folder)
        if not next_uid.isdigit() or current.partition(":")[0] != validity:
            raise CursorExpired(f"{folder}: UIDVALIDITY changed")
        messages = []
        if current != cursor:
            self._select(folder, readonly=True)
            # "N:*" also matches the highest UID when it is below N.
            messages = [
                to_mail_message(msg, folder)
                for msg in self.connection.fetch(
                    AND(uid=U(next_uid, "*")), mark_seen=False
                )
                if int(msg.uid) >= int(next_uid)
            ]
        return FolderState(folder=folder, messages=messages, complete=False, cursor=current)

    def get(self, folder, ref, headers_only=True):
        self._select(folder, readonly=True)
        msg = next(
            self.connection.fetch(uid_list=[ref], mark_seen=False, headers_only=headers_only),
            None,
        )
        return None if msg is None else to_mail_message(msg, folder)

    def find(self, folder, message_id):
        self._select(folder, readonly=True)
        return list(self.connection.uids(AND(header=H("Message-ID", message_id))))

    def _ensure(self, folder):
        if not self.connection.folder.exists(folder):
            self.connection.folder.create(folder)

    def move(self, folder, refs, destination):
        # The destination assigns new UIDs; the next sync reads them.
        self._ensure(destination)
        self._select(folder, readonly=False)
        self.connection.move(list(refs), destination)

    def trash(self, folder, refs):
        self.move(folder, refs, self.account.mailbox.trash)

    def append(self, raw, folder, seen=False):
        flags = [MailMessageFlags.SEEN] if seen else None
        self.connection.append(raw, folder=folder, flag_set=flags)

    def create_draft(self, raw):
        drafts = self.drafts_folder()
        if drafts is None:
            raise RuntimeError(f"Account {self.account.id!r} has no drafts folder")
        self.connection.append(
            raw, folder=drafts, flag_set=[MailMessageFlags.DRAFT, MailMessageFlags.SEEN]
        )
