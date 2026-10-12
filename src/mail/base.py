"""Provider-neutral view of a mailbox, implemented by src/mail/imap.py and,
later, gmail.py and graph.py.

Adapters only translate: they read what is on the server as MailMessage
objects and execute commands (move, trash, append, create_draft). Deciding
what a change means (new mail, correction, removal) is left to the shared
sync code, so every provider produces the same events.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class MailMessage:
    folder: str
    # Handle for commands. IMAP: the UID, valid only in this folder and
    # changed by every move. Gmail: the message id, which never changes.
    ref: str
    # Stable across moves (Gmail message id, Exchange ImmutableId); None for IMAP.
    provider_id: Optional[str]
    message_id: str
    subject: str
    sender: str
    date: Optional[datetime]
    text: str = ""
    html: str = ""
    seen: bool = False
    # First value of each header, keyed by lowercase name.
    headers: dict = field(default_factory=dict)

    def header(self, name):
        return self.headers.get(name.lower(), "")


@dataclass
class FolderState:
    folder: str
    messages: list
    # True: every message currently in the folder, so a stored mail that is
    # missing has left it. False: only some messages (a date window or a
    # provider's change feed); a missing mail says nothing.
    complete: bool


class Mailbox(ABC):
    """One logged-in account; use as a context manager."""

    def __init__(self, account):
        self.account = account

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        pass

    @abstractmethod
    def folders(self):
        """Names of the folders (Gmail: labels) that exist."""

    @abstractmethod
    def sent_folder(self):
        """The folder with sent mail, or None."""

    @abstractmethod
    def drafts_folder(self):
        """The folder with drafts, or None."""

    @abstractmethod
    def read_folder(self, folder, since: Optional[date] = None):
        """Messages in a folder as a FolderState; complete unless since is given."""

    @abstractmethod
    def get(self, folder, ref, headers_only=True):
        """The message at ref in folder, or None if it is gone."""

    @abstractmethod
    def find(self, folder, message_id):
        """Refs of the messages in folder with this Message-ID, oldest first."""

    @abstractmethod
    def move(self, folder, refs, destination):
        """Move messages, creating the destination if needed. Refs may change."""

    @abstractmethod
    def trash(self, folder, refs):
        """Move messages to the account's trash."""

    @abstractmethod
    def append(self, raw: bytes, folder, seen=False):
        """Store a complete RFC 822 message in a folder."""

    @abstractmethod
    def create_draft(self, raw: bytes):
        """Store a message as a draft for the owner to review and send."""
