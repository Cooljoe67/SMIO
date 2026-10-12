"""Mail providers behind one interface (src/mail/base.py)."""

from .base import FolderState, Mailbox, MailMessage


def open_mailbox(account) -> Mailbox:
    """Log in to an account with its provider's adapter; use as a context manager."""
    if account.provider == "imap":
        from .imap import ImapMailbox

        return ImapMailbox(account)
    raise NotImplementedError(
        f"Account {account.id!r}: provider {account.provider!r} is not supported yet"
    )


__all__ = ["FolderState", "Mailbox", "MailMessage", "open_mailbox"]
