"""Mail providers behind one interface (src/mail/base.py)."""

from .base import CursorExpired, FolderState, Mailbox, MailMessage


# Providers with an adapter; others are skipped by the workflows.
PROVIDERS = ("imap", "gmail")


def open_mailbox(account) -> Mailbox:
    """Log in to an account with its provider's adapter; use as a context manager."""
    if account.provider == "imap":
        from .imap import ImapMailbox

        return ImapMailbox(account)
    if account.provider == "gmail":
        from .gmail import GmailMailbox

        return GmailMailbox(account)
    raise NotImplementedError(
        f"Account {account.id!r}: provider {account.provider!r} is not supported yet"
    )


__all__ = ["PROVIDERS", "CursorExpired", "FolderState", "Mailbox", "MailMessage", "open_mailbox"]
