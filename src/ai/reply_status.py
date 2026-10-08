"""Asks Claude Haiku what a reply means for the status of the mail it answers.

Needs ANTHROPIC_API_KEY (Claude Console, billed separately from a Claude
subscription). Without it, replies wait until a key is configured.
"""

import html as html_lib
import os
import re
from typing import Literal

from pydantic import BaseModel

MODEL = os.getenv("SMIO_STATUS_MODEL", "claude-haiku-4-5")
# A decision below this confidence is kept as a suggestion only.
MIN_CONFIDENCE = float(os.getenv("SMIO_STATUS_MIN_CONFIDENCE", "0.8"))
# Roughly 1000 tokens per mail.
MAX_MAIL_CHARS = int(os.getenv("SMIO_STATUS_MAX_MAIL_CHARS", "3200"))

STATUSES = (
    "Zu erledigen",
    "Wartet: Antwort",
    "Wartet: Termin",
    "Wartet: Dokument",
    "Abgeschlossen",
)

SYSTEM_PROMPT = """You track the processing status of business e-mails of a logistics company.
You get an original mail and a later reply in the same thread. Decide the status
of the ORIGINAL mail's matter after the reply. Allowed statuses:
- "Zu erledigen": something still has to be done by the company.
- "Wartet: Antwort": the company answered and now waits for the other side to reply.
- "Wartet: Termin": an appointment or date still has to be agreed or confirmed.
- "Wartet: Dokument": a document (e.g. CMR, invoice, certificate) is still missing.
- "Abgeschlossen": the matter is settled; nothing is left to do or wait for.
A thank-you or confirmation that settles the request means "Abgeschlossen"; a
promise to check or come back later does not. Give a confidence between 0 and 1
and a one-sentence reason in German."""


class StatusDecision(BaseModel):
    status: Literal[STATUSES]
    confidence: float
    reason: str


class StatusCheckError(RuntimeError):
    def __init__(self, message, retryable):
        super().__init__(message)
        self.retryable = retryable


def enabled():
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def mail_text(email, max_chars=MAX_MAIL_CHARS):
    """Plain text of a mail, HTML stripped, whitespace collapsed, capped."""
    text = email.text
    if not (text or "").strip():
        text = re.sub(r"(?is)<(script|style)\b.*?</\1>|<[^>]+>", " ", email.html or "")
        text = html_lib.unescape(text)
    text = re.sub(r"https?://\S+", "[link]", text or "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text[:max_chars]


def _describe(label, email):
    return (
        f"<{label}>\nFrom: {email.sender or ''}\nDate: {email.date}\n"
        f"Subject: {email.subject or ''}\n\n{mail_text(email)}\n</{label}>"
    )


_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def assess_reply(original, reply, reply_is_ours):
    """Status of the original mail's matter after the reply, as a StatusDecision."""
    import anthropic

    author = "the company itself" if reply_is_ours else "the other party"
    prompt = (
        f"Current status of the original mail: {original.status or 'unknown'}\n"
        f"The reply was written by {author}.\n\n"
        f"{_describe('original_mail', original)}\n\n{_describe('reply', reply)}"
    )
    try:
        response = _get_client().messages.parse(
            model=MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=StatusDecision,
        )
    except (anthropic.RateLimitError, anthropic.APIConnectionError) as error:
        raise StatusCheckError(str(error), retryable=True) from error
    except anthropic.APIStatusError as error:
        raise StatusCheckError(str(error), retryable=error.status_code >= 500) from error

    if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
        raise StatusCheckError(f"No usable answer (stop_reason={response.stop_reason})", retryable=False)
    return response.parsed_output
