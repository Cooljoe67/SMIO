import json
from types import SimpleNamespace

import pytest

from src.ai import extraction
from src.api.utils import inbox_processor
from src.config import primary_account
from src.db.models import Email
from src.mail.imap import ImapMailbox
from tests.test_mail_identity import _message, _sync, server  # noqa: F401


class FakeClient:
    def __init__(self, parsed=None, stop_reason="end_turn", error=None):
        self.calls = []
        self.messages = SimpleNamespace(parse=self.parse)
        self.parsed, self.stop_reason, self.error = parsed, stop_reason, error

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason)


DETAILS = extraction.Extraction(
    delivery_company="DB Schenker",
    tracking_id="SCH123",
    delivery_status="IN_TRANSIT",
    delivery_date="2026-10-15",
    item_name="Kugellager 6204-2RS",
    quantity="40 Kartons",
    pallets=2,
    destination="Hamburg",
)


@pytest.fixture
def haiku(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    client = FakeClient(parsed=DETAILS)
    monkeypatch.setattr(extraction, "_get_client", lambda: client)
    return client


def test_shared_fields_get_columns_and_the_rest_goes_to_json(haiku):
    email = Email(sender="dispo@schenker.test", subject="Avis", text="Ihre Sendung SCH123")

    stored = extraction.apply(email, extraction.extract(email))

    assert (email.delivery_company, email.tracking_id, email.delivery_status) == (
        "DB Schenker", "SCH123", "IN_TRANSIT"
    )
    assert (email.delivery_date, email.item_name) == ("2026-10-15", "Kugellager 6204-2RS")
    extras = json.loads(email.extraction)
    assert (extras["quantity"], extras["pallets"], extras["destination"]) == ("40 Kartons", 2, "Hamburg")
    assert email.extracted_at is not None and stored["tracking_id"] == "SCH123"
    assert "<mail>" in haiku.calls[0]["messages"][0]["content"]


def test_no_key_or_no_usable_answer_is_an_extraction_error(monkeypatch, haiku):
    email = Email(subject="Avis", text="...")
    haiku.stop_reason = "max_tokens"
    with pytest.raises(extraction.ExtractionError):
        extraction.extract(email)

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(extraction.ExtractionError, match="ANTHROPIC_API_KEY"):
        extraction.extract(email)


def _process(db, server, monkeypatch, classification):
    mailbox = server(**{"INBOX": [_message("1", subject="Avis")], "INBOX/delivery": [], "INBOX/tech": []})
    _sync()
    monkeypatch.setattr(
        inbox_processor, "open_mailbox", lambda account: ImapMailbox(account, connection=mailbox)
    )
    monkeypatch.setattr(
        inbox_processor.classifier, "predict_email", lambda email, folder_list: (classification, 0.9)
    )
    _, results, errors = inbox_processor.process_unprocessed_emails(db, primary_account())
    assert errors == []
    return results[0], db.query(Email).one()


def test_mails_of_an_extract_folder_get_details(db, server, monkeypatch, haiku):
    result, email = _process(db, server, monkeypatch, "delivery")

    assert result["extracted"] is True
    assert (email.tracking_id, email.folder) == ("SCH123", "INBOX/delivery")


def test_other_folders_are_not_extracted(db, server, monkeypatch, haiku):
    result, email = _process(db, server, monkeypatch, "tech")

    assert result["extracted"] is False and haiku.calls == []
    assert email.tracking_id is None


def test_a_failed_extraction_still_sorts_the_mail(db, server, monkeypatch, haiku):
    haiku.parsed = None

    result, email = _process(db, server, monkeypatch, "delivery")

    assert result["extracted"] is False
    assert (email.folder, email.classification) == ("INBOX/delivery", "delivery")
