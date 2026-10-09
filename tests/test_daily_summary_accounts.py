from datetime import timedelta

import pytest

from src import config
from src.ai import daily_summary
from src.api.utils import command_processor
from src.db import database
from src.db.models import Email

THREE_ACCOUNTS = """
folder_lists:
  default:
    delivery: {}
    tech: {}
  small:
    delivery: {}
accounts:
  - id: first
    provider: imap
    email: first@test
    user_first_name: Anna
    folder_list: default
    imap: {host: imap.first, username: first@test, password: x}
  - id: second
    provider: imap
    email: second@test
    user_first_name: Ben
    folder_list: default
    imap: {host: imap.second, username: second@test, password: x}
  - id: third
    provider: imap
    email: third@test
    user_first_name: Cleo
    folder_list: small
    imap: {host: imap.third, username: third@test, password: x}
"""

# The SUMMARY command always reports yesterday's period.
SUMMARY_DATE, PERIOD_START, _ = daily_summary.summary_period()
PROCESSED_AT = PERIOD_START + timedelta(hours=1)
DELIVERY_DATE = (SUMMARY_DATE + timedelta(days=1)).isoformat()


@pytest.fixture
def accounts(monkeypatch):
    parsed = config.parse_config(THREE_ACCOUNTS)
    monkeypatch.setattr(config, "_config", parsed)
    # The fixture database already has every table; keep the real one untouched.
    monkeypatch.setattr(database, "ensure_retrain_run_table", lambda: None)
    return parsed.accounts


def _email(db, account, **fields):
    email = Email(
        account=account.id,
        folder_list=account.folder_list,
        processed=1,
        processed_at=PROCESSED_AT,
        **fields,
    )
    db.add(email)
    db.commit()
    return email


def _correction(db, account):
    return _email(
        db,
        account,
        classification="other",
        true_label="delivery",
        classification_source="imap_folder",
    )


@pytest.fixture
def mailboxes(db, accounts):
    first, second, _ = accounts
    _email(db, first, sender="a@first.test", classification="other")
    _email(db, first, subject="Parcel", item_name="First parcel", classification="delivery",
           delivery_date=DELIVERY_DATE, delivery_status="shipped")
    _email(db, second, sender="ceo@secret.test", classification="other")
    _email(db, second, subject="Parcel", item_name="Secret parcel", classification="delivery",
           delivery_date=DELIVERY_DATE, delivery_status="shipped")
    return accounts


def test_daily_summary_covers_only_the_primary_account(db, mailboxes):
    summary = daily_summary.gather_daily_summary(db, SUMMARY_DATE)

    assert summary["message"].startswith("Hi Anna,")
    assert summary["other_senders"] == ["a@first.test"]
    assert [item["item"] for item in summary["deliveries"]["upcoming"]] == ["First parcel"]
    assert summary["db"]["total_entries"] == 2
    assert "ceo@secret.test" not in summary["message"] + summary["html_message"]
    assert "Secret parcel" not in summary["message"] + summary["html_message"]


def test_summary_of_another_account_shows_only_its_mail(db, mailboxes):
    _, second, _ = mailboxes

    summary = daily_summary.build_daily_summary(db, SUMMARY_DATE, account=second)

    assert summary["message"].startswith("Hi Ben,")
    assert summary["other_senders"] == ["ceo@secret.test"]
    assert [item["item"] for item in summary["deliveries"]["upcoming"]] == ["Secret parcel"]
    assert "a@first.test" not in summary["message"] + summary["html_message"]


def test_summary_command_reports_the_requesting_accounts_mail(db, mailboxes, monkeypatch):
    _, second, _ = mailboxes
    appended = []
    monkeypatch.setattr(
        command_processor,
        "append_summary_to_inbox",
        lambda body, html_body=None, subject=None, account=None: appended.append((body, account)) or True,
    )

    result = command_processor._run_command(db, "summary", second)

    assert result == {"command": "summary", "sent": True}
    [(body, account)] = appended
    assert account is second
    assert body.startswith("Hi Ben,")
    assert "ceo@secret.test" in body
    assert "a@first.test" not in body
    assert "First parcel" not in body


def test_pending_corrections_count_what_retraining_counts(db, accounts, monkeypatch):
    first, second, third = accounts
    monkeypatch.setattr(daily_summary, "RETRAIN_MIN_CORRECTIONS", 2)
    # Both accounts on the default list feed the same model.
    _correction(db, first)
    _correction(db, second)
    for _ in range(3):
        _correction(db, third)

    first_summary = daily_summary.build_daily_summary(db, SUMMARY_DATE, account=first)
    third_summary = daily_summary.build_daily_summary(db, SUMMARY_DATE, account=third)

    assert first_summary["db"]["pending_corrections"] == 2
    assert first_summary["db"]["retrain_due"] is True
    # Only the default list's model is retrained, so the small list is never due.
    assert third_summary["db"]["pending_corrections"] == 3
    assert third_summary["db"]["retrain_due"] is False
    assert "RETRAINING DUE" not in third_summary["message"]
    assert set(third_summary["new_counts"]) == {"delivery", "other"}
