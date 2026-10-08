from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.ai import reply_status
from src.api.utils import inbox_processor, reply_tracking
from src.config import primary_account
from src.db.models import Email
from tests.test_mail_identity import SENT, _message, _sync, server  # noqa: F401

OWNER = primary_account().email


def _received(db, message_id, subject="Anfrage Transport", date=SENT, **fields):
    email = Email(
        account="primary", message_id=message_id, subject=subject, sender="kunde@firma.de",
        date=date, text="Bitte um Angebot", folder="INBOX", processed=1, **fields,
    )
    db.add(email)
    db.commit()
    return email


def _reply(db, message_id, in_reply_to=None, references=None, outbound=True, date=SENT + timedelta(hours=1), **fields):
    email = Email(
        account="primary", message_id=message_id, subject="AW: Anfrage Transport",
        sender=OWNER if outbound else "kunde@firma.de", date=date, text="Danke, erledigt.",
        direction="OUTBOUND" if outbound else None, in_reply_to=in_reply_to,
        reference_ids=references, processed=1, **fields,
    )
    db.add(email)
    db.commit()
    return email


@pytest.fixture
def haiku(monkeypatch):
    """Replace the Claude call; records the arguments and returns a fixed decision."""
    calls = []
    answer = {"decision": reply_status.StatusDecision(status="Abgeschlossen", confidence=0.95, reason="Erledigt.")}

    def assess(original, reply, reply_is_ours):
        calls.append((original.id, reply.id, reply_is_ours))
        if isinstance(answer["decision"], Exception):
            raise answer["decision"]
        return answer["decision"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(reply_status, "assess_reply", assess)
    return SimpleNamespace(calls=calls, answer=answer)


# ---------- Sent folder ----------

def test_sent_folder_is_detected_and_its_mails_stored_as_outbound(db, server, monkeypatch):
    sent = _message("5", message_id="<reply@test>", sender=OWNER, **{"In-Reply-To": "<orig@test>"})
    mailbox = server(flags={"Gesendet": ("\\Sent",)}, INBOX=[_message("1", message_id="<orig@test>")], Gesendet=[sent])

    _sync()

    reply = db.query(Email).filter(Email.message_id == "<reply@test>").one()
    assert (reply.folder, reply.direction, reply.processed) == ("Gesendet", "OUTBOUND", 1)
    assert reply.in_reply_to == "<orig@test>"
    assert "SINCE" in mailbox.criteria["Gesendet"]
    # Sent mails are never classified or moved.
    monkeypatch.setattr(inbox_processor, "open_mailbox", lambda account: mailbox)
    _, results, _ = inbox_processor.process_unprocessed_emails(db, primary_account())
    assert [r["email_id"] for r in results] == [db.query(Email).filter(Email.message_id == "<orig@test>").one().id]


def test_mail_sent_to_oneself_keeps_its_inbox_copy(db, server):
    server(flags={"Sent": ("\\Sent",)}, INBOX=[_message("1")], Sent=[_message("7")])

    _sync()
    _sync()

    email = db.query(Email).one()
    assert email.folder == "INBOX" and email.direction is None


def test_references_are_stored_in_order(db, server):
    server(INBOX=[_message("1", References="<r1@test>\r\n <r2@test>", **{"In-Reply-To": "<r2@test>"})])

    _sync()

    email = db.query(Email).one()
    assert (email.in_reply_to, email.reference_ids) == ("<r2@test>", "<r1@test> <r2@test>")


# ---------- Linking replies ----------

def test_our_reply_updates_the_mail_it_answers(db, haiku):
    original = _received(db, "<orig@test>", status="Zu erledigen")
    reply = _reply(db, "<r@test>", in_reply_to="<orig@test>")

    result = reply_tracking.update_statuses_from_replies(db, primary_account())

    assert result["updated"] == 1
    assert haiku.calls == [(original.id, reply.id, True)]
    assert (original.status, original.status_source, original.status_reason) == ("Abgeschlossen", "reply", "Erledigt.")
    assert reply.reply_to_id == original.id and reply.reply_checked_at is not None


def test_their_answer_to_our_reply_updates_the_original_mail(db, haiku):
    original = _received(db, "<orig@test>")
    ours = _reply(db, "<ours@test>", in_reply_to="<orig@test>", references="<orig@test>")
    ours.reply_checked_at = datetime(2026, 10, 7)
    db.commit()
    theirs = _reply(
        db, "<theirs@test>", in_reply_to="<ours@test>", references="<orig@test> <ours@test>",
        outbound=False, date=SENT + timedelta(hours=2),
    )

    reply_tracking.update_statuses_from_replies(db, primary_account())

    assert haiku.calls == [(original.id, theirs.id, False)]


def test_case_reference_links_mails_without_thread_headers(db, haiku):
    original = _received(db, "<orig@test>", case_ref="VG:NOR-1")
    later = _reply(db, "<later@test>", outbound=False, case_ref="VG:NOR-1")

    reply_tracking.update_statuses_from_replies(db, primary_account())

    assert haiku.calls == [(original.id, later.id, False)]


def test_replies_to_smio_mails_are_not_checked(db, haiku):
    summary = _received(db, "<summary@test>", classification_source="system")
    summary.sender = OWNER
    db.commit()
    command = _reply(db, "<cmd@test>", in_reply_to="<summary@test>")

    result = reply_tracking.update_statuses_from_replies(db, primary_account())

    assert result["unlinked"] == 1 and haiku.calls == []
    assert command.reply_checked_at is not None


# ---------- Decisions ----------

def test_low_confidence_only_leaves_a_suggestion(db, haiku):
    haiku.answer["decision"] = reply_status.StatusDecision(status="Abgeschlossen", confidence=0.5, reason="Unklar.")
    original = _received(db, "<orig@test>", status="Zu erledigen", status_source="model")
    _reply(db, "<r@test>", in_reply_to="<orig@test>")

    assert reply_tracking.update_statuses_from_replies(db, primary_account())["suggested"] == 1
    assert (original.status, original.status_suggestion) == ("Zu erledigen", "Abgeschlossen")
    assert "0.50" in original.status_reason


def test_a_status_set_by_the_user_is_never_overwritten(db, haiku):
    original = _received(db, "<orig@test>", status="Wartet: Termin", status_source="command")
    _reply(db, "<r@test>", in_reply_to="<orig@test>")

    reply_tracking.update_statuses_from_replies(db, primary_account())

    assert (original.status, original.status_suggestion) == ("Wartet: Termin", "Abgeschlossen")


def test_without_api_key_replies_wait(db, haiku, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    _received(db, "<orig@test>")
    reply = _reply(db, "<r@test>", in_reply_to="<orig@test>")

    assert reply_tracking.update_statuses_from_replies(db, primary_account())["waiting"] == 1
    assert reply.reply_checked_at is None and haiku.calls == []


def test_retryable_errors_stop_and_retry_later(db, haiku):
    haiku.answer["decision"] = reply_status.StatusCheckError("rate limited", retryable=True)
    _received(db, "<orig@test>")
    first = _reply(db, "<r1@test>", in_reply_to="<orig@test>")
    _reply(db, "<r2@test>", in_reply_to="<orig@test>", date=SENT + timedelta(hours=2))

    result = reply_tracking.update_statuses_from_replies(db, primary_account())

    assert result["failed"] == 1 and len(haiku.calls) == 1
    assert first.reply_checked_at is None


def test_permanent_errors_are_not_retried(db, haiku):
    haiku.answer["decision"] = reply_status.StatusCheckError("bad request", retryable=False)
    _received(db, "<orig@test>")
    reply = _reply(db, "<r@test>", in_reply_to="<orig@test>")

    reply_tracking.update_statuses_from_replies(db, primary_account())

    assert reply.reply_checked_at is not None


# ---------- Claude call ----------

class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("anthropic")

    def use(response):
        fake = SimpleNamespace(messages=FakeMessages(response))
        monkeypatch.setattr(reply_status, "_get_client", lambda: fake)
        return fake.messages
    return use


def test_assess_reply_asks_haiku_with_both_mails(client):
    decision = reply_status.StatusDecision(status="Wartet: Dokument", confidence=0.9, reason="CMR fehlt.")
    messages = client(SimpleNamespace(stop_reason="end_turn", parsed_output=decision))
    original = Email(subject="Transport", sender="k@f.de", text="Bitte um Abholung", status="Zu erledigen")
    reply = Email(subject="AW: Transport", sender=OWNER, text="", html="<p>Bitte <b>CMR</b> senden</p>")

    assert reply_status.assess_reply(original, reply, reply_is_ours=True) == decision
    assert messages.kwargs["model"] == "claude-haiku-4-5"
    assert messages.kwargs["output_format"] is reply_status.StatusDecision
    prompt = messages.kwargs["messages"][0]["content"]
    assert "Zu erledigen" in prompt and "the company itself" in prompt
    assert "Bitte CMR senden" in prompt and "<b>" not in prompt


def test_refusal_is_a_permanent_error(client):
    client(SimpleNamespace(stop_reason="refusal", parsed_output=None))

    with pytest.raises(reply_status.StatusCheckError) as error:
        reply_status.assess_reply(Email(), Email(), reply_is_ours=False)
    assert error.value.retryable is False


def test_mail_text_is_capped():
    assert len(reply_status.mail_text(Email(text="x " * 5000), max_chars=100)) == 100
