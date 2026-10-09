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


ANSWER = '{"status": "Abgeschlossen", "confidence": 0.95, "reason": "Erledigt."}'


def _answer(stop_reason, text):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": reply_status.MODEL,
        "content": [] if text is None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _api_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


@pytest.fixture
def api(monkeypatch):
    """The real SDK client; its HTTP transport serves the given responses in order, repeating the last."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")  # the SDK's HTTP library

    def use(*responses):
        queue = list(responses)
        requests = []

        def handler(request):
            requests.append(request)
            response = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(response, Exception):
                raise response
            status, body = response
            return httpx2.Response(status, json=body)

        http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
        sdk = anthropic.Anthropic(api_key="test", http_client=http_client, max_retries=0)
        monkeypatch.setattr(reply_status, "_client", sdk)
        return requests
    return use


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
    haiku.answer["decision"] = reply_status.StatusCheckError("rate limited", "retryable")
    _received(db, "<orig@test>")
    first = _reply(db, "<r1@test>", in_reply_to="<orig@test>")
    _reply(db, "<r2@test>", in_reply_to="<orig@test>", date=SENT + timedelta(hours=2))

    result = reply_tracking.update_statuses_from_replies(db, primary_account())

    assert result["failed"] == 1 and len(haiku.calls) == 1
    assert first.reply_checked_at is None


def test_permanent_errors_are_not_retried(db, haiku):
    haiku.answer["decision"] = reply_status.StatusCheckError("bad request", "request")
    _received(db, "<orig@test>")
    reply = _reply(db, "<r@test>", in_reply_to="<orig@test>")

    reply_tracking.update_statuses_from_replies(db, primary_account())

    assert reply.reply_checked_at is not None


def test_account_wide_errors_keep_all_replies_pending(db, api, monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    requests = api((401, _api_error("authentication_error", "invalid x-api-key")))
    _received(db, "<orig@test>")
    first = _reply(db, "<r1@test>", in_reply_to="<orig@test>")
    second = _reply(db, "<r2@test>", in_reply_to="<orig@test>", date=SENT + timedelta(hours=2))

    result = reply_tracking.update_statuses_from_replies(db, primary_account())

    # One failing call, then nothing is marked, so the backlog is checked once the key works.
    assert result["failed"] == 1 and len(requests) == 1
    assert first.reply_checked_at is None and second.reply_checked_at is None
    assert [r.levelname for r in caplog.records if "stopped" in r.getMessage()] == ["ERROR"]


def test_an_unusable_answer_skips_only_that_reply(db, api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    api(
        (200, _answer("end_turn", ANSWER)),
        (200, _answer("max_tokens", '{"status": "Abgeschl')),
        (200, _answer("end_turn", ANSWER)),
    )
    original = _received(db, "<orig@test>")
    replies = [
        _reply(db, f"<r{hour}@test>", in_reply_to="<orig@test>", date=SENT + timedelta(hours=hour))
        for hour in (1, 2, 3)
    ]

    result = reply_tracking.update_statuses_from_replies(db, primary_account())

    assert (result["updated"], result["failed"]) == (2, 1)
    assert [reply.reply_to_id for reply in replies] == [original.id, None, original.id]
    assert all(reply.reply_checked_at is not None for reply in replies)


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


def _failure_kind():
    with pytest.raises(reply_status.StatusCheckError) as error:
        reply_status.assess_reply(Email(), Email(), reply_is_ours=False)
    return error.value.kind


def test_assess_reply_reads_the_answer_through_the_sdk(api):
    api((200, _answer("end_turn", ANSWER)))

    decision = reply_status.assess_reply(Email(), Email(), reply_is_ours=False)

    assert decision == reply_status.StatusDecision(status="Abgeschlossen", confidence=0.95, reason="Erledigt.")


@pytest.mark.parametrize("stop_reason, text", [
    ("max_tokens", '{"status": "Abgeschl'),
    ("refusal", "Dazu kann ich nichts sagen."),
    ("refusal", ""),
    ("refusal", None),
])
def test_unusable_answers_fail_only_this_request(api, stop_reason, text):
    api((200, _answer(stop_reason, text)))

    assert _failure_kind() == "request"


@pytest.mark.parametrize("status, error_type, message, kind", [
    (401, "authentication_error", "invalid x-api-key", "fatal"),
    (402, "billing_error", "Payment required", "fatal"),
    (400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API.", "fatal"),
    (403, "permission_error", "Your API key does not have permission", "fatal"),
    (404, "not_found_error", "model: claude-haiku-0", "fatal"),
    (400, "invalid_request_error", "prompt is too long", "request"),
    (413, "request_too_large", "Request exceeds the maximum size", "request"),
    (429, "rate_limit_error", "Number of requests has exceeded your rate limit", "retryable"),
    (500, "api_error", "Internal server error", "retryable"),
    (529, "overloaded_error", "Overloaded", "retryable"),
])
def test_api_errors_are_classified_by_scope(api, status, error_type, message, kind):
    api((status, _api_error(error_type, message)))

    assert _failure_kind() == kind


@pytest.mark.parametrize("error", ["ConnectError", "ReadTimeout"])
def test_connection_problems_are_retried_next_run(api, error):
    httpx2 = pytest.importorskip("httpx2")
    api(getattr(httpx2, error)("no answer"))

    assert _failure_kind() == "retryable"


def test_mail_text_is_capped():
    assert len(reply_status.mail_text(Email(text="x " * 5000), max_chars=100)) == 100
