from datetime import datetime, timedelta, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from types import SimpleNamespace

import pytest

from src import config
from src.ai import reply_draft
from src.api.utils import inbox_processor
from src.config import primary_account
from src.db.models import Email, MailEventAction
from src.mail import drafts, events
from src.mail.imap import ImapMailbox
from tests.test_mail_identity import _message, _predict, _sync, server  # noqa: F401

RULES = """
folder_lists:
  default:
    delivery: {}
    tech: {}
accounts:
  - id: primary
    provider: imap
    email: owner@firma.test
    user_first_name: Marcus
    folder_list: default
    imap: {host: imap.test, username: owner@firma.test, password: x}
actions:
  default:
    draft:
      when:
        - {event: moved, to: delivery}
      do:
        - create_draft: {instructions: "Mit freundlichen Grüßen"}
    sorted_by_model:
      when:
        - {event: moved, to: tech, source: system}
        - {event: received, from: "@alarm.test", subject_contains: STÖRUNG}
      do:
        - create_draft
"""

OVERRIDES = """
    actions:
      draft:
        when:
          - {event: moved, to: delivery}
        do:
          - create_draft: {instructions: "Immer duzen"}
      sorted_by_model: {enabled: false}
"""


def _with_overrides(text=RULES, overrides=OVERRIDES):
    return text.replace("    imap: {host: imap.test", overrides.lstrip("\n") + "    imap: {host: imap.test", 1)


@pytest.fixture
def rules(monkeypatch):
    monkeypatch.setattr(config, "_config", config.parse_config(RULES))


def _parse_error(text):
    with pytest.raises(ValueError) as error:
        config.parse_config(text)
    return str(error.value)


def test_a_rule_must_name_a_folder_of_its_accounts_list():
    assert "has no folder 'zu_beantworten'" in _parse_error(
        RULES.replace("to: delivery", "to: zu_beantworten")
    )


def test_rules_belong_to_a_known_folder_list():
    assert "unknown folder list(s) ['logistik']" in _parse_error(
        RULES.replace("actions:\n  default:", "actions:\n  logistik:")
    )


def test_an_account_can_only_switch_off_a_rule_its_list_has():
    assert "switches off unknown action 'drafts'" in _parse_error(
        _with_overrides(overrides="    actions: {drafts: {enabled: false}}\n")
    )


def test_an_account_rule_replaces_or_switches_off_the_lists_rule():
    parsed = config.parse_config(_with_overrides())

    rules = parsed.rules_for(parsed.account("primary"))

    assert list(rules) == ["draft"]
    assert rules["draft"].do[0].create_draft.instructions == "Immer duzen"
    assert set(config.parse_config(RULES).rules_for(primary_account_of(RULES))) == {
        "draft", "sorted_by_model"
    }


def primary_account_of(text):
    return config.parse_config(text).account("primary")


def test_a_folder_condition_needs_a_moved_event():
    assert "'to' only applies to moved events" in _parse_error(
        RULES.replace("{event: moved, to: delivery}", "{event: received, to: delivery}")
    )


def test_actions_that_do_not_exist_yet_are_refused():
    assert "forward" in _parse_error(
        RULES.replace("        - create_draft\n", "        - forward: {to: a@b.test}\n")
    )


@pytest.fixture
def drafted(monkeypatch, server):
    """The mailbox, with Haiku replaced by a fixed text; records its calls."""
    calls = []
    mailbox = server(**{
        "INBOX": [_message("1")], "INBOX/delivery": [], "INBOX/tech": [], "Entwürfe": [],
    }, flags={"Entwürfe": ("\\Drafts",)})
    monkeypatch.setattr(drafts, "open_mailbox", lambda account: ImapMailbox(account, connection=mailbox))
    monkeypatch.setattr(
        drafts, "write_reply",
        lambda email, signer=None, instructions=None: calls.append((email.subject, signer, instructions))
        or "Guten Tag,\n\nwir prüfen das.\n\nMarcus",
    )
    mailbox.calls = calls
    return mailbox


RECENT = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)


def _sort_new_mails(db, mailbox, monkeypatch, *messages):
    """Deliver messages after the first read and let the model sort them into tech."""
    _sync()
    for message in messages:
        mailbox.deliver("INBOX", message)
    _sync()
    _predict(monkeypatch, mailbox, "tech")
    monkeypatch.setattr(
        inbox_processor, "open_mailbox", lambda account: ImapMailbox(account, connection=mailbox)
    )
    inbox_processor.process_unprocessed_emails(db, primary_account())


def _user_move(mailbox, folder):
    mailbox.current = "INBOX" if mailbox.folders["INBOX"] else mailbox.current
    mailbox.move([mailbox.folders[mailbox.current][0].uid], folder)


def test_moving_a_mail_into_the_folder_writes_one_draft(db, rules, drafted):
    _sync()
    _user_move(drafted, "INBOX/delivery")
    _sync()

    assert events.dispatch(db, primary_account())["done"] == 1
    drafted.current = "INBOX/delivery"
    drafted.move([drafted.folders["INBOX/delivery"][0].uid], "INBOX")
    _sync()
    _user_move(drafted, "INBOX/delivery")
    _sync()
    events.dispatch(db, primary_account())

    assert drafted.calls == [("Lieferung", "Marcus", "Mit freundlichen Grüßen")]
    [(raw, folder, flags)] = drafted.appended
    assert (folder, flags) == ("Entwürfe", ["\\Draft", "\\Seen"])
    draft = message_from_bytes(raw)
    assert draft["Subject"] == "Re: Lieferung"
    assert (draft["To"], draft["From"]) == ("Sender <sender@test>", "owner@firma.test")
    assert draft["In-Reply-To"] == draft["References"] == "<a@test>"
    assert draft["X-SMIO-Event"] == "primary:1:2"
    text = draft.get_payload(decode=True).decode()
    assert text.startswith("Guten Tag,") and "schrieb Sender <sender@test>:\n> Hallo" in text


def test_a_system_only_rule_ignores_the_users_move(db, rules, drafted):
    _sync()
    _user_move(drafted, "INBOX/tech")
    _sync()

    events.dispatch(db, primary_account())

    assert drafted.calls == []


def test_a_rule_fires_when_the_model_sorts_the_mail(db, rules, drafted, monkeypatch):
    _sync()
    drafted.deliver("INBOX", _message(None, message_id="<new@test>", subject="Neu", date=RECENT))
    _sync()
    _predict(monkeypatch, drafted, "tech")
    monkeypatch.setattr(
        inbox_processor, "open_mailbox", lambda account: ImapMailbox(account, connection=drafted)
    )

    inbox_processor.process_unprocessed_emails(db, primary_account())
    events.dispatch(db, primary_account())

    # Lieferung came with the first read, so sorting it triggers nothing.
    assert [call[0] for call in drafted.calls] == ["Neu"]


def test_content_conditions_must_all_hold(db, rules, drafted):
    _sync()
    for uid_message in (
        _message(None, message_id="<x@test>", subject="Störung Tor 3", sender="Alarm <ops@alarm.test>"),
        _message(None, message_id="<y@test>", subject="Wartung", sender="Alarm <ops@alarm.test>"),
        _message(None, message_id="<z@test>", subject="Störung", sender="ops@alarm.test.evil"),
    ):
        drafted.deliver("INBOX", uid_message)
    _sync()

    events.dispatch(db, primary_account())

    assert [call[0] for call in drafted.calls] == ["Störung Tor 3"]


def test_events_of_the_first_read_trigger_nothing(db, rules, drafted):
    drafted.folders["INBOX"].append(
        _message("2", message_id="<old@test>", subject="Störung", sender="ops@alarm.test")
    )

    _sync()

    assert events.dispatch(db, primary_account())["events"] == 0
    assert drafted.calls == [] and db.query(MailEventAction).count() == 0


def test_replies_keep_the_thread_and_encode_umlauts():
    email = Email(
        id=7, subject="AW: Lieferung Köln", sender="Müller <m@kunde.test>", message_id="<b@k>",
        reference_ids="<a@k>", text="Wann kommt die Ware?",
    )
    account = SimpleNamespace(email="owner@firma.test")

    draft = message_from_bytes(drafts.build_reply(email, account, "Hallo").as_bytes())

    assert str(make_header(decode_header(draft["Subject"]))) == "AW: Lieferung Köln"
    assert draft["References"] == "<a@k> <b@k>"
    assert str(make_header(decode_header(draft["To"]))) == "Müller <m@kunde.test>"


class FakeClient:
    def __init__(self, stop_reason="end_turn", text="Sehr geehrte Damen und Herren"):
        self.requests = []
        self.messages = SimpleNamespace(create=self.create)
        self.response = SimpleNamespace(
            stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
        )

    def create(self, **request):
        self.requests.append(request)
        return self.response


def test_haiku_gets_the_mail_the_signer_and_the_rules_instructions(monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(reply_draft, "_get_client", lambda: client)
    email = Email(subject="Lieferung", sender="a@b.test", text="Wann kommt die Ware?")

    assert reply_draft.write_reply(email, "Marcus", "Immer per Sie") == "Sehr geehrte Damen und Herren"

    [request] = client.requests
    assert request["model"] == "claude-haiku-4-5"
    assert "Wann kommt die Ware?" in request["messages"][0]["content"]
    assert "Marcus" in request["system"] and "Immer per Sie" in request["system"]


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_a_cut_off_or_refused_draft_is_an_error(monkeypatch, stop_reason):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(reply_draft, "_get_client", lambda: FakeClient(stop_reason))

    with pytest.raises(reply_draft.DraftError):
        reply_draft.write_reply(Email(subject="x", sender="a@b.test", text="x"))


def test_the_accounts_override_is_what_runs(db, drafted, monkeypatch):
    monkeypatch.setattr(config, "_config", config.parse_config(_with_overrides()))
    _sync()
    _user_move(drafted, "INBOX/delivery")
    _sync()

    events.dispatch(db, primary_account())

    assert drafted.calls == [("Lieferung", "Marcus", "Immer duzen")]


def test_the_model_sorting_old_or_answered_mail_writes_no_draft(db, rules, drafted, monkeypatch):
    _sort_new_mails(
        db, drafted, monkeypatch,
        _message(None, message_id="<old@test>", subject="Alt", date=RECENT - timedelta(days=30)),
        _message(None, message_id="<done@test>", subject="Beantwortet", date=RECENT),
    )
    answered = db.query(Email).filter(Email.subject == "Beantwortet").one()
    db.add(Email(account="primary", subject="AW: Beantwortet", direction="OUTBOUND",
                 reply_to_id=answered.id))
    db.commit()

    events.dispatch(db, primary_account())

    assert drafted.calls == []


def test_drafts_over_the_per_run_cap_wait_for_the_next_run(db, rules, drafted, monkeypatch):
    monkeypatch.setattr(drafts, "MAX_PER_RUN", 1)
    _sort_new_mails(
        db, drafted, monkeypatch,
        _message(None, message_id="<n1@test>", subject="Eins", date=RECENT),
        _message(None, message_id="<n2@test>", subject="Zwei", date=RECENT),
    )

    first = events.dispatch(db, primary_account())
    second = events.dispatch(db, primary_account())

    assert (first["done"], first["deferred"], second["done"]) == (1, 1, 1)
    assert sorted(call[0] for call in drafted.calls) == ["Eins", "Zwei"]


def test_no_drafts_folder_fails_before_haiku_is_paid(db, rules, drafted, monkeypatch):
    monkeypatch.setattr(ImapMailbox, "drafts_folder", lambda self: None)
    _sync()
    _user_move(drafted, "INBOX/delivery")
    _sync()

    assert events.dispatch(db, primary_account())["failed"] == 1
    assert drafted.calls == []
