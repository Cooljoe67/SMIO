"""Action rules from config/smio.yaml, run on mail events.

Rules are defined per folder list and overridden per account (see
src.config.ActionRule). A rule applies to an event if any of its `when`
entries matches; an entry matches if every condition it gives holds. Each of
its steps becomes an action "rule:<name>:<step>" of src/mail/events.py, so it
runs at most once per mail: a mail that matches several entries, or is moved
into the folder twice, is handled once, also after the rule was overridden.
"""

from email.utils import parseaddr

from src.api.utils.folder_rules import folder_path
from src.config import get_config
from src.db.models import Email

from . import drafts

# Step kind (a key of src.config.RuleStep) -> run(db, event, options).
STEPS = {"create_draft": drafts.create_reply_draft}


def per_run_limit(action_name):
    """(kind, limit) if runs of this action are capped per dispatch, else None."""
    if action_name.startswith("rule:") and action_name.endswith(":create_draft"):
        return "create_draft", drafts.MAX_PER_RUN
    return None


def _sender_matches(pattern, sender):
    address = parseaddr(sender or "")[1].casefold()
    pattern = pattern.casefold()
    return address.endswith(pattern) if pattern.startswith("@") else address == pattern


def _contains(text, part):
    return part.casefold() in (text or "").casefold()


def matches(trigger, event, email, account):
    """Whether one `when` entry matches an event of a stored mail."""
    return (
        event.event_type == trigger.event
        and (trigger.source is None or event.source == trigger.source)
        and (trigger.to is None or event.to_folder == folder_path(trigger.to, account))
        and (trigger.sender is None or _sender_matches(trigger.sender, email.sender))
        and (trigger.subject_contains is None or _contains(email.subject, trigger.subject_contains))
        and (
            trigger.body_contains is None
            or _contains(email.text or email.html, trigger.body_contains)
        )
    )


def _step(event, name, kind):
    """The event's account and its rule `name` if that has a step `kind`, else (None, None)."""
    config = get_config()
    try:
        account = config.account(event.account)
    except ValueError:
        # The account was removed from the config.
        return None, None
    rule = config.rules_for(account).get(name)
    step = next((step for step in rule.do if step.kind == kind), None) if rule else None
    return (account, rule) if step else (None, None)


def _applies(name, kind, db, event):
    account, rule = _step(event, name, kind)
    if rule is None:
        return False
    email = db.get(Email, event.email_id)
    return email is not None and any(
        matches(trigger, event, email, account) for trigger in rule.when
    )


def _run(name, kind, db, event):
    _, rule = _step(event, name, kind)
    step = next(step for step in rule.do if step.kind == kind)
    STEPS[kind](db, event, getattr(step, kind))


def configured_actions():
    """Every account's rules as events actions: "rule:<name>:<kind>" -> (applies, run)."""
    config = get_config()
    steps = {
        (name, step.kind)
        for account in config.accounts
        for name, rule in config.rules_for(account).items()
        for step in rule.do
    }
    return {
        f"rule:{name}:{kind}": (
            lambda db, event, name=name, kind=kind: _applies(name, kind, db, event),
            lambda db, event, name=name, kind=kind: _run(name, kind, db, event),
        )
        for name, kind in sorted(steps)
    }
