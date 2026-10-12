"""Action rules from config/smio.yaml (actions:), run on mail events.

A rule applies to an event if any of its `when` entries matches; an entry
matches if every condition it gives holds. Each of its steps becomes an action
"rule:<name>:<step>" of src/mail/events.py, so it runs at most once per mail:
a mail that matches several entries, or is moved into the folder twice, is
handled once.
"""

from email.utils import parseaddr

from src.api.utils.folder_rules import folder_path
from src.config import get_config
from src.db.models import Email

from .drafts import create_reply_draft

# Step kind (a key of src.config.RuleStep) -> run(db, event, options).
STEPS = {"create_draft": create_reply_draft}


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


def _applies(rule, db, event):
    if rule.accounts is not None and event.account not in rule.accounts:
        return False
    try:
        account = get_config().account(event.account)
    except ValueError:
        # The account was removed from the config.
        return False
    email = db.get(Email, event.email_id)
    return email is not None and any(
        matches(trigger, event, email, account) for trigger in rule.when
    )


def configured_actions():
    """The configured rules as events actions: name -> (applies, run)."""
    actions = {}
    for name, rule in get_config().actions.items():
        for step in rule.do:
            actions[f"rule:{name}:{step.kind}"] = (
                lambda db, event, rule=rule: _applies(rule, db, event),
                lambda db, event, step=step: STEPS[step.kind](
                    db, event, getattr(step, step.kind)
                ),
            )
    return actions
