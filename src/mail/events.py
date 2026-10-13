"""Normalized mail events and the outbox their actions run from.

Every change of a stored mail's state is recorded as a MailEvent in the same
transaction as the change itself: by the change detector (src/mail/sync.py)
for what it finds on the server, and by SMIO's own commands for what they do.
So an event exists exactly when its change was committed.

Event types (folders are the account's folder names, Gmail: label names):
  received  a new mail arrived in to_folder
  sent      a mail sent by the account was seen in to_folder (the sent folder)
  moved     the mail went from from_folder to to_folder
  removed   the mail left from_folder: deleted or moved to an unwatched folder
            (source "user", found by a complete read), or trashed by retention
            (source "system", to_folder is the trash)
  read      the mail was first seen as read (complete reads only)

source: "user" for changes found on the server, whoever made them (the owner,
a mail client, a server rule); "system" for SMIO's own commands.

A user's move teaches the classifier at once (_learn_from_move). Other
reactions are actions: registered with @action, or configured as rules in
config/smio.yaml (src/mail/rules.py), and run by dispatch() after the change
is committed. Each action runs at most once per mail, whichever of its events
triggers it; one that raises is retried up to MAX_ATTEMPTS times, so it must
raise only if its effect did not happen.

Events of a folder's first read (an account's existing mail) are initial:
recorded, but no action runs for them, nor for SMIO sorting such mail.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from src.api.utils.folder_rules import classification_for_folder, inbox_folder
from src.db.models import MailEvent, MailEventAction
from src.mail import rules

logger = logging.getLogger(__name__)

EVENT_TYPES = ("received", "sent", "moved", "removed", "read")
SOURCES = ("user", "system")
MAX_ATTEMPTS = 3

# name -> (applies(db, event) -> bool, run(db, event))
_ACTIONS = {}


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def as_dict(event):
    """The event in the provider-neutral wire format (camelCase, like Jira's)."""
    return {
        "eventKey": event.event_key,
        "eventType": event.event_type,
        "provider": event.provider,
        "account": event.account,
        "emailId": event.email_id,
        "messageId": event.message_id,
        "fromFolder": event.from_folder,
        "toFolder": event.to_folder,
        "source": event.source,
        "occurredAt": event.occurred_at.isoformat() if event.occurred_at else None,
    }


def record(
    db, account, email, event_type, from_folder=None, to_folder=None, source="user", initial=False
):
    """Record a change of email that the caller is committing; email needs an id."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type!r}")
    if source not in SOURCES:
        raise ValueError(f"Unknown event source: {source!r}")
    if source == "system" and not initial:
        # Sorting the existing mail of a first read must not draft or forward either.
        initial = (
            db.query(MailEvent.id)
            .filter(
                MailEvent.email_id == email.id,
                MailEvent.event_type.in_(("received", "sent")),
                MailEvent.initial.is_(True),
            )
            .first()
            is not None
        )
    occurred_at = _utcnow()
    seq = (
        db.query(func.max(MailEvent.seq)).filter(MailEvent.email_id == email.id).scalar() or 0
    ) + 1
    event = MailEvent(
        event_key=f"{account.id}:{email.id}:{seq}",
        account=account.id,
        provider=account.provider,
        email_id=email.id,
        seq=seq,
        event_type=event_type,
        message_id=email.message_id,
        from_folder=from_folder,
        to_folder=to_folder,
        source=source,
        occurred_at=occurred_at,
        initial=initial,
        dispatched_at=occurred_at if initial else None,
    )
    db.add(event)
    # Visible to the next record() for this mail (autoflush is off).
    db.flush()
    logger.info(
        "Email %s %s (%s): %s -> %s", email.id, event_type, source, from_folder, to_folder
    )
    if event_type == "moved" and source == "user":
        _learn_from_move(account, email, from_folder, to_folder, event.occurred_at)
    return event


def _learn_from_move(account, email, from_folder, to_folder, now):
    """A user's move between the inbox and the folder list's folders is a label."""
    folder_classification = classification_for_folder(to_folder, account)
    previous_folder_classification = classification_for_folder(from_folder or "", account)
    previous_classification = email.classification
    if folder_classification and int(email.processed or 0) <= 0:
        # Filed by the user before SMIO processed it: the folder is the label.
        # No processing batch, so undo leaves it where the user put it.
        email.processed = 1
        email.processed_at = now
        email.classification = folder_classification
        email.true_label = folder_classification
        email.classification_source = "imap_folder"
        email.folder_list = account.folder_list
    elif (
        folder_classification
        and previous_classification
        and previous_classification != folder_classification
    ):
        email.true_label = folder_classification
        email.classification_source = "imap_folder"
        email.folder_list = account.folder_list
    elif (
        to_folder == inbox_folder(account)
        and previous_folder_classification is not None
        and email.classification_source != "undo"
    ):
        email.classification = "other"
        email.true_label = "other"
        email.classification_source = "imap_folder"
        email.folder_list = account.folder_list

    if folder_classification and email.classification != folder_classification:
        email.classification = folder_classification
        email.classification_source = "imap_folder"
        email.folder_list = account.folder_list


def action(name, applies):
    """Register run(db, event) for the events applies(db, event) accepts."""
    def register(run):
        _ACTIONS[name] = (applies, run)
        return run
    return register


def _run_once(db, event, name, run):
    """Run an action for an event unless it ran or runs for this mail, or failed
    for good; returns its status ("skipped" when another event of the mail ran it)."""
    run_row = (
        db.query(MailEventAction)
        .filter(MailEventAction.email_id == event.email_id, MailEventAction.action == name)
        .first()
    )
    if run_row is not None:
        if run_row.status in ("done", "running") and run_row.event_id != event.id:
            return "skipped"
        if run_row.status in ("done", "running") or (
            run_row.event_id == event.id and run_row.attempts >= MAX_ATTEMPTS
        ):
            if run_row.status == "running":
                logger.warning(
                    "Action %s for event %s was interrupted and is not repeated",
                    name, event.event_key,
                )
            return run_row.status
        if run_row.event_id != event.id:
            # A failure of an earlier event: the new event gets its own attempts.
            run_row.attempts = 0
    else:
        run_row = MailEventAction(email_id=event.email_id, action=name, attempts=0)
        db.add(run_row)
    run_row.event_id = event.id
    run_row.status = "running"
    run_row.attempts += 1
    run_row.updated_at = _utcnow()
    try:
        # Claimed before the effect: from here on the action is never started twice.
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.info("Action %s for event %s was claimed elsewhere", name, event.event_key)
        return "running"
    try:
        run(db, event)
    except Exception as error:
        db.rollback()
        logger.exception("Action %s failed for event %s", name, event.event_key)
        run_row.status = "failed"
        run_row.error = str(error)[:2000]
    else:
        run_row.status = "done"
        run_row.error = None
    run_row.updated_at = _utcnow()
    db.commit()
    return run_row.status


def dispatch(db, account=None):
    """Run the registered actions for the events not yet dispatched, oldest first."""
    query = db.query(MailEvent).filter(MailEvent.dispatched_at.is_(None))
    if account is not None:
        query = query.filter(MailEvent.account == account.id)
    # Rules are read from the config on every call, so edits apply on the next run.
    actions = {**_ACTIONS, **rules.configured_actions()}
    counts = {"events": 0, "done": 0, "skipped": 0, "failed": 0}
    # One attempt per mail and action and call, even if several of its events apply.
    attempted = set()
    for event in query.order_by(MailEvent.id).all():
        counts["events"] += 1
        settled = True
        for name, (applies, run) in actions.items():
            if not applies(db, event):
                continue
            if (event.email_id, name) in attempted:
                settled = False
                continue
            attempted.add((event.email_id, name))
            status = _run_once(db, event, name, run)
            if status in ("done", "skipped"):
                counts[status] += 1
            elif status == "failed":
                counts["failed"] += 1
                attempts = (
                    db.query(MailEventAction.attempts)
                    .filter(
                        MailEventAction.email_id == event.email_id, MailEventAction.action == name
                    )
                    .scalar()
                )
                settled = settled and attempts >= MAX_ATTEMPTS
        if settled:
            event.dispatched_at = _utcnow()
            db.commit()
    return counts
