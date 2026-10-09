"""Updates the status of a mail when a reply in its thread arrives or is sent.

Replies are linked through In-Reply-To/References, or else the case reference.
The status belongs to the received mail the thread is about, so the walk skips
the account's own sent mails. A status set by the user is never overwritten;
the reply then only leaves a suggestion.
"""

import logging
from datetime import datetime, timezone
from email.utils import parseaddr

from sqlalchemy import or_

from src.ai import reply_status
from src.db.models import Email

logger = logging.getLogger(__name__)

USER_STATUS_SOURCES = {"command"}


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _address(value):
    return parseaddr(value or "")[1].strip().casefold()


def _is_own(email, account):
    return email.direction == "OUTBOUND" or _address(email.sender) == _address(account.email)


def _thread_target(db, account, reply):
    """Nearest received mail this reply answers, or None."""
    emails = db.query(Email).filter(
        Email.account == account.id,
        Email.id != reply.id,
        Email.is_training_data.is_(False),
    )
    # In-Reply-To first, then References from the newest to the oldest.
    answered = [reply.in_reply_to] + list(reversed((reply.reference_ids or "").split()))
    for message_id in filter(None, dict.fromkeys(answered)):
        for candidate in emails.filter(Email.message_id == message_id).all():
            if not _is_own(candidate, account):
                return candidate
    if reply.case_ref:
        candidates = emails.filter(
            Email.case_ref == reply.case_ref,
            Email.date < reply.date,
        ).order_by(Email.date.desc()).all()
        return next((c for c in candidates if not _is_own(c, account)), None)
    return None


def _pending_replies(db, account):
    return db.query(Email).filter(
        Email.account == account.id,
        Email.is_training_data.is_(False),
        Email.reply_checked_at.is_(None),
        or_(
            Email.in_reply_to.isnot(None),
            Email.reference_ids.isnot(None),
            Email.case_ref.isnot(None),
        ),
    ).order_by(Email.date).all()


def apply_decision(target, decision):
    target.status_updated_at = _now()
    if decision.confidence >= reply_status.MIN_CONFIDENCE and target.status_source not in USER_STATUS_SOURCES:
        target.status = decision.status
        target.status_source = "reply"
        target.status_reason = decision.reason
        target.status_suggestion = None
        return "updated"
    target.status_suggestion = decision.status
    target.status_reason = f"{decision.reason} (Vorschlag, Konfidenz {decision.confidence:.2f})"
    return "suggested"


def update_statuses_from_replies(db, account):
    """Check new replies of an account; returns counts per outcome."""
    result = {"updated": 0, "suggested": 0, "unlinked": 0, "failed": 0, "waiting": 0}
    pending = _pending_replies(db, account)
    if pending and not reply_status.enabled():
        # Kept unchecked, so they are evaluated once a key is configured.
        logger.info("Account %s: %d replies wait for ANTHROPIC_API_KEY", account.id, len(pending))
        result["waiting"] = len(pending)
        return result

    for reply in pending:
        target = _thread_target(db, account, reply)
        if target is None or _is_own(target, account) or target.classification_source == "system":
            reply.reply_checked_at = _now()
            result["unlinked"] += 1
            continue
        try:
            decision = reply_status.assess_reply(target, reply, reply_is_ours=_is_own(reply, account))
        except reply_status.StatusCheckError as error:
            result["failed"] += 1
            if error.kind == "fatal":
                # Key, credit or model problem: every reply waits until it is fixed.
                logger.error("Status checks for account %s stopped: %s", account.id, error)
                break
            logger.warning("Status check for reply %s failed: %s", reply.id, error)
            if error.kind == "retryable":
                # Rate limit or outage: stop and try again next run.
                break
            reply.reply_checked_at = _now()
            continue
        reply.reply_to_id = target.id
        reply.reply_checked_at = _now()
        outcome = apply_decision(target, decision)
        result[outcome] += 1
        logger.info(
            "Email %s: reply %s -> %s %s (%.2f)",
            target.id, reply.id, outcome, decision.status, decision.confidence,
        )
    db.commit()
    return result
