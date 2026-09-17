"""Builds the plain-language daily mailbox summary shown to the user."""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.api.utils.folder_rules import CLASSIFICATION_FOLDERS
from src.api.utils.settings import app_settings
from src.db.models import Email

STATE_FILE = Path("./data/summary_state.json")


def _load_period_start():
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        last_summary_at = state.get("last_summary_at")
        if last_summary_at:
            return datetime.fromisoformat(last_summary_at)
    # First run ever: fall back to the start of today.
    return datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)


def _save_period_end(timestamp):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"last_summary_at": timestamp.isoformat()}))


def build_daily_summary(db):
    period_start = _load_period_start()
    period_end = datetime.now(timezone.utc).replace(tzinfo=None)

    emails = (
        db.query(Email)
        .filter(Email.processed_at.isnot(None), Email.processed_at >= period_start)
        .all()
    )

    other_senders = sorted({
        email.sender for email in emails
        if email.classification == "other" and email.sender
    })
    moved_counts = {
        classification: sum(1 for email in emails if email.classification == classification)
        for classification in CLASSIFICATION_FOLDERS
    }

    lines = [
        f"Hi {app_settings.user_first_name},",
        "I'm SMIO and this is what is going on in your mailbox today:",
    ]
    if other_senders:
        lines.append(f"You have mails from {', '.join(other_senders)} in your mailbox.")
    for classification, count in moved_counts.items():
        if count:
            lines.append(f"I moved {count} mails to {classification}.")
    if not other_senders and not any(moved_counts.values()):
        lines.append("Nothing new since the last summary.")

    _save_period_end(period_end)

    return {
        "message": "\n".join(lines),
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "other_senders": other_senders,
        "moved_counts": moved_counts,
    }
