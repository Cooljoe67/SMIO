"""Builds the daily mailbox summary shown to the user."""

import html
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import func

from src.api.utils.folder_rules import CLASSIFICATION_FOLDERS
from src.api.utils.settings import app_settings
from src.db.models import Email, SummarySnapshot

RETRAIN_LOG_FILE = Path("./models/retrain_log.jsonl")
RETRAIN_STATE_FILE = Path("./models/retrain_state.json")
RETRAIN_MIN_CORRECTIONS = int(os.getenv("SMIO_MIN_CORRECTIONS", "20"))
SUMMARY_START_HOUR = int(os.getenv("SMIO_SUMMARY_START_HOUR", "8"))
SUMMARY_TIMEZONE = os.getenv("SMIO_SUMMARY_TIMEZONE", "Europe/Berlin")


def _load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_retraining_history():
    entries = []
    if RETRAIN_LOG_FILE.exists():
        for line in RETRAIN_LOG_FILE.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    state = _load_json(RETRAIN_STATE_FILE)
    return entries, state


def _parse_date(value):
    if not value:
        return None
    text = str(value).strip()
    for parser in (
        lambda item: datetime.fromisoformat(item.replace("Z", "+00:00")).date(),
        lambda item: datetime.strptime(item, "%d.%m.%Y").date(),
        lambda item: datetime.strptime(item, "%Y-%m-%d").date(),
    ):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


def _format_datetime(value):
    if not value:
        return "not available"
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(_summary_zone())
    return value.strftime("%d/%m/%Y %H:%M")


def _format_display_date(value):
    return date.fromisoformat(str(value)).strftime("%d/%m/%Y")


def _summary_zone():
    try:
        return ZoneInfo(SUMMARY_TIMEZONE)
    except Exception:
        return timezone.utc


def _summary_date(value=None):
    if value is None:
        return datetime.now(_summary_zone()).date() - timedelta(days=1)
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value))


def summary_period(summary_date=None):
    """Return logical date plus UTC-naive DB boundaries for a summary period."""
    logical_date = _summary_date(summary_date)
    zone = _summary_zone()
    local_start = datetime.combine(
        logical_date,
        time(hour=SUMMARY_START_HOUR),
        tzinfo=zone,
    )
    local_end = local_start + timedelta(days=1)
    return (
        logical_date,
        local_start.astimezone(timezone.utc).replace(tzinfo=None),
        local_end.astimezone(timezone.utc).replace(tzinfo=None),
    )


def _email_command_help_lines():
    return [
        "",
        "Email commands",
        "Reply to this summary with one command on the first line; optionally prefix it with 'SMIO:'.",
        "Use a summary reply subject (for example, AW: SMIO Daily Summary) or the fallback subject 'SMIO Command'.",
        "- UNDO: restore the last processing batch to INBOX.",
        "- RETRAIN: force a retraining attempt; an evaluation holdout is required.",
        "- RELOAD MODEL: reload the deployed classifier.",
        "- SUMMARY: send an on-demand summary.",
    ]


def _delivery_snapshot(emails, period_start, reference_date):
    deliveries = [email for email in emails if email.classification == "delivery"]
    today = reference_date
    upcoming = []
    delivered_since_summary = []
    status_counts = {}
    for email in deliveries:
        status = (email.delivery_status or "unknown").strip().lower()
        status_counts[status] = status_counts.get(status, 0) + 1
        delivery_date = _parse_date(email.delivery_date or email.entity_date)
        if delivery_date and delivery_date >= today:
            upcoming.append({
                "date": delivery_date.isoformat(),
                "item": email.item_name or email.entity_name or email.subject or "Unknown item",
                "company": email.delivery_company or email.entity_company or "",
                "status": email.delivery_status or "unknown",
                "tracking_id": email.tracking_id or "",
            })
        if (
            delivery_date == today
            and email.processed_at
            and email.processed_at >= period_start
        ):
            delivered_since_summary.append(email.item_name or email.subject or "Unknown item")
    upcoming.sort(key=lambda item: item["date"])
    return {
        "today_count": sum(item["date"] == today.isoformat() for item in upcoming),
        "upcoming": upcoming[:10],
        "delivered_since_summary": delivered_since_summary[:10],
        "status_counts": status_counts,
    }


def _retraining_snapshot():
    history, state = _load_retraining_history()
    latest = history[-1] if history else {}
    promoted = [entry for entry in history if entry.get("promoted")]
    return {
        "last_run_at": state.get("last_run_at") or latest.get("run_at"),
        "last_promoted_at": (
            state.get("last_promoted_at")
            or (promoted[-1].get("run_at") if promoted else None)
        ),
        "last_run_id": state.get("last_run_id"),
        "last_metrics": latest or state.get("last_metrics"),
        "history": history[-10:],
        "promoted_runs": len(promoted),
    }


def build_daily_summary(db, summary_date=None, persist=False):
    """Build a summary for one logical day, defaulting to yesterday."""
    logical_date, period_start, period_end = summary_period(summary_date)

    emails = (
        db.query(Email)
        .filter(
            Email.is_training_data.is_(False),
            Email.processed_at.isnot(None),
            Email.processed_at >= period_start,
            Email.processed_at < period_end,
        )
        .all()
    )

    all_emails = db.query(Email).filter(Email.is_training_data.is_(False)).all()

    other_senders = sorted({
        email.sender for email in emails
        if email.classification == "other" and email.sender
    })
    new_counts = {
        classification: sum(1 for email in emails if email.classification == classification)
        for classification in CLASSIFICATION_FOLDERS
    }
    new_counts["other"] = sum(1 for email in emails if email.classification == "other")
    removed_counts = {
        classification: sum(
            1 for email in all_emails
            if email.classification == classification
            and email.removed_at is not None
            and email.removal_reason == "retention"
            and email.removed_at >= period_start
        )
        for classification in (*CLASSIFICATION_FOLDERS, "other")
    }
    unread_counts = {
        classification: sum(
            1 for email in all_emails
            if email.classification == classification
            and email.read_at is None
            and email.removed_at is None
        )
        for classification in (*CLASSIFICATION_FOLDERS, "other")
    }
    pending_corrections = sum(
        1 for email in all_emails
        if email.true_label and email.classification_source == "imap_folder"
        and email.retrain_batch is None and not email.is_eval_holdout
    )
    correction_count = sum(
        1 for email in all_emails
        if email.true_label and email.classification_source == "imap_folder"
    )
    retraining = _retraining_snapshot()
    deliveries = _delivery_snapshot(emails, period_start, logical_date)
    retrain_due = pending_corrections >= RETRAIN_MIN_CORRECTIONS
    db_snapshot = {
        "total_entries": len(all_emails),
        "processed_entries": sum(1 for email in all_emails if email.processed_at),
        "pending_corrections": pending_corrections,
        "corrections": correction_count,
        "correction_threshold": RETRAIN_MIN_CORRECTIONS,
        "retrain_due": retrain_due,
    }

    lines = [
        f"Hi {app_settings.user_first_name},",
        "SMIO daily summary",
        f"Period: {_format_datetime(period_start)} - {_format_datetime(period_end)}",
        "",
        "Mailbox",
        f"- Database entries: {db_snapshot['total_entries']}",
        f"- Corrections: {correction_count} ({pending_corrections}/"
        f"{RETRAIN_MIN_CORRECTIONS} pending for retraining)"
        + (" - RETRAINING DUE" if retrain_due else ""),
    ]
    if other_senders:
        lines.append(f"- Other senders: {', '.join(other_senders)}")
    lines.append("")
    lines.append("Categories")
    for classification, count in new_counts.items():
        if count:
            lines.append(
                f"- {classification}: {count} new, "
                f"{unread_counts[classification]} unread, "
                f"{removed_counts[classification]} removed"
            )
    if not any(new_counts.values()):
        lines.append("- Nothing new since the last summary.")
    lines.extend(("", "Deliveries"))
    lines.append(f"- Due today: {deliveries['today_count']}")
    for item in deliveries["upcoming"][:5]:
        lines.append(
            f"- {_format_display_date(item['date'])}: "
            f"{item['item']} ({item['status']})"
        )
    if deliveries["delivered_since_summary"]:
        lines.append(
            "- Delivered since last summary: "
            + ", ".join(deliveries["delivered_since_summary"])
        )
    lines.extend(("", "Model"))
    lines.append(f"- Last retraining: {retraining['last_run_at'] or 'not available'}")
    lines.append(f"- Last promoted model: {retraining['last_promoted_at'] or 'not available'}")
    metrics = retraining["last_metrics"]
    if metrics:
        lines.append(
            "- Latest metrics: "
            f"accuracy {metrics.get('accuracy', metrics.get('new_accuracy', 'n/a'))}, "
            f"recall {metrics.get('recall_macro', metrics.get('new_recall_macro', 'n/a'))}, "
            f"F1 {metrics.get('f1_macro', metrics.get('new_f1', 'n/a'))}"
        )
    lines.extend(_email_command_help_lines())

    summary = {
        "summary_date": logical_date.isoformat(),
        "message": "\n".join(lines),
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "other_senders": other_senders,
        "new_counts": new_counts,
        "unread_counts": unread_counts,
        "moved_counts": new_counts,
        "removed_counts": removed_counts,
        "db": db_snapshot,
        "deliveries": deliveries,
        "retraining": retraining,
    }
    summary["html_message"] = _format_html_summary(summary)
    return summary


def _metric(metrics, primary_key, fallback_key):
    return metrics.get(primary_key, metrics.get(fallback_key)) if metrics else None


def gather_daily_summary(db, summary_date=None, force=False):
    """Build and persist the snapshot for one logical day.

    Only the most recently stored summary date may be silently re-gathered
    (e.g. re-running the daily job soon after it already ran). Overwriting an
    older, already-completed snapshot requires force=True.
    """
    summary = build_daily_summary(db, summary_date=summary_date)
    existing = db.query(SummarySnapshot).filter(
        SummarySnapshot.summary_date == summary["summary_date"]
    ).one_or_none()
    if existing is not None and not force:
        latest_date = db.query(func.max(SummarySnapshot.summary_date)).scalar()
        if summary["summary_date"] != latest_date:
            raise ValueError(
                f"Summary snapshot for {summary['summary_date']} already exists and is "
                "not the latest stored summary; pass force=True to overwrite it."
            )

    snapshot = existing or SummarySnapshot(summary_date=summary["summary_date"])
    if existing is None:
        db.add(snapshot)
    snapshot.period_start = datetime.fromisoformat(summary["period_start"])
    snapshot.period_end = datetime.fromisoformat(summary["period_end"])
    snapshot.gathered_at = datetime.now(timezone.utc).replace(tzinfo=None)
    snapshot.data_json = json.dumps(summary, default=str)
    snapshot.message = summary["message"]
    snapshot.html_message = summary["html_message"]
    snapshot.retrain_due = summary["db"]["retrain_due"]
    snapshot.send_status = snapshot.send_status or "not_sent"

    metrics = summary["retraining"]["last_metrics"]
    snapshot.total_entries = summary["db"]["total_entries"]
    snapshot.processed_entries = summary["db"]["processed_entries"]
    snapshot.correction_count = summary["db"]["corrections"]
    snapshot.pending_corrections = summary["db"]["pending_corrections"]
    snapshot.correction_threshold = summary["db"]["correction_threshold"]
    snapshot.new_count_total = sum(summary["new_counts"].values())
    snapshot.unread_count_total = sum(summary["unread_counts"].values())
    snapshot.removed_count_total = sum(summary["removed_counts"].values())
    snapshot.retrain_last_run_at = summary["retraining"]["last_run_at"]
    snapshot.retrain_last_promoted_at = summary["retraining"]["last_promoted_at"]
    snapshot.retrain_accuracy = _metric(metrics, "accuracy", "new_accuracy")
    snapshot.retrain_recall = _metric(metrics, "recall_macro", "new_recall_macro")
    snapshot.retrain_f1 = _metric(metrics, "f1_macro", "new_f1")

    db.commit()
    return summary


def stored_daily_summary(db, summary_date=None):
    logical_date = _summary_date(summary_date)
    snapshot = db.query(SummarySnapshot).filter(
        SummarySnapshot.summary_date == logical_date.isoformat()
    ).one_or_none()
    if snapshot is None:
        return None
    summary = json.loads(snapshot.data_json)
    summary["retraining"] = _retraining_snapshot()
    message_lines = summary["message"].splitlines()
    if "Model" in message_lines:
        message_lines = message_lines[:message_lines.index("Model") + 1]
    else:
        message_lines.extend(("", "Model"))
    retraining = summary["retraining"]
    message_lines.append(
        f"- Last retraining: {retraining['last_run_at'] or 'not available'}"
    )
    message_lines.append(
        f"- Last promoted model: {retraining['last_promoted_at'] or 'not available'}"
    )
    metrics = retraining["last_metrics"]
    if metrics:
        message_lines.append(
            "- Latest metrics: "
            f"accuracy {metrics.get('accuracy', metrics.get('new_accuracy', 'n/a'))}, "
            f"recall {metrics.get('recall_macro', metrics.get('new_recall_macro', 'n/a'))}, "
            f"F1 {metrics.get('f1_macro', metrics.get('new_f1', 'n/a'))}"
        )
    message_lines.extend(_email_command_help_lines())
    summary["message"] = "\n".join(message_lines)
    summary["html_message"] = _format_html_summary(summary)
    summary["sent_at"] = snapshot.sent_at.isoformat() if snapshot.sent_at else None
    summary["send_status"] = snapshot.send_status
    summary["retrain_result"] = (
        json.loads(snapshot.retrain_result_json)
        if snapshot.retrain_result_json else None
    )
    return summary


def mark_summary_sent(db, summary_date, sent):
    logical_date = _summary_date(summary_date)
    snapshot = db.query(SummarySnapshot).filter(
        SummarySnapshot.summary_date == logical_date.isoformat()
    ).one()
    snapshot.send_status = "sent" if sent else "failed"
    snapshot.sent_at = datetime.now(timezone.utc).replace(tzinfo=None) if sent else None
    db.commit()
    return stored_daily_summary(db, logical_date)


def _format_html_summary(summary):
    def number(value):
        return html.escape(str(value))

    categories = []
    for classification, count in summary["new_counts"].items():
        unread = summary["unread_counts"].get(classification, 0)
        removed = summary["removed_counts"].get(classification, 0)
        categories.append(
            f"<tr><td>{html.escape(classification.title())}</td>"
            f"<td>{number(count)}</td><td>{number(unread)}</td>"
            f"<td>{number(removed)}</td></tr>"
        )
    bars = []
    maximum = max(summary["new_counts"].values(), default=1)
    for classification, count in summary["new_counts"].items():
        width = round((count / maximum) * 100) if maximum else 0
        bars.append(
            f"<div class=\"bar-row\"><span>{html.escape(classification.title())}</span>"
            f"<div class=\"bar-track\"><div class=\"bar\" style=\"width:{width}%\"></div></div>"
            f"<b>{number(count)}</b></div>"
        )
    delivery_rows = []
    for item in summary["deliveries"]["upcoming"]:
        delivery_rows.append(
            "<tr>"
            f"<td>{number(_format_display_date(item['date']))}</td>"
            f"<td>{html.escape(item['item'])}</td>"
            f"<td>{html.escape(item['company'])}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            "</tr>"
        )
    metrics = summary["retraining"]["last_metrics"]
    metric_text = "No retraining metrics available yet."
    if metrics:
        metric_text = (
            f"Accuracy: {number(metrics.get('accuracy', metrics.get('new_accuracy', 'n/a')))} · "
            f"Recall: {number(metrics.get('recall_macro', metrics.get('new_recall_macro', 'n/a')))} · "
            f"F1: {number(metrics.get('f1_macro', metrics.get('new_f1', 'n/a')))}"
        )
    history_rows = []
    for entry in reversed(summary["retraining"]["history"]):
        history_rows.append(
            "<tr>"
            f"<td>{number(entry.get('run_at', entry.get('run_id', 'n/a')))}</td>"
            f"<td>{number(entry.get('eval_holdout_size', 'n/a'))}</td>"
            f"<td>{number(entry.get('new_accuracy', 'n/a'))}</td>"
            f"<td>{number(entry.get('new_recall_macro', 'n/a'))}</td>"
            f"<td>{number(entry.get('new_f1', entry.get('f1', 'n/a')))}</td>"
            f"<td>{'promoted' if entry.get('promoted') else 'kept as candidate'}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><body style="margin:0;background:#f4f6f8;color:#17212b;font-family:Arial,sans-serif">
<div style="max-width:720px;margin:0 auto;padding:28px 18px">
<div style="background:#173f5f;color:white;padding:24px;border-radius:10px 10px 0 0">
<div style="font-size:13px;letter-spacing:1px;text-transform:uppercase">SMIO</div>
<h1 style="margin:6px 0 4px;font-size:26px">Daily mailbox summary - {number(_format_datetime(summary['period_end']))}</h1>
<div style="opacity:.82">{number(_format_datetime(summary['period_start']))} to {number(_format_datetime(summary['period_end']))}</div>
</div>
<div style="background:white;padding:22px;border:1px solid #dce3e8;border-top:0">
<h2>Mailbox</h2>
<p><b>{number(summary['db']['total_entries'])}</b> database entries ·
<b>{number(summary['db']['pending_corrections'])}/{number(summary['db']['correction_threshold'])}</b> pending corrections</p>
{'<p style="color:#b54708"><b>Retraining is due.</b></p>' if summary['db']['retrain_due'] else ''}
<h2>Categories</h2>
<table style="width:100%;border-collapse:collapse"><tr><th align="left">Category</th><th align="left">New</th><th align="left">Unread</th><th align="left">Removed</th></tr>{''.join(categories)}</table>
<div style="margin:18px 0">{''.join(bars)}</div>
<h2>Deliveries</h2>
<p>Due today: <b>{number(summary['deliveries']['today_count'])}</b></p>
<table style="width:100%;border-collapse:collapse"><tr><th align="left">Date</th><th align="left">Item</th><th align="left">Company</th><th align="left">Status</th></tr>{''.join(delivery_rows) or '<tr><td colspan="4">No upcoming deliveries found.</td></tr>'}</table>
<h2>Model</h2>
<p>Last retraining: {number(summary['retraining']['last_run_at'] or 'not available')}<br>
Last promoted model: {number(summary['retraining']['last_promoted_at'] or 'not available')}<br>{metric_text}</p>
<h3 style="font-size:15px;color:#173f5f">Retraining history</h3>
<table style="width:100%;border-collapse:collapse"><tr><th align="left">Run</th><th align="left">Holdout</th><th align="left">Accuracy</th><th align="left">Recall</th><th align="left">F1</th><th align="left">Status</th></tr>{''.join(history_rows) or '<tr><td colspan="6">No retraining runs found.</td></tr>'}</table>
<h3 style="font-size:15px;color:#173f5f">Email commands</h3>
<p>Reply to this summary with one command on the first line; optionally prefix it with <code>SMIO:</code>.<br>
Use a summary reply subject (for example, <code>AW: SMIO Daily Summary</code>) or the fallback subject <code>SMIO Command</code>.</p>
<ul><li><b>UNDO</b>: restore the last processing batch to INBOX.</li>
<li><b>RETRAIN</b>: force a retraining attempt; an evaluation holdout is required.</li>
<li><b>RELOAD MODEL</b>: reload the deployed classifier.</li>
<li><b>SUMMARY</b>: send an on-demand summary.</li></ul>
</div></div>
<style>h2{{font-size:18px;margin:22px 0 10px;color:#173f5f}}td,th{{padding:8px 6px;border-bottom:1px solid #e5e9ed;font-size:14px}}.bar-row{{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:13px}}.bar-row span{{width:90px}}.bar-track{{height:10px;background:#e6edf2;border-radius:5px;flex:1;overflow:hidden}}.bar{{height:100%;background:#3caea3;border-radius:5px}}</style>
</body></html>"""
