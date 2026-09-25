"""Builds the daily mailbox summary shown to the user."""

import html
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.api.utils.folder_rules import CLASSIFICATION_FOLDERS
from src.api.utils.settings import app_settings
from src.db.models import Email
from src.storage import gcs

STATE_FILE = Path("./data/summary_state.json")
STATE_GCS_OBJECT = "state/summary_state.json"
RETRAIN_LOG_FILE = Path("./models/retrain_log.jsonl")
RETRAIN_STATE_FILE = Path("./models/retrain_state.json")
RETRAIN_MIN_CORRECTIONS = int(os.getenv("SMIO_MIN_CORRECTIONS", "20"))


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
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    today = datetime.now(timezone.utc).replace(tzinfo=None).date()
    if value.date() == today:
        prefix = "Today"
    elif value.date() == today - timedelta(days=1):
        prefix = "Yesterday"
    else:
        return value.strftime("%d/%m/%Y %H:%M")
    return f"{prefix} {value.strftime('%H:%M')}"


def _delivery_snapshot(emails, period_start):
    deliveries = [email for email in emails if email.classification == "delivery"]
    today = date.today()
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
        "last_run_at": state.get("last_run_at"),
        "last_promoted_at": (
            state.get("last_promoted_at")
            or (promoted[-1].get("run_at") if promoted else None)
        ),
        "last_run_id": state.get("last_run_id"),
        "last_metrics": state.get("last_metrics") or latest,
        "history": history[-10:],
        "promoted_runs": len(promoted),
    }


def _load_period_start():
    if gcs.enabled():
        gcs.download_file(STATE_GCS_OBJECT, STATE_FILE)
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        last_summary_at = state.get("last_summary_at")
        if last_summary_at:
            parsed = datetime.fromisoformat(last_summary_at)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
    # First run ever: fall back to the start of today.
    return datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)


def _save_period_end(timestamp):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"last_summary_at": timestamp.isoformat()}))
    gcs.upload_file(STATE_FILE, STATE_GCS_OBJECT)


def persist_summary_period_end(timestamp):
    """Advance the summary period after its scheduled delivery succeeds."""
    _save_period_end(datetime.fromisoformat(timestamp))


def build_daily_summary(db, persist=True):
    """Build the digest since the last summary.

    persist=False builds an on-demand summary (e.g. requested via instruction mail)
    without moving the period start used by the next scheduled summary.
    """
    period_start = _load_period_start()
    period_end = datetime.now(timezone.utc).replace(tzinfo=None)

    emails = (
        db.query(Email)
        .filter(Email.processed_at.isnot(None), Email.processed_at >= period_start)
        .all()
    )

    all_emails = db.query(Email).all()

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
    deliveries = _delivery_snapshot(all_emails, period_start)
    db_snapshot = {
        "total_entries": len(all_emails),
        "processed_entries": sum(1 for email in all_emails if email.processed_at),
        "pending_corrections": pending_corrections,
        "corrections": correction_count,
        "correction_threshold": RETRAIN_MIN_CORRECTIONS,
    }

    lines = [
        f"Hi {app_settings.user_first_name},",
        "SMIO daily summary",
        f"Period: {_format_datetime(period_start)} - {_format_datetime(period_end)}",
        "",
        "Mailbox",
        f"- Database entries: {db_snapshot['total_entries']}",
        f"- Corrections: {correction_count} ({pending_corrections}/"
        f"{RETRAIN_MIN_CORRECTIONS} pending for retraining)",
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
        lines.append(f"- {item['date']}: {item['item']} ({item['status']})")
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

    if persist:
        _save_period_end(period_end)

    summary = {
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
            f"<td>{number(item['date'])}</td>"
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
</div></div>
<style>h2{{font-size:18px;margin:22px 0 10px;color:#173f5f}}td,th{{padding:8px 6px;border-bottom:1px solid #e5e9ed;font-size:14px}}.bar-row{{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:13px}}.bar-row span{{width:90px}}.bar-track{{height:10px;background:#e6edf2;border-radius:5px;flex:1;overflow:hidden}}.bar{{height:100%;background:#3caea3;border-radius:5px}}</style>
</body></html>"""
