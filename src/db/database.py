import atexit
import json
import os
import threading
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from src.config import primary_account
from src.storage import gcs


DATABASE_PATH = Path(os.getenv("SQLITE_DB_PATH", "./smio.db"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")
DATABASE_GCS_OBJECT = os.getenv("GCS_DATABASE_OBJECT", "databases/smio.db")
USING_SQLITE = DATABASE_URL.startswith("sqlite:")

_publish_lock = threading.Lock()
_published_signature = None


def _database_signature():
    """Modification time and size of the local SQLite file, or None if missing."""
    try:
        stat = DATABASE_PATH.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _syncs_with_gcs():
    """Only the deployed service, or a local run with SMIO_GCS=1, mirrors the
    database through GCS (see gcs.syncs)."""
    return gcs.syncs() and USING_SQLITE


if _syncs_with_gcs():
    if not gcs.download_file(DATABASE_GCS_OBJECT, DATABASE_PATH) and DATABASE_PATH.exists():
        gcs.upload_file(DATABASE_PATH, DATABASE_GCS_OBJECT)
    _published_signature = _database_signature()
    if not os.getenv("K_SERVICE"):
        # The service publishes after each workflow and request; an opted-in
        # script publishes its changes when it ends.
        atexit.register(lambda: persist_database())

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if USING_SQLITE else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def persist_database():
    """Publish the local SQLite database to GCS if it changed since the last publish.

    Called once at the end of each workflow and request rather than after every
    commit: each upload sends the whole file. Returns whether it uploaded.
    """
    global _published_signature
    if not _syncs_with_gcs():
        return False
    with _publish_lock:
        # Read the signature before uploading so a write during the upload is
        # still detected by the next call.
        signature = _database_signature()
        if signature is None or signature == _published_signature:
            return False
        gcs.upload_file(DATABASE_PATH, DATABASE_GCS_OBJECT)
        _published_signature = signature
        return True


def ensure_email_columns():
    required_columns = {
        "account": "TEXT",
        "folder_list": "TEXT",
        "provider_message_id": "TEXT",
        "fingerprint": "TEXT",
        "order_id": "TEXT",
        "tracking_id": "TEXT",
        "entity_date": "TEXT",
        "entity_name": "TEXT",
        "entity_company": "TEXT",
        "delivery_status": "TEXT",
        "item_name": "TEXT",
        "delivery_date": "TEXT",
        "delivery_company": "TEXT",
        "extraction": "TEXT",
        "extracted_at": "DATETIME",
        "processing_batch": "INTEGER",
        "message_id": "TEXT",
        "classification_source": "TEXT",
        "read_at": "DATETIME",
        "removed_at": "DATETIME",
        "removal_reason": "TEXT",
        "last_seen_at": "DATETIME",
        "retrain_batch": "INTEGER",
        "processed_at": "DATETIME",
        "is_eval_holdout": "BOOLEAN",
        "is_training_data": "BOOLEAN NOT NULL DEFAULT 0",
        "sender_name": "TEXT",
        "recipients": "TEXT",
        "attachment_names": "TEXT",
        "direction": "TEXT",
        "is_own_domain_sender": "BOOLEAN",
        "is_reply": "BOOLEAN",
        "is_forward": "BOOLEAN",
        "in_reply_to": "TEXT",
        "reference_ids": "TEXT",
        "reply_to_id": "INTEGER",
        "reply_checked_at": "DATETIME",
        "status_source": "TEXT",
        "status_reason": "TEXT",
        "status_suggestion": "TEXT",
        "status_updated_at": "DATETIME",
        "status": "TEXT",
        "priority": "TEXT",
        "action": "TEXT",
        "mail_type_1": "TEXT",
        "mail_type_2": "TEXT",
        "classification_reason": "TEXT",
        "case_ref": "TEXT",
        "true_status": "TEXT",
        "true_priority": "TEXT",
        "true_action": "TEXT",
        "true_mail_type_1": "TEXT",
        "true_mail_type_2": "TEXT",
        "reviewed": "BOOLEAN",
        "note": "TEXT",
        "traits": "TEXT",
        "dataset_key": "TEXT",
        "dataset_split": "TEXT",
        "difficulty": "TEXT",
    }
    existing_columns = {
        column["name"] for column in inspect(engine).get_columns("emails")
    }
    with engine.begin() as connection:
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(f"ALTER TABLE emails ADD COLUMN {column_name} {column_type}")
                )
        # Mails from before multi-account support belong to the primary account;
        # imported labelled data belongs to no mailbox. The config is read only
        # when such mails exist, so migrating never depends on it.
        unassigned = "account IS NULL AND COALESCE(is_training_data, 0) = 0"
        if connection.execute(text(f"SELECT 1 FROM emails WHERE {unassigned} LIMIT 1")).first():
            connection.execute(
                text(f"UPDATE emails SET account = :account WHERE {unassigned}"),
                {"account": primary_account().id},
            )
        # Labels from before folder lists existed refer to the original list.
        connection.execute(text(
            "UPDATE emails SET folder_list = 'default' WHERE folder_list IS NULL "
            "AND (classification IS NOT NULL OR true_label IS NOT NULL)"
        ))
        for column_name in (
            "account", "folder_list", "case_ref", "dataset_key",
            "provider_message_id", "fingerprint", "in_reply_to", "reply_to_id",
        ):
            connection.execute(text(
                f"CREATE INDEX IF NOT EXISTS ix_emails_{column_name} ON emails ({column_name})"
            ))
        connection.execute(
            text("UPDATE emails SET processed = 0 WHERE processed IS NULL")
        )
        connection.execute(
            text("UPDATE emails SET is_eval_holdout = 0 WHERE is_eval_holdout IS NULL")
        )
        connection.execute(
            text("UPDATE emails SET is_training_data = 0 WHERE is_training_data IS NULL")
        )
        connection.execute(
            text(
                "UPDATE emails SET is_training_data = 1 "
                "WHERE COALESCE(processing_batch, 0) = 0 "
                "AND true_label IS NOT NULL "
                "AND COALESCE(uid, '') = '' "
                "AND COALESCE(message_id, '') = '' "
                "AND classification_source IS NULL"
            )
        )
        # Imported labelled data is for training; the holdout measures real corrections.
        connection.execute(
            text("UPDATE emails SET is_eval_holdout = 0 WHERE is_training_data = 1")
        )
        connection.execute(
            text(
                "UPDATE emails SET processing_batch = 0 "
                "WHERE processed > 0 AND processing_batch IS NULL"
            )
        )
        obsolete_columns = {
            "previous_classification",
            "classification_corrected",
            "classification_correction_count",
            "classification_corrected_at",
        }
        for column_name in obsolete_columns & existing_columns:
            connection.execute(text(f"ALTER TABLE emails DROP COLUMN {column_name}"))
    persist_database()


def stored_account_ids():
    """The accounts that stored mails belong to; none before the migration."""
    # The first load of a GCS config asks, and it may come before the tables
    # are created or from inside their migration.
    inspector = inspect(engine)
    if not inspector.has_table("emails") or "account" not in {
        column["name"] for column in inspector.get_columns("emails")
    }:
        return set()
    with engine.connect() as connection:
        return set(connection.execute(
            text("SELECT DISTINCT account FROM emails WHERE account IS NOT NULL")
        ).scalars())


def backfill_fingerprints():
    """Give stored mailbox mails their fingerprint; safe to repeat."""
    from src.api.utils.mail_identity import email_fingerprint
    from src.db.models import Email

    db = SessionLocal()
    try:
        emails = db.query(Email).filter(
            Email.fingerprint.is_(None),
            Email.account.isnot(None),
        ).all()
        for email in emails:
            email.fingerprint = email_fingerprint(email)
        if emails:
            db.commit()
        return len(emails)
    finally:
        db.close()


def requeue_own_address_mails():
    """Queue for classification the inbox mails from an account's own address
    that SMIO took for its own; safe to repeat.

    SMIO used to recognise its mails by their sender, so a FRITZ!Box or a note
    sent from the own address was never classified. Its mails now carry the
    X-SMIO-Mail header and mention SMIO in the subject; the others go back to
    the queue.
    """
    from email.utils import parseaddr

    from src.config import get_config
    from src.db.models import Email

    db = SessionLocal()
    try:
        requeued = 0
        for account in get_config().accounts:
            address = parseaddr(account.email or "")[1].casefold()
            if not address:
                continue
            emails = db.query(Email).filter(
                Email.account == account.id,
                Email.folder == account.mailbox.inbox,
                Email.classification_source == "system",
                Email.removed_at.is_(None),
                Email.is_training_data.is_(False),
                ~Email.subject.ilike("%smio%"),
            ).all()
            for email in emails:
                if parseaddr(email.sender or "")[1].casefold() != address:
                    continue
                email.processed = 0
                email.processed_at = None
                email.processing_batch = None
                email.classification = None
                email.classification_source = None
                email.confidence = None
                requeued += 1
        if requeued:
            db.commit()
        return requeued
    finally:
        db.close()


def ensure_action_claims_per_mail():
    """Rebuild mail_event_actions with one claim per mail and action (it had one
    per event); keeps the strongest claim of each pair. Safe to repeat."""
    from src.db.models import MailEventAction

    if "mail_event_actions" not in inspect(engine).get_table_names():
        return False
    unique = [
        sorted(constraint["column_names"])
        for constraint in inspect(engine).get_unique_constraints("mail_event_actions")
    ]
    if ["action", "email_id"] in unique:
        return False
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE mail_event_actions RENAME TO mail_event_actions_old"))
        # The indexes keep their names when the table is renamed; free them.
        for (index,) in connection.execute(text(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
            " AND tbl_name = 'mail_event_actions_old' AND sql IS NOT NULL"
        )).all():
            connection.execute(text(f'DROP INDEX "{index}"'))
        MailEventAction.__table__.create(bind=connection)
        connection.execute(text("""
            INSERT INTO mail_event_actions
                (id, event_id, email_id, action, status, attempts, error, updated_at)
            SELECT id, event_id, email_id, action, status, attempts, error, updated_at FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY email_id, action
                    ORDER BY CASE status WHEN 'done' THEN 0 WHEN 'running' THEN 1 ELSE 2 END,
                             updated_at DESC, id DESC
                ) AS rank
                FROM mail_event_actions_old
            ) WHERE rank = 1
        """))
        connection.execute(text("DROP TABLE mail_event_actions_old"))
    return True


def ensure_summary_snapshot_columns():
    required_columns = {
        "total_entries": "INTEGER",
        "processed_entries": "INTEGER",
        "correction_count": "INTEGER",
        "pending_corrections": "INTEGER",
        "correction_threshold": "INTEGER",
        "new_count_total": "INTEGER",
        "unread_count_total": "INTEGER",
        "removed_count_total": "INTEGER",
        "retrain_last_run_at": "TEXT",
        "retrain_last_promoted_at": "TEXT",
        "retrain_accuracy": "TEXT",
        "retrain_recall": "TEXT",
        "retrain_f1": "TEXT",
    }
    if "summary_snapshots" not in inspect(engine).get_table_names():
        return
    existing_columns = {
        column["name"] for column in inspect(engine).get_columns("summary_snapshots")
    }
    with engine.begin() as connection:
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(f"ALTER TABLE summary_snapshots ADD COLUMN {column_name} {column_type}")
                )
    persist_database()


def record_retrain_run(db, result, fallback=None):
    """Insert a completed run once, preserving its full result for summaries."""
    from src.db.models import RetrainRun

    ensure_retrain_run_table()
    if not result or not result.get("retrained"):
        return False

    fallback = fallback or {}
    run_at = result.get("run_at") or fallback.get("last_run_at")
    run_id = result.get("run_id") or fallback.get("last_run_id")
    if not run_at:
        return False

    run_at = str(run_at)
    run_key = f"{run_at}:{run_id if run_id is not None else 'unknown'}"
    if db.query(RetrainRun.id).filter(RetrainRun.run_key == run_key).first():
        return False

    persisted_result = {**result, "run_at": run_at, "run_id": run_id}
    db.add(RetrainRun(
        run_key=run_key,
        run_id=run_id,
        run_at=run_at,
        promoted=bool(result.get("promoted")),
        eval_holdout_size=result.get("eval_holdout_size"),
        baseline_accuracy=_string_value(result.get("baseline_accuracy")),
        baseline_recall_macro=_string_value(result.get("baseline_recall_macro")),
        baseline_f1=_string_value(result.get("baseline_f1")),
        new_accuracy=_string_value(result.get("new_accuracy")),
        new_recall_macro=_string_value(result.get("new_recall_macro")),
        new_f1=_string_value(result.get("new_f1")),
        training_duration=_float_value(result.get("training_duration")),
        result_json=json.dumps(persisted_result, default=str),
    ))
    db.flush()
    return True


def _string_value(value):
    return None if value is None else str(value)


def _float_value(value):
    return None if value is None else float(value)


def ensure_retrain_run_table():
    from src.db.models import RetrainRun

    existing_tables = set(inspect(engine).get_table_names())
    RetrainRun.__table__.create(bind=engine, checkfirst=True)
    created = "retrain_runs" not in existing_tables
    columns = {
        column["name"] for column in inspect(engine).get_columns("retrain_runs")
    }
    if "training_duration" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE retrain_runs ADD COLUMN training_duration REAL")
            )
        created = True
    if created:
        persist_database()


def backfill_retrain_runs():
    """Recover run records from historical summary snapshots; safe to repeat."""
    from src.db.models import SummarySnapshot

    ensure_retrain_run_table()
    if "summary_snapshots" not in inspect(engine).get_table_names():
        return 0

    db = SessionLocal()
    inserted = 0
    try:
        snapshots = db.query(SummarySnapshot).order_by(
            SummarySnapshot.summary_date.asc()
        ).all()
        for snapshot in snapshots:
            try:
                summary = json.loads(snapshot.data_json or "{}")
                retraining = summary.get("retraining") or {}
                history = retraining.get("history") or []
            except (TypeError, json.JSONDecodeError):
                retraining = {}
                history = []

            for entry in history:
                if record_retrain_run(db, entry, retraining):
                    inserted += 1

            if snapshot.retrain_result_json:
                try:
                    result = json.loads(snapshot.retrain_result_json)
                except (TypeError, json.JSONDecodeError):
                    continue
                if record_retrain_run(db, result, retraining):
                    inserted += 1

        if inserted:
            db.commit()
        return inserted
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
