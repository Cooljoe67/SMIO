import json
import os
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker

from src.storage import gcs


DATABASE_PATH = Path(os.getenv("SQLITE_DB_PATH", "./smio.db"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")
DATABASE_GCS_OBJECT = os.getenv("GCS_DATABASE_OBJECT", "databases/smio.db")
USING_SQLITE = DATABASE_URL.startswith("sqlite:")

if gcs.enabled() and USING_SQLITE:
    if not gcs.download_file(DATABASE_GCS_OBJECT, DATABASE_PATH) and DATABASE_PATH.exists():
        gcs.upload_file(DATABASE_PATH, DATABASE_GCS_OBJECT)

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if USING_SQLITE else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def persist_database():
    """Publish the committed local SQLite database to GCS when configured."""
    if gcs.enabled() and USING_SQLITE and DATABASE_PATH.exists():
        gcs.upload_file(DATABASE_PATH, DATABASE_GCS_OBJECT)


@event.listens_for(SessionLocal, "after_commit")
def _persist_committed_database(session):
    persist_database()


def ensure_email_columns():
    required_columns = {
        "order_id": "TEXT",
        "tracking_id": "TEXT",
        "entity_date": "TEXT",
        "entity_name": "TEXT",
        "entity_company": "TEXT",
        "delivery_status": "TEXT",
        "item_name": "TEXT",
        "delivery_date": "TEXT",
        "delivery_company": "TEXT",
        "ner_entities": "TEXT",
        "extraction_sources": "TEXT",
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
        result_json=json.dumps(persisted_result, default=str),
    ))
    db.flush()
    return True


def _string_value(value):
    return None if value is None else str(value)


def ensure_retrain_run_table():
    from src.db.models import RetrainRun

    RetrainRun.__table__.create(bind=engine, checkfirst=True)


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
