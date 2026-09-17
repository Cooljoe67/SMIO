from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./smio.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
        "retrain_batch": "INTEGER",
        "processed_at": "DATETIME",
        "is_eval_holdout": "BOOLEAN",
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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
