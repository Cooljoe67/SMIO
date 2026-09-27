"""Backfill fixed-period summary snapshots from an existing SQLite database."""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def _arguments():
    parser = argparse.ArgumentParser(
        description="Create missing SummarySnapshot rows from processed email history."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("./smio.db"),
        help="Local SQLite copy to read and update (default: ./smio.db).",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        help="First logical summary date to backfill (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        help="Last logical summary date to backfill (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing snapshots for historical dates, not just the latest.",
    )
    return parser.parse_args()


def _logical_date(timestamp, start_hour, zone):
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    local_timestamp = timestamp.astimezone(zone)
    logical_date = local_timestamp.date()
    if local_timestamp.hour < start_hour:
        logical_date -= timedelta(days=1)
    return logical_date


def main():
    arguments = _arguments()
    db_path = arguments.db_path.resolve()
    if not db_path.exists():
        raise SystemExit(f"Database copy does not exist: {db_path}")

    # Never let a backfill commit upload the local copy to GCS.
    os.environ.pop("GCS_BUCKET", None)
    os.environ["SQLITE_DB_PATH"] = str(db_path)
    os.environ.pop("DATABASE_URL", None)
    os.environ.setdefault("IMAP_HOST", "backfill.invalid")
    os.environ.setdefault("IMAP_USER", "backfill.invalid")
    os.environ.setdefault("IMAP_PASSWORD", "backfill.invalid")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.ai.daily_summary import (
        RETRAIN_MIN_CORRECTIONS,
        SUMMARY_START_HOUR,
        SUMMARY_TIMEZONE,
        gather_daily_summary,
    )
    from src.db.database import SessionLocal, engine, ensure_summary_snapshot_columns
    from src.db.models import Base, Email

    zone = ZoneInfo(SUMMARY_TIMEZONE)
    Base.metadata.create_all(bind=engine)
    ensure_summary_snapshot_columns()
    db = SessionLocal()
    try:
        timestamps = db.query(Email.processed_at).filter(
            Email.processed_at.isnot(None)
        ).all()
        logical_dates = sorted({
            _logical_date(timestamp, SUMMARY_START_HOUR, zone)
            for (timestamp,) in timestamps
        })
        if arguments.start:
            logical_dates = [
                item for item in logical_dates if item >= arguments.start
            ]
        if arguments.end:
            logical_dates = [
                item for item in logical_dates if item <= arguments.end
            ]

        print(
            f"Backfilling {len(logical_dates)} logical days in {db_path} "
            f"using {SUMMARY_START_HOUR:02d}:00 {SUMMARY_TIMEZONE} "
            f"(threshold {RETRAIN_MIN_CORRECTIONS})"
        )
        for logical_date in logical_dates:
            summary = gather_daily_summary(db, logical_date, force=arguments.force)
            print(
                f"{logical_date.isoformat()}: "
                f"{sum(summary['new_counts'].values())} new, "
                f"{summary['db']['pending_corrections']} pending corrections"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
