import logging

from fastapi import FastAPI
from .middleware import publish_database_changes
from .routers import imap, inbox, jobs, summary

from src.db.database import (
    backfill_retrain_runs,
    engine,
    ensure_email_columns,
    ensure_summary_snapshot_columns,
    persist_database,
)
from src.db.models import Base

logging.basicConfig(level=logging.INFO)

Base.metadata.create_all(bind=engine)
ensure_email_columns()
ensure_summary_snapshot_columns()
backfill_retrain_runs()
persist_database()


app = FastAPI(
    title="SMIO API",
    description="Smart Mail Inbox Organizer",
    version="0.4.0"
)


app.middleware("http")(publish_database_changes)
app.include_router(inbox.router)
app.include_router(summary.router)
app.include_router(imap.router)
app.include_router(jobs.router)

@app.get("/")
def root():
    return {"message": "SMIO API running"}
