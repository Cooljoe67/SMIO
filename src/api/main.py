import logging

from fastapi import FastAPI
from .middleware import publish_database_changes
from .routers import events, inbox, jobs, mail, summary

from src.db.database import (
    backfill_fingerprints,
    backfill_retrain_runs,
    engine,
    ensure_action_claims_per_mail,
    ensure_email_columns,
    forget_deploy_echo_corrections,
    requeue_own_address_mails,
    ensure_summary_snapshot_columns,
    persist_database,
)
from src.db.models import Base

logging.basicConfig(level=logging.INFO)

Base.metadata.create_all(bind=engine)
ensure_email_columns()
ensure_action_claims_per_mail()
ensure_summary_snapshot_columns()
backfill_retrain_runs()
backfill_fingerprints()
requeue_own_address_mails()
forget_deploy_echo_corrections()
persist_database()


app = FastAPI(
    title="SMIO API",
    description="Smart Mail Inbox Organizer",
    version="0.4.0"
)


app.middleware("http")(publish_database_changes)
app.include_router(inbox.router)
app.include_router(summary.router)
app.include_router(mail.router)
app.include_router(events.router)
app.include_router(jobs.router)

@app.get("/")
def root():
    return {"message": "SMIO API running"}
