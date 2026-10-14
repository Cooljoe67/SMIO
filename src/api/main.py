import logging

from fastapi import FastAPI
from .middleware import publish_database_changes
from .routers import events, inbox, jobs, mail, summary

from src.db.database import migrate
from src.scheduler.workflows import publish_startup_changes

logging.basicConfig(level=logging.INFO)

migrate()
publish_startup_changes()


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
