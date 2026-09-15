import logging

from fastapi import FastAPI
from .routers import inbox, summary, imap

from src.db.database import engine, ensure_email_columns
from src.db.models import Base

logging.basicConfig(level=logging.INFO)

Base.metadata.create_all(bind=engine)
ensure_email_columns()


app = FastAPI(
    title="SMIO API",
    description="Smart Mail Inbox Organizer",
    version="0.4.0"
)

app.include_router(inbox.router)
app.include_router(summary.router)
app.include_router(imap.router)

@app.get("/")
def root():
    return {"message": "SMIO API running"}
