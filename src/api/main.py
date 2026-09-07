from fastapi import FastAPI
from .routers import inbox, delivery, summary, model, imap

app = FastAPI(
    title="SMIO API",
    description="Smart Mail Inbox Organizer – Async IMAP Enabled",
    version="0.4.0"
)

app.include_router(inbox.router)
app.include_router(delivery.router)
app.include_router(summary.router)
app.include_router(model.router)
app.include_router(imap.router)

@app.get("/")
def root():
    return {"message": "SMIO API running with async IMAP"}
