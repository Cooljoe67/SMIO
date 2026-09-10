from fastapi import FastAPI
from .routers import inbox, delivery, summary, model, imap, classifier

from src.db.database import engine
from src.db.models import Base

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="SMIO API",
    description="Smart Mail Inbox Organizer",
    version="0.4.0"
)

app.include_router(inbox.router)
app.include_router(delivery.router)
app.include_router(summary.router)
app.include_router(model.router)
app.include_router(imap.router)
app.include_router(classifier.router)

@app.get("/")
def root():
    return {"message": "SMIO API running"}
