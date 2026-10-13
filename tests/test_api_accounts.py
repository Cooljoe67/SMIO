from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import mail as mail_router
from src.api.routers import inbox as inbox_router
from src.db.database import get_db


def _client(db):
    app = FastAPI()
    app.include_router(mail_router.router)
    app.include_router(inbox_router.router)
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def test_unknown_account_is_404(db):
    client = _client(db)

    assert client.get("/mail/sync?account=primay").status_code == 404
    response = client.post("/inbox/undo_last_processing?account=primay")
    assert response.status_code == 404
    assert "primay" in response.json()["detail"]


def test_fetch_reports_the_number_of_new_mails(db, monkeypatch):
    monkeypatch.setattr(mail_router, "fetch_inbox", lambda account: [object(), object()])

    assert _client(db).get("/mail/fetch").json() == {"fetched": 2}
