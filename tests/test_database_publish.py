import os
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.middleware import publish_database_changes
from src.db import database
from src.db.models import Base, Email
from src.scheduler import workflows


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Point the publisher at a temp SQLite file and record GCS uploads."""
    path = tmp_path / "smio.db"
    path.write_bytes(b"v1")
    calls = []
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    monkeypatch.setattr(database, "USING_SQLITE", True)
    monkeypatch.setattr(database, "_published_signature", database._database_signature())
    monkeypatch.setattr(database.gcs, "enabled", lambda: True)
    monkeypatch.setattr(
        database.gcs,
        "upload_file",
        lambda source, name: calls.append((source, name)) or True,
    )
    return path, calls


def _write(path, content):
    path.write_bytes(content)
    # Make the change visible even on filesystems with coarse timestamps.
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


def test_unchanged_database_is_not_uploaded(uploads):
    _, calls = uploads

    assert database.persist_database() is False
    assert calls == []


def test_changed_database_is_uploaded_once(uploads):
    path, calls = uploads
    _write(path, b"v2")

    assert database.persist_database() is True
    assert database.persist_database() is False
    assert calls == [(path, database.DATABASE_GCS_OBJECT)]


def test_nothing_is_uploaded_without_gcs(uploads, monkeypatch):
    path, calls = uploads
    monkeypatch.setattr(database.gcs, "enabled", lambda: False)
    _write(path, b"v2")

    assert database.persist_database() is False
    assert calls == []


def test_failed_upload_is_retried_next_time(uploads, monkeypatch):
    path, calls = uploads
    _write(path, b"v2")

    def failing_upload(source, name):
        raise RuntimeError("network down")

    monkeypatch.setattr(database.gcs, "upload_file", failing_upload)
    with pytest.raises(RuntimeError):
        database.persist_database()

    monkeypatch.setattr(database.gcs, "upload_file", lambda source, name: calls.append(name) or True)
    assert database.persist_database() is True
    assert calls == [database.DATABASE_GCS_OBJECT]


def test_workflow_uploads_once_after_several_writes(uploads):
    path, calls = uploads

    def routine():
        for version in (b"a", b"bb", b"ccc"):
            _write(path, version)
        return {"done": True}

    assert workflows._run_exclusively("test", routine) == {"done": True}
    assert len(calls) == 1


def test_failed_workflow_still_uploads_and_reraises(uploads):
    path, calls = uploads

    def routine():
        _write(path, b"partial")
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        workflows._run_exclusively("test", routine)
    assert len(calls) == 1
    assert workflows._workflow_lock.acquire(blocking=False)
    workflows._workflow_lock.release()


def test_api_request_uploads_once(uploads):
    path, calls = uploads
    app = FastAPI()
    app.middleware("http")(publish_database_changes)

    @app.post("/write")
    def write():
        _write(path, b"x")
        _write(path, b"xy")
        return {"ok": True}

    @app.get("/read")
    def read():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/read").status_code == 200
    assert calls == []
    assert client.post("/write").status_code == 200
    assert len(calls) == 1


def test_sqlite_reads_leave_the_file_unchanged_and_writes_change_it(tmp_path, monkeypatch):
    path = tmp_path / "real.db"
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    before = database._database_signature()
    with Session() as session:
        session.query(Email).all()
        session.commit()
    assert database._database_signature() == before

    time.sleep(0.01)
    with Session() as session:
        session.add(Email(subject="hello"))
        session.commit()
    assert database._database_signature() != before
    engine.dispose()
