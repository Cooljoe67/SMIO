import os

import pytest

from src.db import database
from src.scheduler import workflows
from src.storage import gcs
from tests import fake_gcs


@pytest.fixture
def bucket(monkeypatch):
    monkeypatch.setattr(gcs, "enabled", lambda: True)
    monkeypatch.setenv("K_SERVICE", "smio")
    monkeypatch.setattr(workflows, "reload_config", lambda **kwargs: None)
    return fake_gcs.install(monkeypatch, gcs)


def test_a_lease_is_exclusive_until_released_or_expired(bucket):
    first = gcs.acquire_lease("locks/w.json", "old", 60)

    assert first is not None
    assert gcs.acquire_lease("locks/w.json", "new", 60) is None
    gcs.release_lease("locks/w.json", first)
    second = gcs.acquire_lease("locks/w.json", "new", -1)  # already expired
    assert second is not None
    # An expired lease is taken over; its old holder's release then does nothing.
    third = gcs.acquire_lease("locks/w.json", "other", 60)
    gcs.release_lease("locks/w.json", second)
    assert gcs.acquire_lease("locks/w.json", "new", 60) is None
    gcs.release_lease("locks/w.json", third)


def test_a_workflow_skips_while_another_instance_holds_the_lease(bucket, monkeypatch):
    monkeypatch.setattr(workflows, "refresh_database", lambda: False)
    monkeypatch.setattr(workflows, "persist_database", lambda: False)
    held = gcs.acquire_lease(workflows.LEASE_OBJECT, "other revision", 60)
    ran = []

    result = workflows._run_exclusively("test", lambda: ran.append(1))

    assert result == {"skipped": True, "reason": "workflow_in_progress"}
    assert ran == []
    gcs.release_lease(workflows.LEASE_OBJECT, held)
    workflows._run_exclusively("test", lambda: ran.append(1))
    assert ran == [1]
    # Released after the run.
    assert workflows.LEASE_OBJECT not in bucket.objects


def _touch(path, content):
    path.write_bytes(content)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


@pytest.fixture
def database_file(bucket, tmp_path, monkeypatch):
    path = tmp_path / "smio.db"
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    monkeypatch.setattr(database, "USING_SQLITE", True)
    bucket.blob(database.DATABASE_GCS_OBJECT).upload_from_string(b"v1")
    monkeypatch.setattr(
        database, "_published_generation",
        gcs.download_versioned(database.DATABASE_GCS_OBJECT, path),
    )
    monkeypatch.setattr(database, "_published_signature", database._database_signature())
    return path


def test_a_newer_database_of_another_instance_is_loaded_first(bucket, database_file, monkeypatch):
    migrated = []
    monkeypatch.setattr(database, "migrate", lambda: migrated.append(database_file.read_bytes()))
    bucket.blob(database.DATABASE_GCS_OBJECT).upload_from_string(b"v2 from the other revision")

    assert database.refresh_database() is True
    assert database_file.read_bytes() == b"v2 from the other revision"
    # The older revision's database gets this revision's migrations.
    assert migrated == [b"v2 from the other revision"]
    assert database.refresh_database() is False
    _touch(database_file, b"v3")
    assert database.persist_database() is True
    assert bucket.objects[database.DATABASE_GCS_OBJECT][0] == b"v3"


def test_a_stale_copy_never_overwrites_the_newer_database(bucket, database_file):
    bucket.blob(database.DATABASE_GCS_OBJECT).upload_from_string(b"newer")
    _touch(database_file, b"stale changes")

    assert database.persist_database() is False
    assert bucket.objects[database.DATABASE_GCS_OBJECT][0] == b"newer"


def test_startup_changes_wait_for_the_lease(bucket, database_file):
    _touch(database_file, b"migrated")
    held = gcs.acquire_lease(workflows.LEASE_OBJECT, "old revision", 60)

    assert workflows.publish_startup_changes() is False
    assert bucket.objects[database.DATABASE_GCS_OBJECT][0] == b"v1"
    gcs.release_lease(workflows.LEASE_OBJECT, held)
    assert workflows.publish_startup_changes() is True
    assert bucket.objects[database.DATABASE_GCS_OBJECT][0] == b"migrated"
