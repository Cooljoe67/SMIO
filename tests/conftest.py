import os
import sys
import types
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep tests offline: no GCS, a throwaway SQLite file, dummy IMAP credentials.
os.environ.pop("GCS_BUCKET", None)
os.environ.setdefault("SQLITE_DB_PATH", str(ROOT / "tmp" / "test-smio.db"))
os.environ.setdefault("IMAP_HOST", "imap.test")
os.environ.setdefault("IMAP_USER", "owner@test")
os.environ.setdefault("IMAP_PASSWORD", "secret")


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)


# Model modules pull in torch/transformers; the inbox logic only needs their names.
_stub("src.ai.classifier", predict_text=lambda text: ("other", 0.5), reload_model=lambda: None)
_stub(
    "src.ai.ner",
    amazon_sender_status=lambda *args: None,
    serialize_entities=lambda entities: "[]",
)
_stub(
    "src.ai.email_metadata",
    extract_email_metadata=lambda *args: ({}, [], {}),
    serialize_sources=lambda sources: "{}",
)
_stub(
    "src.ai.retrain",
    restore_previous_model=lambda: {},
    retrain_if_due=lambda db, **kwargs: {},
)


@pytest.fixture
def db():
    from src.db.models import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
