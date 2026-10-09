from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from scripts import import_training_data
from src.db import database
from src.db.models import Base, Email


class FakeMailBox:
    """Folders of the primary account with the messages they contain."""

    def __init__(self, folders):
        self.folders = folders
        self.current = None
        self.folder = SimpleNamespace(set=self._set)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _set(self, folder):
        if folder not in self.folders:
            raise LookupError(folder)
        self.current = folder

    def fetch(self):
        return self.folders[self.current]


def _message(message_id, subject):
    return SimpleNamespace(
        subject=subject,
        from_="sender@test",
        date=datetime(2026, 10, 1),
        text=subject,
        html="",
        headers={"Message-ID": message_id},
    )


@pytest.fixture
def old_database(monkeypatch, tmp_path):
    """A database from before folder lists, holding one mailbox mail."""
    engine = create_engine(f"sqlite:///{(tmp_path / 'smio.db').as_posix()}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        for column in ("account", "folder_list"):
            connection.execute(text(f"DROP INDEX ix_emails_{column}"))
            connection.execute(text(f"ALTER TABLE emails DROP COLUMN {column}"))
        connection.execute(text(
            "INSERT INTO emails (message_id, subject, processed, is_eval_holdout, "
            "is_training_data) VALUES ('<stored@test>', 'Stored', 1, 0, 0)"
        ))
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(import_training_data, "engine", engine)
    monkeypatch.setattr(import_training_data, "SessionLocal", sessionmaker(bind=engine))
    yield engine
    engine.dispose()


def test_repeated_import_stores_each_new_folder_mail_once_as_training_data(
    old_database, monkeypatch
):
    mailbox = FakeMailBox({
        "INBOX/delivery": [_message("<parcel@test>", "Parcel"), _message("<stored@test>", "Stored")],
        "INBOX/tech": [_message("<update@test>", "Update")],
    })
    monkeypatch.setattr(import_training_data, "open_mailbox", lambda account: mailbox)

    import_training_data.import_training_data()
    import_training_data.import_training_data()

    session = sessionmaker(bind=old_database)()
    imported = session.query(Email).filter(Email.is_training_data.is_(True)).order_by(Email.id).all()
    assert [(email.message_id, email.true_label) for email in imported] == [
        ("<parcel@test>", "delivery"), ("<update@test>", "tech"),
    ]
    # Labelled data belongs to no mailbox and is found by the default list's training.
    assert {(email.account, email.folder_list) for email in imported} == {(None, "default")}
    session.close()
