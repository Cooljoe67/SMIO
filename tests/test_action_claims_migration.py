from sqlalchemy import create_engine, inspect, text

from src.db import database


OLD_TABLE = """
CREATE TABLE mail_event_actions (
    id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, email_id INTEGER NOT NULL,
    action VARCHAR NOT NULL, status VARCHAR NOT NULL, attempts INTEGER NOT NULL,
    error TEXT, updated_at DATETIME NOT NULL, UNIQUE (event_id, action)
)
"""


def test_claims_per_event_become_one_claim_per_mail(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as connection:
        connection.execute(text(OLD_TABLE))
        connection.execute(text(
            "CREATE INDEX ix_mail_event_actions_event_id ON mail_event_actions (event_id)"
        ))
        connection.execute(text(
            "CREATE INDEX ix_mail_event_actions_email_id ON mail_event_actions (email_id)"
        ))
        connection.execute(text("""
            INSERT INTO mail_event_actions VALUES
                (1, 10, 5, 'draft', 'failed', 1, 'overloaded', '2026-10-13 10:00:00'),
                (2, 11, 5, 'draft', 'done', 1, NULL, '2026-10-13 10:01:00'),
                (3, 12, 6, 'draft', 'failed', 3, 'down', '2026-10-13 10:02:00')
        """))
    monkeypatch.setattr(database, "engine", engine)

    assert database.ensure_action_claims_per_mail() is True
    assert database.ensure_action_claims_per_mail() is False

    with engine.connect() as connection:
        rows = connection.execute(text(
            "SELECT email_id, event_id, status FROM mail_event_actions ORDER BY email_id"
        )).all()
    assert [tuple(row) for row in rows] == [(5, 11, "done"), (6, 12, "failed")]
    unique = [c["column_names"] for c in inspect(engine).get_unique_constraints("mail_event_actions")]
    assert sorted(unique[0]) == ["action", "email_id"]
