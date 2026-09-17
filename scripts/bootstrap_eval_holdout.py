"""One-time script: carve a stratified eval hold-out from existing labeled emails.

Run this once before the first retrain, so retrain_if_due() has a fixed,
per-class baseline to compare F1 against. Safe to re-run: only labeled
emails not already marked as hold-out are considered.
"""

import random

from src.ai.labels import LABEL2ID
from src.db.database import SessionLocal, engine, ensure_email_columns
from src.db.models import Base, Email

HOLDOUT_RATIO = 0.15


def bootstrap_eval_holdout(ratio=HOLDOUT_RATIO):
    Base.metadata.create_all(bind=engine)
    ensure_email_columns()
    db = SessionLocal()
    try:
        marked = 0
        for label in LABEL2ID:
            candidates = (
                db.query(Email)
                .filter(Email.true_label == label, Email.is_eval_holdout.is_(False))
                .all()
            )
            n_holdout = round(len(candidates) * ratio)
            for email in random.sample(candidates, min(n_holdout, len(candidates))):
                email.is_eval_holdout = True
                marked += 1
        db.commit()
        print(f"Marked {marked} emails as permanent eval hold-out.")
    finally:
        db.close()


if __name__ == "__main__":
    bootstrap_eval_holdout()
