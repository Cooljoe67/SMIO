"""Replay-buffer retraining: mix corrected mails with a sample of existing
labeled data (per class) and fine-tune the deployed classifier.

Trigger: called from scripts/run_workflow.py after each processing run.
Retraining only happens once enough new corrections have accumulated.
"""

import json
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.ai import classifier
from src.ai.labels import LABEL2ID
from src.db.models import Email

logger = logging.getLogger(__name__)

MODEL_DIR = Path(classifier.MODEL_PATH)
MAX_LENGTH = classifier.MAX_LENGTH
STATE_FILE = Path("./models/retrain_state.json")
LOG_FILE = Path("./models/retrain_log.jsonl")

MIN_CORRECTIONS = 20
REPLAY_RATIO = 0.9
# Fraction of *new* corrections permanently reserved for eval (never trained on),
# so the hold-out set stays representative of real, evolving mail patterns.
HOLDOUT_RATIO = 0.15
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 16


class _EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_run_id": 0, "last_f1": None}


def _save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _append_log(entry):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry) + "\n")


def _pending_corrections(db):
    """Corrections from manual folder moves not yet used by a retrain run."""
    return (
        db.query(Email)
        .filter(
            Email.true_label.isnot(None),
            Email.classification_source == "imap_folder",
            Email.retrain_batch.is_(None),
            Email.is_eval_holdout.is_(False),
        )
        .all()
    )


def should_retrain(db, min_corrections=MIN_CORRECTIONS):
    corrections = _pending_corrections(db)
    return len(corrections) >= min_corrections, corrections


def _route_new_corrections_to_holdout(db, corrections, holdout_ratio=HOLDOUT_RATIO):
    """Permanently reserve a per-class share of new corrections for eval, rest for training."""
    by_label = {}
    for email in corrections:
        by_label.setdefault(email.true_label, []).append(email)

    holdout_emails = []
    train_emails = []
    for emails in by_label.values():
        shuffled = list(emails)
        random.shuffle(shuffled)
        n_holdout = round(len(shuffled) * holdout_ratio)
        holdout_emails.extend(shuffled[:n_holdout])
        train_emails.extend(shuffled[n_holdout:])

    for email in holdout_emails:
        email.is_eval_holdout = True
    db.commit()

    return train_emails, holdout_emails


def _old_pool_for_label(db, label, exclude_ids):
    query = db.query(Email).filter(
        Email.true_label == label,
        Email.is_eval_holdout.is_(False),
    )
    if exclude_ids:
        query = query.filter(~Email.id.in_(exclude_ids))
    return query.all()


def build_replay_batch(db, corrections, replay_ratio=REPLAY_RATIO):
    """Per class: 90% random old samples + 10% new corrections.

    Classes without corrections are only filled with old samples, sized to
    the average batch of the affected classes, so they aren't forgotten.
    """
    correction_ids = {email.id for email in corrections}
    by_label = {}
    for email in corrections:
        by_label.setdefault(email.true_label, []).append(email)

    dataset = []
    batch_sizes = {}

    for label, new_emails in by_label.items():
        old_pool = _old_pool_for_label(db, label, correction_ids)
        old_target = round(len(new_emails) * replay_ratio / (1 - replay_ratio))
        old_sample = random.sample(old_pool, min(old_target, len(old_pool)))
        dataset.extend(old_sample)
        dataset.extend(new_emails)
        batch_sizes[label] = len(old_sample) + len(new_emails)

    target_size = round(sum(batch_sizes.values()) / len(batch_sizes)) if batch_sizes else 0
    for label in LABEL2ID:
        if label in by_label:
            continue
        old_pool = _old_pool_for_label(db, label, set())
        old_sample = random.sample(old_pool, min(target_size, len(old_pool)))
        if old_sample:
            dataset.extend(old_sample)
        batch_sizes[label] = len(old_sample)

    return dataset, batch_sizes


def _tokenize(tokenizer, texts):
    return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"f1_macro": f1_score(labels, predictions, average="macro")}


def _deployed_model_f1(eval_texts, eval_labels):
    """Score the currently deployed model on the permanent hold-out for a fair comparison."""
    predictions = [LABEL2ID[classifier.predict_text(text)[0]] for text in eval_texts]
    return f1_score(eval_labels, predictions, average="macro")


def _load_holdout(db):
    holdout_emails = db.query(Email).filter(Email.is_eval_holdout.is_(True)).all()
    texts = [email.text or "" for email in holdout_emails]
    labels = [LABEL2ID[email.true_label] for email in holdout_emails]
    return texts, labels


def retrain_if_due(db, min_corrections=MIN_CORRECTIONS):
    due, corrections = should_retrain(db, min_corrections)
    if not due:
        logger.info(
            "Retrain skipped: only %d/%d pending corrections",
            len(corrections),
            min_corrections,
        )
        return {"retrained": False, "pending_corrections": len(corrections)}

    eval_texts, eval_labels = _load_holdout(db)
    if not eval_texts:
        logger.warning(
            "Retrain skipped: no eval hold-out configured yet, "
            "run scripts/bootstrap_eval_holdout.py first"
        )
        return {"retrained": False, "reason": "no_eval_holdout"}

    train_emails, new_holdout_emails = _route_new_corrections_to_holdout(db, corrections)
    logger.info(
        "Routed %d new corrections to permanent eval hold-out, %d into training",
        len(new_holdout_emails), len(train_emails),
    )

    dataset_emails, batch_sizes = build_replay_batch(db, train_emails)
    train_texts = [email.text or "" for email in dataset_emails]
    train_labels = [LABEL2ID[email.true_label] for email in dataset_emails]

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    train_ds = _EmailDataset(_tokenize(tokenizer, train_texts), train_labels)
    eval_ds = _EmailDataset(_tokenize(tokenizer, eval_texts), eval_labels)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate_dir = Path(f"./models/distilbert_candidate_{timestamp}")

    training_args = TrainingArguments(
        output_dir=str(candidate_dir),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    new_f1 = trainer.evaluate()["eval_f1_macro"]
    baseline_f1 = _deployed_model_f1(eval_texts, eval_labels)

    state = _load_state()
    run_id = state.get("last_run_id", 0) + 1
    promoted = new_f1 >= baseline_f1

    candidate_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(candidate_dir))
    tokenizer.save_pretrained(str(candidate_dir))

    if promoted:
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        shutil.copytree(candidate_dir, MODEL_DIR)
        classifier.reload_model()
        logger.info("Retrain run %d promoted: F1 %.3f -> %.3f", run_id, baseline_f1, new_f1)
    else:
        logger.info(
            "Retrain run %d NOT promoted: F1 %.3f -> %.3f (candidate kept at %s)",
            run_id, baseline_f1, new_f1, candidate_dir,
        )

    for email in train_emails:
        if promoted:
            email.retrain_batch = run_id
    db.commit()

    state.update({
        "last_run_id": run_id,
        "last_f1": new_f1 if promoted else state.get("last_f1"),
        "last_run_at": timestamp,
    })
    _save_state(state)

    result = {
        "retrained": True,
        "run_id": run_id,
        "promoted": promoted,
        "baseline_f1": baseline_f1,
        "new_f1": new_f1,
        "batch_sizes": batch_sizes,
        "new_holdout_count": len(new_holdout_emails),
        "eval_holdout_size": len(eval_texts),
        "candidate_dir": str(candidate_dir),
    }
    _append_log(result)
    return result
