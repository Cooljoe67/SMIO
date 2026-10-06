"""Replay-buffer retraining: mix corrected mails with a sample of existing
labeled data (per class) and fine-tune the deployed classifier.

Trigger: called from scripts/run_workflow.py after each processing run.
Retraining only happens once enough new corrections have accumulated.
"""

import json
import logging
import os
import random
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.ai import classifier
from src.ai.labels import ID2LABEL, LABEL2ID
from src.db.database import record_retrain_run
from src.db.models import Email
from src.storage import gcs

logger = logging.getLogger(__name__)

MODEL_DIR = Path(classifier.MODEL_PATH)
MAX_LENGTH = classifier.MAX_LENGTH
STATE_FILE = Path("./models/retrain_state.json")
LOG_FILE = Path("./models/retrain_log.jsonl")
MODEL_ARCHIVE_PREFIX = f"{classifier.MODEL_GCS_PREFIX.rstrip('/')}_history"
MODEL_ARCHIVE_RETENTION = timedelta(days=7)

MIN_CORRECTIONS = int(os.getenv("SMIO_MIN_CORRECTIONS", "20"))
REPLAY_RATIO = 0.9
# Fraction of *new* corrections permanently reserved for eval (never trained on),
# so the hold-out set stays representative of real, evolving mail patterns.
HOLDOUT_RATIO = 0.15
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 16
PROMOTION_BOOTSTRAP_SAMPLES = 1000
PROMOTION_F1_NONINFERIORITY_MARGIN = 0.02
PROMOTION_MAX_CLASS_RECALL_DROP = 0.10
PROMOTION_MIN_CLASS_SUPPORT = 10


def _model_archive_name(timestamp=None):
    return (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S_%f")


def _parse_model_archive_time(archive_name):
    try:
        return datetime.strptime(archive_name, "%Y%m%d_%H%M%S_%f").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _model_is_complete(path):
    path = Path(path)
    return (path / "config.json").is_file() and any(
        (path / weight_name).is_file()
        for weight_name in ("model.safetensors", "pytorch_model.bin")
    )


def _archive_current_model():
    if not gcs.enabled() or not _model_is_complete(MODEL_DIR):
        return None

    archive_name = _model_archive_name()
    archive_prefix = f"{MODEL_ARCHIVE_PREFIX}/{archive_name}"
    if not gcs.upload_directory(MODEL_DIR, archive_prefix):
        raise RuntimeError(f"Could not archive deployed model to {archive_prefix}")
    return archive_name


def _prune_model_archives(now=None):
    if not gcs.enabled():
        return 0

    now = now or datetime.now(timezone.utc)
    cutoff = now - MODEL_ARCHIVE_RETENTION
    removed = 0
    for archive_name in gcs.list_directory_prefixes(MODEL_ARCHIVE_PREFIX):
        archive_time = _parse_model_archive_time(archive_name)
        if archive_time is not None and archive_time < cutoff:
            removed += gcs.delete_directory(
                f"{MODEL_ARCHIVE_PREFIX}/{archive_name}"
            )
    return removed


def restore_previous_model():
    """Restore the newest complete deployed-model archive from the last week."""
    if not gcs.enabled():
        return {"restored": False, "reason": "gcs_disabled"}

    now = datetime.now(timezone.utc)
    _prune_model_archives(now)
    cutoff = now - MODEL_ARCHIVE_RETENTION
    available_archives = sorted(
        (
            archive_name
            for archive_name in gcs.list_directory_prefixes(MODEL_ARCHIVE_PREFIX)
            if (archive_time := _parse_model_archive_time(archive_name)) is not None
            and cutoff <= archive_time <= now
        ),
        reverse=True,
    )
    if not available_archives:
        return {"restored": False, "reason": "no_recent_model_archive"}

    with tempfile.TemporaryDirectory(prefix="smio-model-restore-") as temporary_dir:
        temporary_dir = Path(temporary_dir)
        for archive_name in available_archives:
            archived_model = temporary_dir / "archived-model"
            if not gcs.download_directory(
                f"{MODEL_ARCHIVE_PREFIX}/{archive_name}", archived_model
            ) or not _model_is_complete(archived_model):
                continue

            current_model = temporary_dir / "current-model"
            if _model_is_complete(MODEL_DIR):
                shutil.copytree(MODEL_DIR, current_model)
                if _archive_current_model() is None:
                    raise RuntimeError("Could not preserve the currently deployed model")

            try:
                if not gcs.upload_directory(archived_model, classifier.MODEL_GCS_PREFIX):
                    raise RuntimeError("Could not upload the restored model to GCS")
                classifier.reload_model()
            except Exception:
                if current_model.exists():
                    try:
                        gcs.upload_directory(current_model, classifier.MODEL_GCS_PREFIX)
                        classifier.reload_model()
                    except Exception:
                        logger.exception("Could not roll back a failed model restore")
                raise

            return {"restored": True, "archive": archive_name}

    return {"restored": False, "reason": "no_complete_model_archive"}


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
    return {
        "accuracy": accuracy_score(labels, predictions),
        "recall_macro": recall_score(
            labels,
            predictions,
            labels=sorted(ID2LABEL),
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            labels,
            predictions,
            labels=sorted(ID2LABEL),
            average="macro",
            zero_division=0,
        ),
    }


def _classification_metrics(labels, predictions):
    label_ids = sorted(ID2LABEL)
    per_class_recall = recall_score(
        labels,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "recall_macro": recall_score(
            labels,
            predictions,
            labels=label_ids,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            labels,
            predictions,
            labels=label_ids,
            average="macro",
            zero_division=0,
        ),
        "recall_by_class": {
            ID2LABEL[label_id]: float(recall)
            for label_id, recall in zip(label_ids, per_class_recall)
        },
    }


def _deployed_model_predictions(eval_texts):
    """Predict the fixed holdout with the currently deployed classifier."""
    return np.asarray(
        [LABEL2ID[classifier.predict_text(text)[0]] for text in eval_texts],
        dtype=np.int64,
    )


def _paired_bootstrap_f1_delta(labels, baseline_predictions, candidate_predictions):
    """Return a stratified paired-bootstrap 95% interval for candidate minus baseline F1."""
    labels = np.asarray(labels, dtype=np.int64)
    baseline_predictions = np.asarray(baseline_predictions, dtype=np.int64)
    candidate_predictions = np.asarray(candidate_predictions, dtype=np.int64)
    label_ids = sorted(ID2LABEL)
    class_indices = [np.flatnonzero(labels == label_id) for label_id in label_ids]
    class_indices = [indices for indices in class_indices if len(indices)]
    if not class_indices:
        raise ValueError("Evaluation holdout contains no recognized labels")

    random_generator = np.random.default_rng(42)
    deltas = np.empty(PROMOTION_BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(PROMOTION_BOOTSTRAP_SAMPLES):
        sampled_indices = np.concatenate([
            random_generator.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        sampled_labels = labels[sampled_indices]
        baseline_f1 = f1_score(
            sampled_labels,
            baseline_predictions[sampled_indices],
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
        candidate_f1 = f1_score(
            sampled_labels,
            candidate_predictions[sampled_indices],
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
        deltas[sample_index] = candidate_f1 - baseline_f1

    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return float(lower), float(upper)


def _promotion_assessment(labels, baseline_predictions, candidate_predictions):
    baseline_metrics = _classification_metrics(labels, baseline_predictions)
    candidate_metrics = _classification_metrics(labels, candidate_predictions)
    ci_lower, ci_upper = _paired_bootstrap_f1_delta(
        labels,
        baseline_predictions,
        candidate_predictions,
    )

    labels = np.asarray(labels, dtype=np.int64)
    class_support = {
        ID2LABEL[label_id]: int(np.count_nonzero(labels == label_id))
        for label_id in sorted(ID2LABEL)
    }
    class_recall_deltas = {
        label: candidate_metrics["recall_by_class"][label]
        - baseline_metrics["recall_by_class"][label]
        for label in class_support
    }
    recall_regressions = {
        label: delta
        for label, delta in class_recall_deltas.items()
        if class_support[label] >= PROMOTION_MIN_CLASS_SUPPORT
        and delta < -PROMOTION_MAX_CLASS_RECALL_DROP
    }
    f1_delta = candidate_metrics["f1_macro"] - baseline_metrics["f1_macro"]
    criteria = {
        "macro_f1_not_worse": f1_delta >= 0,
        "bootstrap_noninferior": ci_lower >= -PROMOTION_F1_NONINFERIORITY_MARGIN,
        "class_recall_guard": not recall_regressions,
    }
    return {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "f1_delta": float(f1_delta),
        "f1_delta_ci95_lower": ci_lower,
        "f1_delta_ci95_upper": ci_upper,
        "class_support": class_support,
        "class_recall_deltas": class_recall_deltas,
        "per_class_recall": {
            label: {
                "support": class_support[label],
                "baseline": baseline_metrics["recall_by_class"][label],
                "candidate": candidate_metrics["recall_by_class"][label],
                "delta": class_recall_deltas[label],
                "guarded": class_support[label] >= PROMOTION_MIN_CLASS_SUPPORT,
            }
            for label in class_support
        },
        "recall_regressions": recall_regressions,
        "promotion_criteria": criteria,
        "promoted": all(criteria.values()),
    }


def _load_holdout(db):
    holdout_emails = db.query(Email).filter(Email.is_eval_holdout.is_(True)).all()
    texts = [email.text or "" for email in holdout_emails]
    labels = [LABEL2ID[email.true_label] for email in holdout_emails]
    return texts, labels


def _append_retraining_result_notice(result):
    promoted = bool(result.get("promoted"))
    promotion_text = "Yes" if promoted else "No; the deployed model was kept"
    subject = (
        "SMIO retraining complete: model promoted"
        if promoted
        else "SMIO retraining complete: model not promoted"
    )
    metrics = (
        ("Accuracy", "baseline_accuracy", "new_accuracy"),
        ("Macro recall", "baseline_recall_macro", "new_recall_macro"),
        ("Macro F1", "baseline_f1", "new_f1"),
    )
    metric_lines = [
        f"- {label}: {result.get(before_key, 0):.3f} -> {result.get(after_key, 0):.3f}"
        for label, before_key, after_key in metrics
    ]
    assessment = result.get("promotion_assessment") or {}
    ci_lower = assessment.get("f1_delta_ci95_lower")
    ci_upper = assessment.get("f1_delta_ci95_upper")
    if ci_lower is None or ci_upper is None:
        f1_interval_text = "not available"
    else:
        f1_interval_text = f"{ci_lower:+.3f} to {ci_upper:+.3f}"
    per_class_recall = assessment.get("per_class_recall") or {}
    recall_lines = [
        f"- {label}: {values['baseline']:.3f} -> {values['candidate']:.3f} "
        f"(n={values['support']}, delta={values['delta']:+.3f})"
        for label, values in sorted(per_class_recall.items())
    ]
    duration = max(float(result.get("training_duration") or 0), 0)
    duration_text = f"{duration / 60:.1f} minutes"
    run_text = str(result.get("run_at", "unknown"))
    body = "\n".join([
        "SMIO retraining completed.",
        f"Run: {run_text}",
        f"Model promoted: {promotion_text}",
        *metric_lines,
        f"Macro F1 delta 95% paired-bootstrap CI: {f1_interval_text}",
        "Promotion gate: macro F1 not lower; CI lower bound >= -0.020; "
        "for classes with at least 10 holdout examples, recall drop <= 0.100.",
        "Per-class recall (baseline -> candidate):",
        *recall_lines,
        f"Training duration: {duration_text}",
    ])
    rows = "".join(
        "<tr>"
        f"<th align=\"left\">{escape(label)}</th>"
        f"<td>{result.get(before_key, 0):.3f}</td>"
        f"<td>{result.get(after_key, 0):.3f}</td>"
        "</tr>"
        for label, before_key, after_key in metrics
    )
    recall_rows = "".join(
        "<tr>"
        f"<th align=\"left\">{escape(label)}</th>"
        f"<td>{values['support']}</td>"
        f"<td>{values['baseline']:.3f}</td>"
        f"<td>{values['candidate']:.3f}</td>"
        f"<td>{values['delta']:+.3f}</td>"
        "</tr>"
        for label, values in sorted(per_class_recall.items())
    )
    html_body = (
        "<html><body><h2>SMIO retraining completed</h2>"
        f"<p>Run: {escape(run_text)}<br>Model promoted: "
        f"<b>{escape(promotion_text)}</b><br>Training duration: {duration_text}<br>"
        f"Macro F1 delta 95% paired-bootstrap CI: {escape(f1_interval_text)}</p>"
        "<p>Promotion gate: macro F1 not lower; CI lower bound &ge; -0.020; "
        "for classes with at least 10 holdout examples, recall drop &le; 0.100.</p>"
        "<table style=\"border-collapse:collapse\"><tr>"
        "<th align=\"left\">Metric</th><th>Baseline</th><th>Candidate</th>"
        f"</tr>{rows}</table>"
        "<h3>Per-class recall</h3>"
        "<table style=\"border-collapse:collapse\"><tr>"
        "<th align=\"left\">Class</th><th>Holdout n</th><th>Baseline</th>"
        "<th>Candidate</th><th>Delta</th>"
        f"</tr>{recall_rows}</table></body></html>"
    )

    try:
        from src.api.utils.mailer import append_summary_to_inbox

        if not append_summary_to_inbox(body, html_body=html_body, subject=subject):
            logger.error("Could not append retraining result notice")
    except Exception:
        logger.exception("Could not append retraining result notice")


def retrain_if_due(db, min_corrections=MIN_CORRECTIONS):
    try:
        removed_archives = _prune_model_archives()
        if removed_archives:
            logger.info("Removed %d expired model archive objects", removed_archives)
    except Exception:
        logger.exception("Could not prune expired model archives")

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
        eval_strategy="epoch",
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
    training_started_at = time.perf_counter()
    trainer.train()
    evaluation = trainer.predict(eval_ds)
    training_duration = time.perf_counter() - training_started_at
    candidate_predictions = np.argmax(evaluation.predictions, axis=1)
    baseline_predictions = _deployed_model_predictions(eval_texts)
    promotion_assessment = _promotion_assessment(
        eval_labels,
        baseline_predictions,
        candidate_predictions,
    )
    baseline_metrics = promotion_assessment["baseline_metrics"]
    new_metrics = promotion_assessment["candidate_metrics"]

    state = _load_state()
    run_id = state.get("last_run_id", 0) + 1
    promoted = promotion_assessment["promoted"]

    candidate_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(candidate_dir))
    tokenizer.save_pretrained(str(candidate_dir))

    replaced_model_archive = None
    if promoted:
        try:
            replaced_model_archive = _archive_current_model()
            if gcs.enabled() and replaced_model_archive is None:
                raise RuntimeError("Could not archive the currently deployed model")
        except Exception:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            raise
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        shutil.copytree(candidate_dir, MODEL_DIR)
        gcs.upload_directory(MODEL_DIR, classifier.MODEL_GCS_PREFIX)
        classifier.reload_model()
        logger.info(
            "Retrain run %d promoted: F1 %.3f -> %.3f",
            run_id,
            baseline_metrics["f1_macro"],
            new_metrics["f1_macro"],
        )
    else:
        logger.info(
            "Retrain run %d NOT promoted: F1 %.3f -> %.3f; candidate discarded",
            run_id,
            baseline_metrics["f1_macro"],
            new_metrics["f1_macro"],
        )
        shutil.rmtree(candidate_dir, ignore_errors=True)

    for email in train_emails:
        if promoted:
            email.retrain_batch = run_id
    db.commit()

    state.update({
        "last_run_id": run_id,
        "last_f1": new_metrics["f1_macro"] if promoted else state.get("last_f1"),
        "last_run_at": timestamp,
        "last_metrics": new_metrics,
    })
    if promoted:
        state["last_promoted_at"] = timestamp
    _save_state(state)

    result = {
        "retrained": True,
        "run_at": timestamp,
        "run_id": run_id,
        "promoted": promoted,
        "promotion_assessment": {
            key: value
            for key, value in promotion_assessment.items()
            if key not in {"baseline_metrics", "candidate_metrics", "promoted"}
        },
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_recall_macro": baseline_metrics["recall_macro"],
        "baseline_f1": baseline_metrics["f1_macro"],
        "new_accuracy": new_metrics["accuracy"],
        "new_recall_macro": new_metrics["recall_macro"],
        "new_f1": new_metrics["f1_macro"],
        "training_duration": training_duration,
        "batch_sizes": batch_sizes,
        "new_holdout_count": len(new_holdout_emails),
        "eval_holdout_size": len(eval_texts),
        "candidate_dir": str(candidate_dir) if promoted else None,
        "replaced_model_archive": replaced_model_archive,
    }
    _append_log(result)
    record_retrain_run(db, result)
    db.commit()
    _append_retraining_result_notice(result)
    return result
