"""Tune and train the DistilBERT classifier of one folder list with Optuna.

Each trial is scored by stratified group k-fold cross-validation: mails of the
same business case (case_ref) always stay in one fold, so the score is not
inflated by near-identical mails. The best trial's out-of-fold predictions give
the confusion matrix; the final model is trained on all labelled mails and saved
with its label names to the folder list's model location.

The cross-validated score of the best trial is slightly optimistic, because the
same folds were used to choose the learning rate.

Usage:
    python -m scripts.train_distilbert_optuna --folder-list rhein-cargo [--publish]
"""

import argparse
import csv
import os
import shutil
from collections import Counter

import numpy as np
import optuna
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src.ai.classifier import DEFAULT_FOLDER_LIST, model_location
from src.ai.labels import LABEL2ID as DEFAULT_LABEL2ID
from src.config import get_config
from src.db.database import SessionLocal
from src.db.models import Email
from src.storage import gcs


BASE_MODEL = "distilbert-base-uncased"
BASE_OUTPUT_DIR = "./models/distilbert_optuna"
STUDY_STORAGE = "sqlite:///optuna_distilbert.db"   # persistent Optuna DB

NUM_FOLDS = 5
NUM_TRAIN_EPOCHS = 4
MAX_LENGTH = 256
BATCH_SIZE = 16
WARMUP_RATIO = 0.15
N_TRIALS = 5
TARGET_F1 = 0.89
RANDOM_STATE = 42


# ---------- Dataset ----------

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------- Weighted DistilBERT ----------

class WeightedDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float)
            if class_weights is not None else None
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs,
        )

        loss = None
        if labels is not None:
            labels = labels.to(
                device=outputs.logits.device,
                dtype=torch.long,
            )
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=(self.class_weights.to(outputs.logits.device)
                        if self.class_weights is not None else None)
            )
            loss = loss_fct(outputs.logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------- Data Loading ----------

def label_ids(folder_list):
    """Label -> id. The default list keeps the ids the retraining code expects."""
    if folder_list == DEFAULT_FOLDER_LIST:
        return dict(DEFAULT_LABEL2ID)
    labels = sorted(get_config().folder_lists[folder_list]) + ["other"]
    return {label: index for index, label in enumerate(labels)}


def load_data(db, folder_list, label2id):
    """Texts, label ids and case groups of all mails labelled for the folder list."""
    emails = db.query(Email).filter(
        Email.folder_list == folder_list,
        Email.true_label.isnot(None),
    ).order_by(Email.id).all()
    unknown = Counter(email.true_label for email in emails if email.true_label not in label2id)
    if unknown:
        raise ValueError(f"Labels not in folder list {folder_list!r}: {dict(unknown)}")

    texts = [email.text or "" for email in emails]
    labels = [label2id[email.true_label] for email in emails]
    # Mails without a case are their own group.
    groups = [email.case_ref or f"mail-{email.id}" for email in emails]
    return texts, labels, groups


def make_folds(labels, groups, num_folds):
    smallest = min(Counter(labels).values())
    if smallest < num_folds:
        print(f"Warning: the rarest label has only {smallest} mails for {num_folds} folds.")
    splitter = StratifiedGroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


# ---------- Metrics ----------

def scores(labels_true, labels_pred):
    result = {"accuracy": accuracy_score(labels_true, labels_pred)}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_true, labels_pred, average=average, zero_division=0
        )
        result.update({
            f"{average}_precision": precision,
            f"{average}_recall": recall,
            f"{average}_f1": f1,
        })
    return result


def compute_metrics(eval_pred):
    logits, labels_eval = eval_pred
    return scores(labels_eval, np.argmax(logits, axis=-1))


# ---------- Training ----------

def _training_args(output_dir, learning_rate, num_epochs, evaluate, metric):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch" if evaluate else "no",
        save_strategy="epoch" if evaluate else "no",
        save_total_limit=1,
        load_best_model_at_end=evaluate,
        metric_for_best_model=f"eval_{metric}_f1" if evaluate else None,
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
    )


def _new_model(base_model, label2id, class_weights):
    return WeightedDistilBertForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        id2label={index: label for label, index in label2id.items()},
        label2id=label2id,
        class_weights=class_weights,
    )


def _best_epoch(trainer):
    """Epoch of the checkpoint early stopping kept."""
    best = trainer.state.best_model_checkpoint
    for entry in trainer.state.log_history:
        if "eval_loss" in entry and best and best.endswith(f"checkpoint-{entry['step']}"):
            return int(round(entry["epoch"]))
    return NUM_TRAIN_EPOCHS


def objective(trial, data, tokenizer, class_weights, options):
    texts, labels, folds, label2id = data
    learning_rate = trial.suggest_float("learning_rate", 2e-5, 4e-5, log=True)

    out_of_fold = np.full(len(labels), -1)
    fold_scores = []
    best_epochs = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        output_dir = f"{BASE_OUTPUT_DIR}/{options.folder_list}/trial_{trial.number}_fold_{fold}"
        trainer = Trainer(
            model=_new_model(options.base_model, label2id, class_weights),
            args=_training_args(output_dir, learning_rate, NUM_TRAIN_EPOCHS, True, options.metric),
            train_dataset=EmailDataset(
                [texts[i] for i in train_idx], [labels[i] for i in train_idx], tokenizer, MAX_LENGTH
            ),
            eval_dataset=EmailDataset(
                [texts[i] for i in val_idx], [labels[i] for i in val_idx], tokenizer, MAX_LENGTH
            ),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )
        trainer.train()
        predictions = np.argmax(trainer.predict(trainer.eval_dataset).predictions, axis=-1)
        out_of_fold[val_idx] = predictions
        best_epochs.append(_best_epoch(trainer))
        shutil.rmtree(output_dir, ignore_errors=True)

        fold_score = scores([labels[i] for i in val_idx], predictions)
        fold_scores.append(fold_score[f"{options.metric}_f1"])
        trial.set_user_attr(f"fold_{fold}_f1", fold_scores[-1])

        # Let Optuna stop a clearly worse trial early.
        trial.report(float(np.mean(fold_scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    overall = scores(labels, out_of_fold)
    trial.set_user_attr("mean_fold_f1", float(np.mean(fold_scores)))
    trial.set_user_attr("std_fold_f1", float(np.std(fold_scores)))
    trial.set_user_attr("out_of_fold_scores", {key: float(value) for key, value in overall.items()})
    trial.set_user_attr("out_of_fold_predictions", out_of_fold.tolist())
    trial.set_user_attr("best_epochs", best_epochs)

    mean_f1 = float(np.mean(fold_scores))
    # Stop the sequential study after reaching the target without discarding the trial.
    if options.target_f1 is not None and mean_f1 >= options.target_f1:
        trial.set_user_attr("target_reached", True)
        trial.study.stop()
    return mean_f1


# ---------- Reporting ----------

def report(best_trial, labels, label2id, folder_list):
    names = sorted(label2id, key=label2id.get)
    predictions = best_trial.user_attrs["out_of_fold_predictions"]
    print("\n=== BEST TRIAL ===")
    print("Mean fold F1:", best_trial.value, "+/-", best_trial.user_attrs["std_fold_f1"])
    print("Out-of-fold scores:", best_trial.user_attrs["out_of_fold_scores"])
    print("Params:", best_trial.params, "best epochs:", best_trial.user_attrs["best_epochs"])
    print("\nOut-of-fold classification report:")
    print(classification_report(
        labels, predictions, labels=list(range(len(names))), target_names=names, zero_division=0
    ))
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(names))))
    print("Out-of-fold confusion matrix (rows: true, columns: predicted):")
    print(names)
    print(matrix)

    path = os.path.join(BASE_OUTPUT_DIR, folder_list, "confusion_matrix.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row])


def write_trials(study, folder_list):
    path = os.path.join(BASE_OUTPUT_DIR, folder_list, "optuna_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial", "state", "mean_fold_f1", "std_fold_f1", "oof_macro_f1",
            "oof_weighted_f1", "learning_rate", "best_epochs",
        ])
        for t in study.trials:
            oof = t.user_attrs.get("out_of_fold_scores", {})
            writer.writerow([
                t.number,
                t.state.name,
                t.value,
                t.user_attrs.get("std_fold_f1"),
                oof.get("macro_f1"),
                oof.get("weighted_f1"),
                t.params.get("learning_rate"),
                t.user_attrs.get("best_epochs"),
            ])


# ---------- Main ----------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder-list", default=DEFAULT_FOLDER_LIST)
    parser.add_argument("--folds", type=int, default=NUM_FOLDS)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--metric",
        choices=("macro", "weighted"),
        default="macro",
        help="F1 average to optimise; macro weighs rare folders equally (default)",
    )
    parser.add_argument("--target-f1", type=float, default=TARGET_F1)
    parser.add_argument("--publish", action="store_true", help="upload the final model to GCS")
    return parser.parse_args()


def main():
    options = parse_args()
    label2id = label_ids(options.folder_list)
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, options.folder_list), exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        options.base_model,
        clean_up_tokenization_spaces=True,
    )
    db = SessionLocal()
    try:
        texts, labels, groups = load_data(db, options.folder_list, label2id)
    finally:
        db.close()
    id2label = {index: label for label, index in label2id.items()}
    print(
        f"{len(texts)} labelled mails in {len(set(groups))} case groups:",
        dict(Counter(id2label[label] for label in labels)),
    )
    folds = make_folds(labels, groups, options.folds)
    labels_arr = np.array(labels)

    # Labels without any mail get the largest weight instead of breaking the weighting.
    present = np.unique(labels_arr)
    present_weights = compute_class_weight(class_weight="balanced", classes=present, y=labels_arr)
    class_weights = np.full(len(label2id), present_weights.max())
    class_weights[present] = present_weights

    # One persistent study per folder list, metric, fold count and base model.
    study = optuna.create_study(
        direction="maximize",
        study_name=(
            f"distilbert_{options.folder_list}_{options.metric}_cv{options.folds}_"
            f"{options.base_model.replace('/', '-')}"
        ),
        storage=STUDY_STORAGE,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )
    study.optimize(
        lambda trial: objective(
            trial, (texts, labels, folds, label2id), tokenizer, class_weights, options
        ),
        n_trials=options.trials,
        gc_after_trial=True,
    )

    best_trial = study.best_trial
    write_trials(study, options.folder_list)
    report(best_trial, labels, label2id, options.folder_list)

    # Final training on all labelled mails, as long as early stopping kept on average.
    num_epochs = max(1, int(round(np.median(best_trial.user_attrs["best_epochs"]))))
    final_model = _new_model(options.base_model, label2id, class_weights)
    final_trainer = Trainer(
        model=final_model,
        args=_training_args(
            f"{BASE_OUTPUT_DIR}/{options.folder_list}/final",
            best_trial.params.get("learning_rate", 3e-5),
            num_epochs,
            False,
            options.metric,
        ),
        train_dataset=EmailDataset(texts, labels, tokenizer, MAX_LENGTH),
    )
    final_trainer.train()

    # Saved as a plain classifier; the class weights only matter for training.
    final_model.config.architectures = ["DistilBertForSequenceClassification"]
    deploy_dir, gcs_prefix = model_location(options.folder_list)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(deploy_dir)
    tokenizer.save_pretrained(deploy_dir)
    print(f"\nModel ({num_epochs} epochs) saved to: {deploy_dir}")

    if options.publish:
        if not gcs.upload_directory(deploy_dir, gcs_prefix):
            raise RuntimeError("GCS is not configured (GCS_BUCKET); model not published")
        print(f"Published to GCS prefix {gcs_prefix}; send RELOAD MODEL to use it.")
    else:
        print(
            "Not published: the service loads models from GCS when GCS_BUCKET is set, "
            "so rerun with --publish to deploy."
        )


if __name__ == "__main__":
    main()
