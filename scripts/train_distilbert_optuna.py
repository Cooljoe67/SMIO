import os
import numpy as np
import torch
import optuna
import csv

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from torch.utils.data import Dataset
from src.db.database import SessionLocal
from src.db.models import Email
from src.ai.labels import LABEL2ID, ID2LABEL


MODEL_NAME = "distilbert-base-uncased"
BASE_OUTPUT_DIR = "./models/distilbert_optuna"
DEPLOY_DIR = "./models/distilbert_deployed"

# Fixed training configuration for this study.
STUDY_NAME = "distilbert_email_classification_fixed_256"
STUDY_STORAGE = "sqlite:///optuna_distilbert.db"   # persistent Optuna DB
NUM_SPLITS = 1
NUM_TRAIN_EPOCHS = 4
MAX_LENGTH = 256
BATCH_SIZE = 16
WARMUP_RATIO = 0.15
N_TRIALS = 5
TARGET_F1 = 0.89


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

def load_data():
    db = SessionLocal()
    try:
        emails = db.query(Email).filter(Email.true_label != None).all()
        texts, labels = [], []

        for email in emails:
            if email.true_label in LABEL2ID:
                texts.append(email.text)
                labels.append(LABEL2ID[email.true_label])

        return texts, labels
    finally:
        db.close()


# ---------- Metrics ----------

def compute_metrics(eval_pred):
    logits, labels_eval = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_eval, preds, average="weighted"
    )
    acc = accuracy_score(labels_eval, preds)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def confusion_matrix_per_class(labels_true, labels_pred):
    return confusion_matrix(labels_true, labels_pred, labels=list(LABEL2ID.values()))


# ---------- Optuna Objective ----------

def objective(trial, texts, labels, tokenizer, class_weights):

    num_labels = len(LABEL2ID)
    learning_rate = trial.suggest_float("learning_rate", 2e-5, 4e-5, log=True)

    train_idx, val_idx = train_test_split(
        np.arange(len(texts)),
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate([(train_idx, val_idx)]):

        train_dataset = EmailDataset(
            [texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            tokenizer,
            max_length=MAX_LENGTH,
        )

        val_dataset = EmailDataset(
            [texts[i] for i in val_idx],
            [labels[i] for i in val_idx],
            tokenizer,
            max_length=MAX_LENGTH,
        )

        model = WeightedDistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            class_weights=class_weights,
        )

        training_args = TrainingArguments(
            output_dir=f"{BASE_OUTPUT_DIR}/trial_{trial.number}_fold_{fold}",
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=learning_rate,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=0.01,
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="none",
            dataloader_num_workers=4,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        trainer.train()
        metrics = trainer.evaluate()

        if "eval_f1" not in metrics:
            raise optuna.TrialPruned("No eval_f1 score returned")

        f1 = float(metrics["eval_f1"])
        fold_f1_scores.append(f1)

        # Zwischenergebnisse speichern
        trial.set_user_attr(f"fold_{fold}_f1", f1)
        trial.set_user_attr(f"fold_{fold}_eval_loss", float(metrics.get("eval_loss", 0.0)))

    avg_f1 = float(np.mean(fold_f1_scores))
    trial.set_user_attr("avg_f1", avg_f1)

    # Stop the sequential study after reaching the target without discarding the trial.
    if avg_f1 >= TARGET_F1:
        trial.set_user_attr("target_reached", True)
        trial.study.stop()

    return avg_f1


# ---------- Main ----------

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_NAME,
        clean_up_tokenization_spaces=True,
    )
    texts, labels = load_data()
    labels_arr = np.array(labels)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_arr),
        y=labels_arr,
    )

    # Persistente Optuna-Study
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        load_if_exists=True,
    )

    def wrapped_objective(trial):
        return objective(trial, texts, labels_arr, tokenizer, class_weights)

    study.optimize(
        wrapped_objective,
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    best_trial = study.best_trial
    print("\n=== BEST TRIAL ===")
    print("Value (F1):", best_trial.value)
    print("Params:", best_trial.params)

    # CSV Logging
    csv_path = os.path.join(BASE_OUTPUT_DIR, "optuna_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "f1", "learning_rate", "batch_size", "max_length", "warmup_ratio"])
        for t in study.trials:
            writer.writerow([
                t.number,
                t.value,
                t.params.get("learning_rate"),
                BATCH_SIZE,
                MAX_LENGTH,
                WARMUP_RATIO,
            ])

    # Finales Training
    num_labels = len(LABEL2ID)
    best_lr = best_trial.params.get("learning_rate", 3e-5)
    best_bs = BATCH_SIZE

    final_dataset = EmailDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)
    final_model = WeightedDistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        class_weights=class_weights,
    )

    final_args = TrainingArguments(
        output_dir=f"{BASE_OUTPUT_DIR}/final",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=best_bs,
        per_device_eval_batch_size=best_bs,
        learning_rate=best_lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=final_dataset,
        eval_dataset=final_dataset,
        compute_metrics=compute_metrics,
    )

    final_trainer.train()

    # Deployment
    final_model.save_pretrained(DEPLOY_DIR)
    tokenizer.save_pretrained(DEPLOY_DIR)
    print(f"\nDeployed model saved to: {DEPLOY_DIR}")

    # Confusion Matrix
    dataloader = torch.utils.data.DataLoader(
        final_dataset,
        batch_size=best_bs,
        shuffle=False,
    )

    final_model.eval()
    all_preds, all_true = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = final_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs["logits"], dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels_batch.cpu().numpy())

    cm = confusion_matrix_per_class(all_true, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nLabels mapping:", LABEL2ID)


if __name__ == "__main__":
    main()
