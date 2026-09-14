import os
import numpy as np
import torch
import optuna
import csv

from sklearn.model_selection import StratifiedKFold
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

STUDY_NAME = "distilbert_email_classification"
STUDY_STORAGE = "sqlite:///optuna_distilbert.db"  # <-- Persistente Optuna-Study


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

class WeightedDistilBertForSequenceClassification(
    DistilBertForSequenceClassification
):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float)
            if class_weights is not None
            else None
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs,
        )

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=(
                    self.class_weights.to(outputs.logits.device)
                    if self.class_weights is not None
                    else None
                )
            )
            loss = loss_fct(
                outputs.logits.view(-1, self.num_labels), labels.view(-1)
            )

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

        texts = []
        labels = []

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

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def confusion_matrix_per_class(labels_true, labels_pred):
    return confusion_matrix(labels_true, labels_pred, labels=list(LABEL2ID.values()))


# ---------- Optuna Objective ----------

def objective(trial, texts, labels, tokenizer, class_weights):

    num_labels = len(LABEL2ID)

    # Hyperparameter Search Space (leicht gestrafft)
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32])  # größer für Speed
    ml = trial.suggest_categorical("max_length", [128, 256])
    wr = trial.suggest_float("warmup_ratio", 0.05, 0.2)

    # 3-Fold statt 5-Fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):

        train_dataset = EmailDataset(
            [texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            tokenizer,
            max_length=ml,
        )

        val_dataset = EmailDataset(
            [texts[i] for i in val_idx],
            [labels[i] for i in val_idx],
            tokenizer,
            max_length=ml,
        )

        model = WeightedDistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            class_weights=class_weights,
        )

        training_args = TrainingArguments(
            output_dir=f"{BASE_OUTPUT_DIR}/trial_{trial.number}_fold_{fold}",
            num_train_epochs=3,  # statt 5 → schneller
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            learning_rate=lr,
            warmup_ratio=wr,
            weight_decay=0.01,
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="none",
            dataloader_pin_memory=False,
            dataloader_num_workers=4,  # mehr Worker für Speed
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
            raise optuna.TrialPruned("No F1 score returned")

        fold_f1 = float(metrics["eval_f1"])
        fold_f1_scores.append(fold_f1)

        # Zwischenergebnisse pro Fold im Trial speichern
        trial.set_user_attr(f"fold_{fold}_f1", fold_f1)
        trial.set_user_attr(f"fold_{fold}_eval_loss", float(metrics.get("eval_loss", 0.0)))

    avg_f1 = float(np.mean(fold_f1_scores))
    trial.set_user_attr("avg_f1", avg_f1)

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

    # Optuna Study mit SQLite-Persistenz & Resume
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        load_if_exists=True,
    )

    def wrapped_objective(trial):
        return objective(trial, texts, labels_arr, tokenizer, class_weights)

    # Nur 5 Trials, GC nach jedem Trial
    study.optimize(
        wrapped_objective,
        n_trials=5,
        gc_after_trial=True,
    )

    best_trial = study.best_trial
    print("\n=== BEST TRIAL ===")
    print("Value (F1):", best_trial.value)
    print("Params:", best_trial.params)

    # Logging Trials to CSV
    csv_path = os.path.join(BASE_OUTPUT_DIR, "optuna_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "f1", "learning_rate", "batch_size", "max_length", "warmup_ratio"])
        for t in study.trials:
            writer.writerow([
                t.number,
                t.value,
                t.params.get("learning_rate"),
                t.params.get("batch_size"),
                t.params.get("max_length"),
                t.params.get("warmup_ratio"),
            ])

    # Train final model with best params on full data
    num_labels = len(LABEL2ID)
    best_lr = best_trial.params["learning_rate"]
    best_bs = best_trial.params["batch_size"]
    best_ml = best_trial.params["max_length"]
    best_wr = best_trial.params["warmup_ratio"]

    final_dataset = EmailDataset(texts, labels, tokenizer, max_length=best_ml)
    final_model = WeightedDistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        class_weights=class_weights,
    )

    final_args = TrainingArguments(
        output_dir=f"{BASE_OUTPUT_DIR}/final",
        num_train_epochs=3,  # auch hier kürzer
        per_device_train_batch_size=best_bs,
        per_device_eval_batch_size=best_bs,
        learning_rate=best_lr,
        warmup_ratio=best_wr,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=final_dataset,
        eval_dataset=final_dataset,
        compute_metrics=compute_metrics,
    )

    final_trainer.train()

    # Save final model for deployment
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
    all_preds = []
    all_true = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = final_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels_batch.cpu().numpy())

    cm = confusion_matrix_per_class(all_true, all_preds)
    print("\nConfusion Matrix (label indices):")
    print(cm)
    print("\nLabels mapping:", LABEL2ID)


if __name__ == "__main__":
    main()
