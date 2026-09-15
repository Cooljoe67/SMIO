# scripts/train_distilbert_hparam_cv.py

import os
from collections import Counter
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import torch
from torch.utils.data import Dataset

from src.db.database import SessionLocal
from src.db.models import Email
from src.ai.labels import LABEL2ID, ID2LABEL  # {"commercial":0, "delivery":1, ...}

MODEL_BASE = "distilbert-base-uncased"
OUTPUT_DIR = "./models/distilbert_hparam_cv"


# ---------------- Dataset ----------------

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
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
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ---------------- Weighted DistilBERT ----------------

class WeightedDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

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
            labels=None,  # wir setzen Loss selbst
            **kwargs,
        )

        logits = outputs.logits

        if labels is not None and self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs.loss = loss

        return outputs


# ---------------- Daten laden ----------------

def load_data():
    db = SessionLocal()
    emails = db.query(Email).filter(Email.true_label != None).all()

    texts = []
    labels = []

    for e in emails:
        if e.true_label in LABEL2ID:
            texts.append(e.text)
            labels.append(LABEL2ID[e.true_label])

    return texts, labels


# ---------------- Hyperparameter + Cross-Validation ----------------

def train_distilbert_hparam_cv():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_BASE)
    texts, labels = load_data()
    labels_arr = np.array(labels)
    num_labels = len(LABEL2ID)

    # Class Weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_arr),
        y=labels_arr,
    )
    print("Class Weights:", class_weights)

    # Hyperparameter Grid
    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_sizes = [8, 16]
    max_lengths = [128, 256]
    warmup_ratios = [0.1, 0.2]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_f1 = -1.0
    best_config = None

    for lr in learning_rates:
        for bs in batch_sizes:
            for ml in max_lengths:
                for wr in warmup_ratios:
                    print(f"\n=== Config: LR={lr}, BS={bs}, ML={ml}, WR={wr} ===")

                    fold_f1_scores = []

                    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
                        print(f"Fold {fold+1}/5")

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
                            MODEL_BASE,
                            num_labels=num_labels,
                            class_weights=class_weights,
                        )

                        training_args = TrainingArguments(
                            output_dir=f"{OUTPUT_DIR}/lr{lr}_bs{bs}_ml{ml}_wr{wr}_fold{fold}",
                            num_train_epochs=3,
                            per_device_train_batch_size=bs,
                            per_device_eval_batch_size=bs,
                            learning_rate=lr,
                            warmup_ratio=wr,
                            weight_decay=0.01,
                            logging_steps=50,
                            evaluation_strategy="epoch",
                            save_strategy="no",
                            report_to="none",
                        )

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

                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            compute_metrics=compute_metrics,
                        )

                        trainer.train()
                        metrics = trainer.evaluate()
                        fold_f1_scores.append(metrics["f1"])

                    avg_f1 = float(np.mean(fold_f1_scores))
                    print(f"→ Avg F1 for config: {avg_f1:.4f}")

                    if avg_f1 > best_f1:
                        best_f1 = avg_f1
                        best_config = (lr, bs, ml, wr)

    print("\n=== BEST CONFIG FOUND ===")
    print(f"LR={best_config[0]}, BS={best_config[1]}, ML={best_config[2]}, WR={best_config[3]}")
    print(f"Best weighted F1: {best_f1:.4f}")

    # Optional: finales Modell mit bester Config noch einmal auf allen Daten trainieren
    lr, bs, ml, wr = best_config

    print("\nTraining final DistilBERT model on full dataset with best config...")

    final_dataset = EmailDataset(texts, labels, tokenizer, max_length=ml)

    final_model = WeightedDistilBertForSequenceClassification.from_pretrained(
        MODEL_BASE,
        num_labels=num_labels,
        class_weights=class_weights,
    )

    final_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/final_model",
        num_train_epochs=3,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        warmup_ratio=wr,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="no",
        save_strategy="epoch",
        report_to="none",
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=final_dataset,
    )

    final_trainer.train()

    final_model.save_pretrained(f"{OUTPUT_DIR}/best_distilbert")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_distilbert")
    print("Final DistilBERT model saved to ./models/distilbert_hparam_cv/best_distilbert")


if __name__ == "__main__":
    train_distilbert_hparam_cv()
