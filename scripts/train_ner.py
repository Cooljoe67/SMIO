from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

LABELS = [
    "O",
    "ORDER_ID",
    "TRACKING_ID",
    "DATE",
    "NAME",
    "COMPANY",
    "STATUS_SENT",
    "STATUS_IN_TRANSIT",
    "STATUS_DELIVERED",
    "STATUS_DELAYED",
    "STATUS_FAILED",
    "STATUS_READY_FOR_PICKUP",
    "STATUS_IN_WAREHOUSE"
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
MAX_LENGTH = 128

def load_ner_data(path="ner_training.jsonl"):
    tokens = []
    tags = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if len(record["tokens"]) != len(record["tags"]):
                raise ValueError(
                    f"Line {line_number}: tokens and tags have different lengths"
                )
            unknown_tags = set(record["tags"]) - set(LABEL2ID)
            if unknown_tags:
                raise ValueError(
                    f"Line {line_number}: unknown labels: {sorted(unknown_tags)}"
                )
            tokens.append(record["tokens"])
            tags.append([LABEL2ID[t] for t in record["tags"]])
    return Dataset.from_dict({"tokens": tokens, "tags": tags})

def tokenize(batch):
    input_ids = []
    attention_masks = []
    labels = []
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_length = MAX_LENGTH - special_tokens

    for token_seq, label_seq in zip(batch["tokens"], batch["tags"]):
        token_ids = tokenizer.convert_tokens_to_ids(token_seq[:content_length])
        label_ids = label_seq[:content_length]
        encoded = tokenizer.prepare_for_model(
            token_ids,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        pad_length = MAX_LENGTH - special_tokens - len(token_ids)
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        labels.append([-100] + label_ids + [-100] * (pad_length + 1))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []
    for prediction, label in zip(predictions, labels):
        for predicted_id, true_id in zip(prediction, label):
            if true_id != -100:
                true_labels.append(true_id)
                true_predictions.append(predicted_id)

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "f1": f1_score(
            true_labels,
            true_predictions,
            average="weighted",
            zero_division=0,
        ),
    }

if __name__ == "__main__":
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        clean_up_tokenization_spaces=True,
    )
    dataset = load_ner_data()
    if len(dataset) < 2:
        raise ValueError(
            "ner_training.jsonl contains fewer than 2 valid examples. "
            "Run export_ner_from_db.py after adding delivery emails."
        )
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    train_ds = dataset["train"].map(
        tokenize,
        batched=True,
        remove_columns=["tokens", "tags"],
    )
    eval_ds = dataset["test"].map(
        tokenize,
        batched=True,
        remove_columns=["tokens", "tags"],
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir="./models/ner-smio",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model("./models/ner-smio")
    tokenizer.save_pretrained("./models/ner-smio")
