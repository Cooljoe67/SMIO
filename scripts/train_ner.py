from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import json

# Erweiterte NER-Labels inklusive Lieferstatus
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

def load_ner_data(path="ner_training.jsonl"):
    tokens = []
    tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            tokens.append(record["tokens"])
            tags.append([LABEL2ID[t] for t in record["tags"]])

    return Dataset.from_dict({"tokens": tokens, "tags": tags})

def tokenize(batch):
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    labels = []
    for i, label_seq in enumerate(batch["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(label_seq[word_id])
        labels.append(aligned_labels)

    tokenized["labels"] = labels
    return tokenized

if __name__ == "__main__":
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    dataset = load_ner_data()
    dataset = dataset.train_test_split(test_size=0.1)

    train_ds = dataset["train"].map(tokenize, batched=True)
    eval_ds = dataset["test"].map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="./models/ner-smio",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model("./models/ner-smio")
    tokenizer.save_pretrained("./models/ner-smio")
