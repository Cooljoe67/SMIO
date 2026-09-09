# scripts/train_bert_classifier.py

from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from src.ai.labels import LABEL2ID, ID2LABEL
from scripts.export_training_data import load_training_data

def prepare_dataset():
    texts, labels = load_training_data()
    label_ids = [LABEL2ID[l] for l in labels]
    return Dataset.from_dict({"text": texts, "label": label_ids})

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

if __name__ == "__main__":
    model_name = "bert-base-uncased"

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    dataset = prepare_dataset()
    dataset = dataset.train_test_split(test_size=0.1)

    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="./models/bert-smio",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_steps=50,
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

    trainer.save_model("./models/bert-smio")
    tokenizer.save_pretrained("./models/bert-smio")
