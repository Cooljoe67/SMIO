# scripts/export_training_data.py

import json
from src.db.database import SessionLocal
from src.db.models import Email

VALID_LABELS = {
    "delivery",
    "trash",
    "social",
    "other",
    "commercial",
    "tech",
}

def export_training_data(output_file="training_data.jsonl"):
    db = SessionLocal()
    emails = db.query(Email).all()

    count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for email in emails:
            if email.classification not in VALID_LABELS:
                continue

            record = {
                "text": email.text or email.html or "",
                "label": email.classification,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Export completed. {count} samples written to {output_file}")


def load_training_data(input_file="training_data.jsonl"):
    texts = []
    labels = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(record["label"])

    return texts, labels
    
if __name__ == "__main__":
    export_training_data()
