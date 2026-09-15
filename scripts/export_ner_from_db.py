import json
import re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from src.db.database import SessionLocal
from src.db.models import Email

LABELS = [
    "O",
    "ORDER_ID",
    "TRACKING_ID",
    "DATE",
    "STATUS_SENT",
    "STATUS_IN_TRANSIT",
    "STATUS_DELIVERED",
    "STATUS_DELAYED",
    "STATUS_FAILED",
    "STATUS_READY_FOR_PICKUP",
    "STATUS_IN_WAREHOUSE"
]
MAX_TOKENS = 200

def clean_text(text):
    if not text:
        return ""

    # Remove HTML completely
    try:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        pass

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def auto_tag(tokens, text, offsets):
    tags = ["O"] * len(tokens)

    status_keywords = {
        r"\bversendet\b": "STATUS_SENT",
        r"\bin\s+zustellung\b": "STATUS_IN_TRANSIT",
        r"\bwird\s+zugestellt\b": "STATUS_IN_TRANSIT",
        r"\bkonnte\s+nicht\s+zugestellt\s+werden\b": "STATUS_FAILED",
        r"\b(?:wurde|ist)\s+zugestellt\b": "STATUS_DELIVERED",
        r"\bgeliefert\b": "STATUS_DELIVERED",
        r"\bverzögert\b": "STATUS_DELAYED",
        r"\bfehlgeschlagen\b": "STATUS_FAILED",
        r"\babholung\b": "STATUS_READY_FOR_PICKUP",
        r"\bpaketzentrum\b": "STATUS_IN_WAREHOUSE",
    }

    def apply_match(match, label, group=0):
        start, end = match.span(group)
        for index, (token_start, token_end) in enumerate(offsets):
            if token_start < end and token_end > start:
                tags[index] = label

    for pattern, label in status_keywords.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            apply_match(match, label)

    tracking_patterns = [
        r"(?<!\w)#\d[\d-]{4,}(?!\w)",
        r"\b1Z[0-9A-Z]{16}\b",
        r"\b[A-Z]{2}[- ]?\d{8,20}(?:[A-Z]{2})?\b",
        r"\b\d{3,}(?:-\d{2,}){1,}\b",
    ]
    for pattern in tracking_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            apply_match(match, "TRACKING_ID")

    date_patterns = (
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    )
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            apply_match(match, "DATE")

    order_pattern = (
        r"\b(?:bestell(?:ung|nummer|nr\.?)|order\s*(?:id|number|nummer))"
        r"\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{3,})\b"
    )
    for match in re.finditer(order_pattern, text, flags=re.IGNORECASE):
        apply_match(match, "ORDER_ID", group=1)

    return tags

def export_ner_data(output_path="ner_training.jsonl"):
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased",
        clean_up_tokenization_spaces=True,
    )
    db = SessionLocal()

    try:
        emails = db.query(Email).filter(Email.classification == "delivery").all()

        with open(output_path, "w", encoding="utf-8") as f:
            for email in emails:
                text = clean_text(email.text)
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=MAX_TOKENS,
                    return_offsets_mapping=True,
                )
                tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
                offsets = encoded["offset_mapping"]
                tags = auto_tag(tokens, text, offsets)

                record = {"tokens": tokens, "tags": tags}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        db.close()

    print("NER training data exported to", output_path)

if __name__ == "__main__":
    export_ner_data()
