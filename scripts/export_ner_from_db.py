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
    "STATUS_SENT",
    "STATUS_IN_TRANSIT",
    "STATUS_DELIVERED",
    "STATUS_DELAYED",
    "STATUS_FAILED",
    "STATUS_READY_FOR_PICKUP",
    "STATUS_IN_WAREHOUSE"
]

def clean_text(text):
    if not text:
        return ""

    # Remove HTML completely
    try:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except:
        pass

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def auto_tag(tokens, text):
    tags = ["O"] * len(tokens)

    status_keywords = {
        "versendet": "STATUS_SENT",
        "in zustellung": "STATUS_IN_TRANSIT",
        "zugestellt": "STATUS_DELIVERED",
        "geliefert": "STATUS_DELIVERED",
        "verzögert": "STATUS_DELAYED",
        "fehlgeschlagen": "STATUS_FAILED",
        "abholung": "STATUS_READY_FOR_PICKUP",
        "paketzentrum": "STATUS_IN_WAREHOUSE"
    }

    low_text = text.lower()

    for i, tok in enumerate(tokens):
        low_tok = tok.lower()

        for keyword, label in status_keywords.items():
            if keyword in low_text and keyword in low_tok:
                tags[i] = label

        if tok.startswith("#") and tok[1:].isdigit():
            tags[i] = "TRACKING_ID"

    return tags

def export_ner_data(output_path="ner_training.jsonl"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    db = SessionLocal()

    emails = db.query(Email).filter(Email.classification == "delivery").all()

    with open(output_path, "w", encoding="utf-8") as f:
        for email in emails:
            text = clean_text(email.text)

            tokens = tokenizer.tokenize(text)

            # ULTRA HARD CUT — guaranteed <512 tokens
            MAX_TOKENS = 200
            tokens = tokens[:MAX_TOKENS]

            tags = auto_tag(tokens, text)

            record = {
                "tokens": tokens,
                "tags": tags
            }

            f.write(json.dumps(record) + "\n")

    print("NER training data exported to", output_path)

if __name__ == "__main__":
    export_ner_data()
