# scripts/export_training_data.py
from src.db.database import SessionLocal
from src.db.models import Email

def load_training_data():
    db = SessionLocal()
    emails = db.query(Email).filter(Email.classification != None).all()

    texts = [e.text for e in emails]
    labels = [e.classification for e in emails]
    return texts, labels

if __name__ == "__main__":
    texts, labels = load_training_data()
    print(f"Loaded {len(texts)} samples")
