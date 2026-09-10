# scripts/test_classifier.py

from src.db.database import SessionLocal
from src.db.models import Email
from src.ai.model import classify_email   

def test_classifier():
    db = SessionLocal()

    emails = db.query(Email).filter(Email.true_label != None).all()

    correct = 0
    total = len(emails)

    for email in emails:
        pred_label, confidence = classify_email(email.text)

        # store prediction
        email.classification = pred_label
        email.confidence = confidence

        if pred_label == email.true_label:
            correct += 1

    db.commit()

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    test_classifier()
