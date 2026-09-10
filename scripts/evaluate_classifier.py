# scripts/evaluate_classifier.py

from src.db.database import SessionLocal
from src.db.models import Email
from src.ai.model import classify_email

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_classifier():
    db = SessionLocal()

    emails = db.query(Email).filter(Email.true_label != None).all()

    y_true = []
    y_pred = []

    for email in emails:
        pred_label, confidence = classify_email(email.text)

        email.classification = pred_label
        email.confidence = confidence

        y_true.append(email.true_label)
        y_pred.append(pred_label)

    db.commit()

    # Print precision, recall, F1
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n=== Confusion Matrix ===")
    print("Labels:", labels)
    print(cm)

    # --- Plot confusion matrix ---
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix — SMIO Classifier")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_classifier()
