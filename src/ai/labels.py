# src/ai/labels.py
LABEL2ID = {
    "delivery": 0,
    "commercial": 1,
    "social": 2,
    "trash": 3,
    "other": 4,
    "tech": 5,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
