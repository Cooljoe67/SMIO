# src/ai/labels.py
LABEL2ID = {
    "delivery": 0,
    "commercial": 1,
    "social": 2,
    "other": 3,
    "tech": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
