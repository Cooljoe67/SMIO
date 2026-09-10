# src/ai/model.py
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from src.ai.labels import ID2LABEL

MODEL_PATH = "./models/bert-smio"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def classify_email(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id])
    return label, confidence
