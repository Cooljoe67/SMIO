from fastapi import APIRouter
from pydantic import BaseModel
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from src.ai.labels import ID2LABEL

router = APIRouter(prefix="/model", tags=["Model"])

# Load model once at startup
MODEL_PATH = "./models/bert-smio"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class EmailInput(BaseModel):
    text: str

@router.post("/classify")
def classify_email(data: EmailInput):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, label_id = torch.max(probs, dim=1)

    label = ID2LABEL[label_id.item()]

    return {
        "label": label,
        "confidence": float(confidence.item())
    }
