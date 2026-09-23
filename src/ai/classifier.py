import os

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.ai.labels import ID2LABEL
from src.storage import gcs


MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "./models/distilbert_deployed")
MODEL_GCS_PREFIX = os.getenv("GCS_CLASSIFIER_MODEL_PREFIX", "models/distilbert_deployed")
MAX_LENGTH = 256


def _sync_deployed_model():
    """Use the GCS model when configured, seeding it from the local model once."""
    if not gcs.enabled():
        return
    if not gcs.download_directory(MODEL_GCS_PREFIX, MODEL_PATH):
        gcs.upload_directory(MODEL_PATH, MODEL_GCS_PREFIX)


_sync_deployed_model()
_tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH,
    clean_up_tokenization_spaces=True,
)
_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
_model.eval()


def reload_model():
    """Reload tokenizer/model from MODEL_PATH, e.g. after a promoted retrain run."""
    global _tokenizer, _model
    _sync_deployed_model()
    _tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_PATH,
        clean_up_tokenization_spaces=True,
    )
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.eval()


def predict_text(text):
    inputs = _tokenizer(
        text or "",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        probabilities = torch.softmax(_model(**inputs).logits, dim=1)
        confidence, label_id = torch.max(probabilities, dim=1)

    return ID2LABEL[label_id.item()], float(confidence.item())
