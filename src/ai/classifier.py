"""One DistilBERT classifier per folder list, loaded on first use.

The "default" list keeps the original model location (CLASSIFIER_MODEL_PATH /
GCS_CLASSIFIER_MODEL_PREFIX); any other list lives under models/classifier-<list>.
"""

import os
from pathlib import Path
from threading import Lock

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.ai.labels import ID2LABEL
from src.ai.model_input import build_input, model_input_format
from src.config import get_config
from src.storage import gcs


DEFAULT_FOLDER_LIST = "default"
MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "./models/distilbert_deployed")
MODEL_GCS_PREFIX = os.getenv("GCS_CLASSIFIER_MODEL_PREFIX", "models/distilbert_deployed")
MAX_LENGTH = 256

_models = {}
_models_lock = Lock()


class ModelMissingError(FileNotFoundError):
    pass


def model_location(folder_list):
    """Local directory and GCS prefix of a folder list's model."""
    if folder_list == DEFAULT_FOLDER_LIST:
        return Path(MODEL_PATH), MODEL_GCS_PREFIX
    return Path(f"./models/classifier-{folder_list}"), f"models/classifier-{folder_list}"


def _model_is_complete(path):
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def _sync_deployed_model(folder_list):
    """Use GCS when configured, seeding it only from a complete local model."""
    local_model, gcs_prefix = model_location(folder_list)
    if gcs.syncs() and gcs.download_directory(gcs_prefix, local_model):
        return
    if not _model_is_complete(local_model):
        raise ModelMissingError(
            f"No classifier model for folder list {folder_list!r} "
            f"(GCS prefix {gcs_prefix!r}, local {str(local_model)!r})"
        )
    if gcs.syncs():
        gcs.upload_directory(local_model, gcs_prefix)


def _labels(folder_list, model):
    id2label = {int(label_id): label for label_id, label in model.config.id2label.items()}
    if all(label.startswith("LABEL_") for label in id2label.values()):
        # The original deployed model was trained without label names.
        if folder_list != DEFAULT_FOLDER_LIST:
            raise ValueError(f"Model for folder list {folder_list!r} has no label names")
        id2label = ID2LABEL

    expected = get_config().labels(folder_list)
    if set(id2label.values()) != expected:
        raise ValueError(
            f"Model for folder list {folder_list!r} predicts {sorted(id2label.values())}, "
            f"but the list defines {sorted(expected)}; a changed list needs a new model"
        )
    return id2label


def _load(folder_list):
    _sync_deployed_model(folder_list)
    local_model, _ = model_location(folder_list)
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        local_model,
        clean_up_tokenization_spaces=True,
    )
    model = DistilBertForSequenceClassification.from_pretrained(local_model)
    model.eval()
    return tokenizer, model, _labels(folder_list, model), model_input_format(model.config)


def _get(folder_list):
    with _models_lock:
        if folder_list not in _models:
            _models[folder_list] = _load(folder_list)
        return _models[folder_list]


def model_available(folder_list):
    try:
        _get(folder_list)
    except ModelMissingError:
        return False
    return True


def ensure_loaded(folder_list=DEFAULT_FOLDER_LIST):
    """Load a list's model, syncing its directory from GCS, before code reads that directory."""
    _get(folder_list)


def reload_model(folder_list=DEFAULT_FOLDER_LIST):
    """Reload a list's model, e.g. after a promoted retrain run."""
    loaded = _load(folder_list)
    with _models_lock:
        _models[folder_list] = loaded


def reload_models():
    """Reload every model this process has loaded, plus the default one; returns the lists."""
    with _models_lock:
        folder_lists = sorted(set(_models) | {DEFAULT_FOLDER_LIST})
    for folder_list in folder_lists:
        reload_model(folder_list)
    return folder_lists


def predict_email(email, folder_list=DEFAULT_FOLDER_LIST):
    """Label and confidence of a mail, built in the input format the model was trained on."""
    input_format = _get(folder_list)[3]
    return predict_text(
        build_input(input_format, email.subject, email.sender, email.text), folder_list
    )


def predict_text(text, folder_list=DEFAULT_FOLDER_LIST):
    """Label and confidence of an already built model input."""
    tokenizer, model, id2label, _ = _get(folder_list)
    inputs = tokenizer(
        text or "",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)
        confidence, label_id = torch.max(probabilities, dim=1)

    return id2label[label_id.item()], float(confidence.item())
