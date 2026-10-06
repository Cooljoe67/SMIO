import json
import logging
import os
import re
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.storage import gcs

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("NER_MODEL_PATH", "./models/ner-smio")
MODEL_GCS_PREFIX = os.getenv("GCS_NER_MODEL_PREFIX", "models/ner-smio")
EXPECTED_MODEL_REVISION = os.getenv("SMIO_NER_MODEL_REVISION", "")
MAX_LENGTH = 256


def _model_is_complete(path):
    path = Path(path)
    has_weights = any(
        (path / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    has_tokenizer = any(
        (path / name).is_file()
        for name in ("tokenizer.json", "vocab.txt", "spiece.model")
    )
    return (path / "config.json").is_file() and has_weights and has_tokenizer


def _active_model_prefix():
    releases_prefix = f"{MODEL_GCS_PREFIX.rstrip('/')}/releases/"
    if EXPECTED_MODEL_REVISION:
        if not re.fullmatch(r"\d{8}T\d{12}Z", EXPECTED_MODEL_REVISION):
            raise ValueError("SMIO_NER_MODEL_REVISION has an invalid format")
        return f"{releases_prefix}{EXPECTED_MODEL_REVISION}", EXPECTED_MODEL_REVISION

    pointer_name = f"{MODEL_GCS_PREFIX.rstrip('/')}/current.json"
    with tempfile.TemporaryDirectory(prefix="smio-ner-pointer-") as temporary_dir:
        pointer_path = Path(temporary_dir) / "current.json"
        if not gcs.download_file(pointer_name, pointer_path):
            return MODEL_GCS_PREFIX, None

        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    revision = pointer.get("revision")
    prefix = pointer.get("prefix")
    expected_prefix = f"{releases_prefix}{revision}"
    if (
        not isinstance(revision, str)
        or not re.fullmatch(r"\d{8}T\d{12}Z", revision)
        or prefix != expected_prefix
    ):
        raise ValueError(f"Invalid NER model pointer at {pointer_name!r}")
    return prefix, revision


def _sync_model():
    if not gcs.enabled():
        return "local"

    active_prefix, revision = _active_model_prefix()

    if not gcs.download_directory(active_prefix, MODEL_PATH):
        if revision:
            raise FileNotFoundError(
                f"Pinned NER model release {active_prefix!r} is missing from GCS."
            )
        if not _model_is_complete(MODEL_PATH):
            raise FileNotFoundError(
                f"NER model is missing from GCS prefix {active_prefix!r} "
                "and no complete local fallback is available."
            )
        gcs.upload_directory(MODEL_PATH, active_prefix)

    if not _model_is_complete(MODEL_PATH):
        raise FileNotFoundError(
            f"NER model at GCS prefix {active_prefix!r} is incomplete."
        )
    return revision or "legacy"


MODEL_REVISION = _sync_model()
logger.info("Loading NER model revision %s", MODEL_REVISION)
if EXPECTED_MODEL_REVISION and MODEL_REVISION != EXPECTED_MODEL_REVISION:
    logger.warning(
        "Cloud Run expects NER revision %s, but GCS pointer selects %s",
        EXPECTED_MODEL_REVISION,
        MODEL_REVISION,
    )
_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    clean_up_tokenization_spaces=True,
)
_model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
_model.eval()


def extract_entities(text):
    encoded = _tokenizer(
        text or "",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        predictions = _model(**encoded).logits.argmax(dim=-1)[0].tolist()

    entities = []
    for label_id, (start, end) in zip(predictions, offsets):
        if start == end:
            continue
        label = _model.config.id2label.get(label_id, "O")
        if label == "O":
            continue
        entities.append({
            "label": label,
            "text": text[start:end],
            "start": start,
            "end": end,
        })

    rule_entities = _extract_rule_entities(text)
    return _merge_entities(rule_entities + entities, text)


def amazon_sender_status(sender, subject="", text=""):
    sender_text = (sender or "").lower()
    if "@amazon." not in sender_text:
        return None

    message_text = f"{subject or ''} {text or ''}"
    if re.search(r"konnte\s+nicht\s+zugestellt|fehlgeschlagen", message_text, re.I):
        return "FAILED"
    if re.search(r"wurde\s+zugestellt|ist\s+zugestellt|geliefert", message_text, re.I):
        return "DELIVERED"
    if re.search(r"in\s+zustellung|wird\s+zugestellt", message_text, re.I):
        return "IN_TRANSIT"

    local_part = sender_text.split("@", 1)[0]
    if local_part == "shipment-tracking":
        return "IN_TRANSIT"
    if local_part in {"versandbestaetigung", "versandbestätigung"}:
        return "SENT"
    return None


def _extract_rule_entities(text):
    rules = {
        "STATUS_SENT": [r"\bversendet\b"],
        "STATUS_IN_TRANSIT": [
            r"\bin\s+zustellung\b",
            r"\bwird\s+zugestellt\b",
        ],
        "STATUS_DELIVERED": [
            r"\b(?:wurde|ist)\s+zugestellt\b",
            r"\bgeliefert\b",
        ],
        "STATUS_FAILED": [
            r"\bkonnte\s+nicht\s+zugestellt\s+werden\b",
            r"\bfehlgeschlagen\b",
        ],
        "STATUS_DELAYED": [r"\bverzögert\b", r"\bverspätet\b"],
        "STATUS_READY_FOR_PICKUP": [r"\babholung\b", r"\babholbereit\b"],
        "STATUS_IN_WAREHOUSE": [r"\bpaketzentrum\b", r"\bim\s+lager\b"],
    }
    entities = []

    for label, patterns in rules.items():
        for pattern in patterns:
            entities.extend(_matches(text, pattern, label))

    entities.extend(_matches(
        text,
        r"\b(?:bestell(?:ung|nummer|nr\.?)|order\s*(?:id|number|nummer))"
        r"\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{3,})\b",
        "ORDER_ID",
        group=1,
    ))
    for pattern in (
        r"(?<!\w)#\d[\d-]{4,}(?!\w)",
        r"\b1Z[0-9A-Z]{16}\b",
        r"\b[A-Z]{2}[- ]?\d{8,20}(?:[A-Z]{2})?\b",
        r"\b\d{3,}(?:-\d{2,}){1,}\b",
    ):
        entities.extend(_matches(text, pattern, "TRACKING_ID"))

    for pattern in (
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ):
        entities.extend(_matches(text, pattern, "DATE"))

    return entities


def _matches(text, pattern, label, group=0):
    matches = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start, end = match.span(group)
        matches.append({
            "label": label,
            "text": text[start:end],
            "start": start,
            "end": end,
        })
    return matches


def _merge_entities(entities, source_text):
    merged = []
    for entity in sorted(entities, key=lambda item: (item["start"], -item["end"])):
        overlapping = next(
            (
                existing
                for existing in merged
                if entity["start"] < existing["end"]
                and entity["end"] > existing["start"]
            ),
            None,
        )
        if overlapping is not None:
            if overlapping["label"] == entity["label"]:
                overlapping["start"] = min(overlapping["start"], entity["start"])
                overlapping["end"] = max(overlapping["end"], entity["end"])
                overlapping["text"] = source_text[
                    overlapping["start"]:overlapping["end"]
                ]
            continue

        if (
            merged
            and merged[-1]["label"] == entity["label"]
            and entity["start"] <= merged[-1]["end"] + 1
        ):
            merged[-1]["end"] = entity["end"]
            merged[-1]["text"] = source_text[
                merged[-1]["start"]:merged[-1]["end"]
            ]
        else:
            merged.append(entity.copy())
    return merged


def entities_to_fields(entities):
    fields = {
        "order_id": None,
        "tracking_id": None,
        "entity_date": None,
        "entity_name": None,
        "entity_company": None,
        "delivery_status": None,
    }
    for entity in entities:
        label = entity["label"]
        if label == "ORDER_ID" and fields["order_id"] is None:
            fields["order_id"] = entity["text"].strip()
        elif label == "TRACKING_ID" and fields["tracking_id"] is None:
            fields["tracking_id"] = entity["text"].strip()
        elif label == "DATE" and fields["entity_date"] is None:
            fields["entity_date"] = entity["text"].strip()
        elif label == "NAME" and fields["entity_name"] is None:
            fields["entity_name"] = entity["text"].strip()
        elif label == "COMPANY" and fields["entity_company"] is None:
            fields["entity_company"] = entity["text"].strip()
        elif label.startswith("STATUS_") and fields["delivery_status"] is None:
            fields["delivery_status"] = label.removeprefix("STATUS_")
    return fields


def serialize_entities(entities):
    return json.dumps(entities, ensure_ascii=False)
