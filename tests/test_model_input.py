import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ai import model_input
from src.db.models import Email


def test_body_format_is_the_mail_text():
    assert model_input.build_input("body", "Rechnung 4711", "a@dhl.de", "Hallo") == "Hallo"


def test_subject_and_sender_domain_come_before_the_body():
    built = model_input.build_input(
        "subject_sender_body", "DRINGEND: Schaden", "Dispo <dispo@Spedition.DE>", "Text"
    )
    assert built == "DRINGEND: Schaden\nspedition.de\nText"


def test_models_without_a_stored_format_see_the_body():
    assert model_input.model_input_format(SimpleNamespace()) == "body"
    assert model_input.model_input_format(SimpleNamespace(smio_input="subject_sender_body")) == "subject_sender_body"


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        model_input.build_input("subject_only", "s", "a@b.c", "t")


def test_sorting_builds_the_input_the_model_was_trained_with(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = Path(__file__).resolve().parents[1] / "src" / "ai" / "classifier.py"
    spec = importlib.util.spec_from_file_location("real_classifier", path)
    classifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classifier)
    seen = []
    monkeypatch.setattr(classifier, "_get", lambda folder_list: (None, None, {}, "subject_sender_body"))
    monkeypatch.setattr(classifier, "predict_text", lambda text, folder_list: seen.append(text) or ("x", 1.0))

    classifier.predict_email(Email(subject="CMR fehlt", sender="ops@carrier.de", text="Bitte senden"), "logistics")

    assert seen == ["CMR fehlt\ncarrier.de\nBitte senden"]
