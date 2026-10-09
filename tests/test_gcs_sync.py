import importlib.util
from pathlib import Path

import pytest

from src.storage import gcs


@pytest.mark.parametrize("bucket, service, opt_in, expected", [
    ("prod", None, None, False),   # GCS_BUCKET from .env alone: local stays local
    ("prod", "smio", None, True),  # Cloud Run
    ("prod", None, "1", True),     # explicit local opt-in
    ("", "smio", "1", False),      # no bucket
])
def test_only_the_service_or_an_opt_in_syncs(monkeypatch, bucket, service, opt_in, expected):
    monkeypatch.setenv("GCS_BUCKET", bucket)
    for variable, value in (("K_SERVICE", service), ("SMIO_GCS", opt_in)):
        if value:
            monkeypatch.setenv(variable, value)
        else:
            monkeypatch.delenv(variable, raising=False)

    assert gcs.syncs() is expected


def test_a_local_run_keeps_its_own_model(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = Path(__file__).resolve().parents[1] / "src" / "ai" / "classifier.py"
    spec = importlib.util.spec_from_file_location("real_classifier", path)
    classifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classifier)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"LOCAL")
    monkeypatch.setattr(classifier, "model_location", lambda folder_list: (model, "models/x"))
    monkeypatch.setenv("GCS_BUCKET", "prod")
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("SMIO_GCS", raising=False)
    touched = []
    monkeypatch.setattr(gcs, "download_directory", lambda *args: touched.append("download"))
    monkeypatch.setattr(gcs, "upload_directory", lambda *args: touched.append("upload"))

    classifier._sync_deployed_model("x")

    assert touched == []
    assert (model / "model.safetensors").read_bytes() == b"LOCAL"
