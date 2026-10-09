import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.db.models import Email

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
LIVE_PREFIX = "models/distilbert_deployed"


class ModelMissingError(FileNotFoundError):
    pass


class _Unavailable:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("not available in tests")


def _load_real(monkeypatch, relative_path, **modules):
    """Load a model module file itself (conftest stubs it) on stand-ins for torch/transformers."""
    torch = types.ModuleType("torch")
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.Dataset = object
    transformers = types.ModuleType("transformers")
    for name in (
        "DistilBertForSequenceClassification",
        "DistilBertTokenizerFast",
        "Trainer",
        "TrainingArguments",
    ):
        setattr(transformers, name, type(name, (_Unavailable,), {}))
    modules = {
        "torch": torch,
        "torch.utils": torch.utils,
        "torch.utils.data": torch.utils.data,
        "transformers": transformers,
        **modules,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        f"{Path(relative_path).stem}_under_test", ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBucket:
    """GCS objects in a dict, with the directory calls of src.storage.gcs."""

    def __init__(self):
        self.objects = {}

    def install(self, monkeypatch, gcs):
        monkeypatch.setattr(gcs, "enabled", lambda: True)
        monkeypatch.setenv("K_SERVICE", "smio")  # syncing with GCS is service behaviour
        for name in (
            "download_directory", "upload_directory", "list_directory_prefixes", "delete_directory",
        ):
            monkeypatch.setattr(gcs, name, getattr(self, name))

    def put_model(self, prefix, weights):
        self.objects[f"{prefix}/config.json"] = b"{}"
        self.objects[f"{prefix}/model.safetensors"] = weights

    def _names(self, prefix):
        prefix = prefix.rstrip("/") + "/"
        return [name for name in self.objects if name.startswith(prefix)]

    def download_directory(self, prefix, destination):
        names = self._names(prefix)
        for name in names:
            path = Path(destination, name[len(prefix.rstrip("/")) + 1:])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.objects[name])
        return bool(names)

    def upload_directory(self, source, prefix):
        for path in Path(source).rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(source).as_posix()
                self.objects[f"{prefix.rstrip('/')}/{relative_path}"] = path.read_bytes()
        return True

    def list_directory_prefixes(self, prefix):
        start = len(prefix.rstrip("/")) + 1
        return sorted({name[start:].split("/", 1)[0] for name in self._names(prefix)})

    def delete_directory(self, prefix):
        names = self._names(prefix)
        for name in names:
            del self.objects[name]
        return len(names)


@pytest.fixture
def retrain(monkeypatch, tmp_path):
    """The real src/ai/retrain.py with a classifier whose model directory starts empty."""
    classifier = types.ModuleType("src.ai.classifier")
    classifier.__dict__.update(
        DEFAULT_FOLDER_LIST="default",
        MODEL_PATH=str(tmp_path / "deployed"),
        MODEL_GCS_PREFIX=LIVE_PREFIX,
        MAX_LENGTH=256,
        ensure_loaded=lambda folder_list="default": None,
        reload_model=lambda folder_list="default": None,
    )
    return _load_real(monkeypatch, "src/ai/retrain.py", **{"src.ai.classifier": classifier})


@pytest.fixture
def bucket(monkeypatch):
    from src.storage import gcs

    bucket = FakeBucket()
    bucket.install(monkeypatch, gcs)
    return bucket


def test_ensure_loaded_syncs_the_model_directory_from_gcs(monkeypatch, tmp_path):
    from src.config import get_config
    from src.storage import gcs

    get_config()  # the local config: the fake bucket has none
    bucket = FakeBucket()
    bucket.install(monkeypatch, gcs)
    classifier = _load_real(monkeypatch, "src/ai/classifier.py")
    bucket.put_model(classifier.MODEL_GCS_PREFIX, b"live")
    monkeypatch.setattr(classifier, "MODEL_PATH", str(tmp_path / "deployed"))
    model = SimpleNamespace(
        # The original model has no label names.
        config=SimpleNamespace(id2label={index: f"LABEL_{index}" for index in range(5)}),
        eval=lambda: None,
    )
    monkeypatch.setattr(
        classifier, "DistilBertTokenizerFast",
        SimpleNamespace(from_pretrained=lambda path, **kwargs: "tokenizer"),
    )
    monkeypatch.setattr(
        classifier, "DistilBertForSequenceClassification",
        SimpleNamespace(from_pretrained=lambda path: model),
    )

    classifier.ensure_loaded()

    assert (tmp_path / "deployed" / "model.safetensors").read_bytes() == b"live"
    assert classifier._models["default"][1] is model


def test_retrain_stops_before_routing_corrections_when_the_model_cannot_be_synced(
    db, monkeypatch, retrain
):
    db.add(Email(folder_list="default", true_label="tech", text="old holdout", is_eval_holdout=True))
    db.add_all(
        Email(
            folder_list="default",
            true_label="delivery",
            text=f"moved {index}",
            classification_source="imap_folder",
        )
        for index in range(20)
    )
    db.commit()

    def missing_model(folder_list="default"):
        raise ModelMissingError("no classifier model in GCS")

    monkeypatch.setattr(retrain.classifier, "ensure_loaded", missing_model)

    with pytest.raises(ModelMissingError):
        retrain.retrain_if_due(db, min_corrections=1)

    # No correction was moved to the holdout by the failed run.
    assert db.query(Email).filter(Email.is_eval_holdout.is_(True)).count() == 1


def _recent_archive(retrain, bucket, weights):
    archive_name = retrain._model_archive_name(datetime.now(timezone.utc) - timedelta(hours=1))
    bucket.put_model(f"{retrain.MODEL_ARCHIVE_PREFIX}/{archive_name}", weights)
    return archive_name


def test_restore_refuses_when_the_live_model_is_not_on_disk(monkeypatch, retrain, bucket):
    _recent_archive(retrain, bucket, b"previous")
    bucket.put_model(LIVE_PREFIX, b"live")

    def missing_model(folder_list="default"):
        raise ModelMissingError("GCS download failed")

    monkeypatch.setattr(retrain.classifier, "ensure_loaded", missing_model)

    assert retrain.restore_previous_model() == {
        "restored": False, "reason": "current_model_unavailable",
    }
    assert bucket.objects[f"{LIVE_PREFIX}/model.safetensors"] == b"live"


@pytest.mark.parametrize("load_error", [None, ValueError("labels changed")])
def test_restore_after_a_cold_start_archives_the_live_model_first(
    monkeypatch, retrain, bucket, load_error
):
    archive_name = _recent_archive(retrain, bucket, b"previous")
    bucket.put_model(LIVE_PREFIX, b"live")

    def sync_live_model(folder_list="default"):
        bucket.download_directory(LIVE_PREFIX, retrain.MODEL_DIR)
        # A live model that does not load is still archived and replaced.
        if load_error:
            raise load_error

    monkeypatch.setattr(retrain.classifier, "ensure_loaded", sync_live_model)

    assert retrain.restore_previous_model() == {"restored": True, "archive": archive_name}
    assert bucket.objects[f"{LIVE_PREFIX}/model.safetensors"] == b"previous"
    archived = {
        name: weights
        for name, weights in bucket.objects.items()
        if name.startswith(retrain.MODEL_ARCHIVE_PREFIX) and name.endswith("model.safetensors")
    }
    assert sorted(archived.values()) == [b"live", b"previous"]


def test_forced_retrain_without_corrections_is_a_clean_no_op(db, retrain):
    db.add(Email(folder_list="default", true_label="tech", text="holdout", is_eval_holdout=True))
    db.commit()

    result = retrain.retrain_if_due(db, min_corrections=0)

    assert result == {"retrained": False, "reason": "no_corrections", "pending_corrections": 0}
