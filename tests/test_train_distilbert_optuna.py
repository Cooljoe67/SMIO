import importlib
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.db import database
from src.db.models import Base, Email

pytest.importorskip("sklearn")
pytest.importorskip("optuna")


def _stub_ml_modules():
    """Let the script import without torch/transformers; only data code is tested."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return
    except ImportError:
        pass

    class Placeholder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("not available in tests")

    torch = types.ModuleType("torch")
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.Dataset = object
    transformers = types.ModuleType("transformers")
    for name in (
        "DistilBertForSequenceClassification",
        "DistilBertTokenizerFast",
        "EarlyStoppingCallback",
        "Trainer",
        "TrainingArguments",
    ):
        setattr(transformers, name, type(name, (Placeholder,), {}))
    outputs = types.ModuleType("transformers.modeling_outputs")
    outputs.SequenceClassifierOutput = dict
    sys.modules.update({
        "torch": torch,
        "torch.utils": torch.utils,
        "torch.utils.data": torch.utils.data,
        "transformers": transformers,
        "transformers.modeling_outputs": outputs,
    })


@pytest.fixture(scope="module")
def script():
    saved = dict(sys.modules)
    _stub_ml_modules()
    try:
        yield importlib.import_module("scripts.train_distilbert_optuna")
    finally:
        for name in set(sys.modules) - set(saved):
            if name.split(".")[0] in {"torch", "transformers", "scripts"}:
                sys.modules.pop(name)


def test_label_ids(script):
    assert script.label_ids("default") == {
        "delivery": 0, "commercial": 1, "social": 2, "other": 3, "tech": 4,
    }
    ids = script.label_ids("logistics")
    assert ids["other"] == len(ids) - 1
    assert sorted(ids.values()) == list(range(11))


def test_load_data_only_uses_the_folder_lists_labels(db, script):
    db.add_all([
        Email(folder_list="logistics", true_label="dokumente", text="a", case_ref="VG:1"),
        Email(folder_list="logistics", true_label="other", text="b"),
        Email(folder_list="logistics", true_label=None, text="unlabelled"),
        Email(folder_list="default", true_label="other", text="other list"),
    ])
    db.commit()
    ids = script.label_ids("logistics")

    texts, labels, groups, _ = script.load_data(db, "logistics", ids)

    assert texts == ["a", "b"]
    assert labels == [ids["dokumente"], ids["other"]]
    assert groups[0] == "VG:1" and groups[1].startswith("mail-")


def test_load_data_rejects_labels_outside_the_list(db, script):
    db.add(Email(folder_list="logistics", true_label="delivery", text="a"))
    db.commit()

    with pytest.raises(ValueError, match="delivery"):
        script.load_data(db, "logistics", script.label_ids("logistics"))


def test_folds_keep_each_case_in_one_fold_and_cover_every_mail(script):
    labels = [0, 1] * 20
    groups = [f"case-{i // 4}" for i in range(40)]

    folds = script.make_folds(labels, groups, 5)

    assert len(folds) == 5
    validated = sorted(i for _, val_idx in folds for i in val_idx)
    assert validated == list(range(40))
    for train_idx, val_idx in folds:
        assert not {groups[i] for i in train_idx} & {groups[i] for i in val_idx}


def test_scores_report_macro_and_weighted_f1(script):
    result = script.scores([0, 0, 0, 1], [0, 0, 0, 0])

    assert result["accuracy"] == 0.75
    assert result["macro_f1"] < result["weighted_f1"]


DEFAULT_LABELS = ["delivery", "commercial", "social", "other", "tech"]


class FakeTokenizer:
    def save_pretrained(self, path):
        Path(path, "tokenizer.json").write_text("{}")


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace()

    def save_pretrained(self, path):
        Path(path, "config.json").write_text("{}")


@pytest.fixture
def training(script, monkeypatch, tmp_path):
    """main() on its own database and study storage, with DistilBERT replaced by stand-ins."""
    engine = create_engine(f"sqlite:///{(tmp_path / 'smio.db').as_posix()}")
    Base.metadata.create_all(engine)
    run = SimpleNamespace(
        engine=engine, output_dir=tmp_path / "output", score=0.9, cv_texts=[], final_trainers=[],
    )

    def objective(trial, data, tokenizer, class_weights, options):
        texts, labels, folds, label2id = data
        run.cv_texts.append(list(texts))
        trial.suggest_float("learning_rate", 2e-5, 4e-5, log=True)
        trial.set_user_attr("std_fold_f1", 0.0)
        trial.set_user_attr("out_of_fold_scores", {})
        trial.set_user_attr("out_of_fold_predictions", list(labels))
        trial.set_user_attr("best_epochs", [2])
        return run.score

    class FakeTrainer:
        def __init__(self, model, args, train_dataset):
            self.train_dataset = train_dataset
            run.final_trainers.append(self)

        def train(self):
            pass

        def predict(self, dataset):
            # Right on every mail, so the holdout scores are known.
            return SimpleNamespace(predictions=np.eye(max(dataset.labels) + 1)[dataset.labels])

    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(script, "engine", engine)
    monkeypatch.setattr(script, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(script, "BASE_OUTPUT_DIR", str(run.output_dir))
    monkeypatch.setattr(script, "STUDY_STORAGE", f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}")
    monkeypatch.setattr(
        script, "model_location", lambda folder_list: (tmp_path / "deployed", "models/deployed")
    )
    monkeypatch.setattr(
        script, "DistilBertTokenizerFast",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
    )
    monkeypatch.setattr(script, "TrainingArguments", lambda **kwargs: kwargs)
    monkeypatch.setattr(script, "_new_model", lambda *args: FakeModel())
    monkeypatch.setattr(script, "Trainer", FakeTrainer)
    monkeypatch.setattr(script, "objective", objective)

    def main(*args):
        monkeypatch.setattr(sys, "argv", ["train", "--folds", "2", "--trials", "1", *args])
        script.main()

    run.main = main
    yield run
    engine.dispose()


def _add_mails(engine, count, holdout=False):
    session = sessionmaker(bind=engine)()
    session.add_all(
        Email(
            folder_list="default",
            true_label=DEFAULT_LABELS[index % len(DEFAULT_LABELS)],
            text=f"{'holdout' if holdout else 'training'} mail {index}",
            is_eval_holdout=holdout,
        )
        for index in range(count)
    )
    session.commit()
    session.close()


def test_eval_holdout_is_never_trained_on_and_scores_the_final_model(training, capsys):
    _add_mails(training.engine, 20)
    _add_mails(training.engine, 5, holdout=True)

    training.main()

    (cv_texts,) = training.cv_texts
    (final,) = training.final_trainers
    assert len(cv_texts) == len(final.train_dataset.texts) == 20
    assert not any(text.startswith("holdout") for text in cv_texts + final.train_dataset.texts)
    holdout_scores = json.loads(
        (training.output_dir / "default" / "holdout_scores.json").read_text()
    )
    assert holdout_scores["mails"] == 5 and holdout_scores["macro_f1"] == 1.0
    output = capsys.readouterr().out
    assert "20 labelled mails for training" in output
    assert "Warning: only 20 mails to train on" in output


def test_rerun_on_changed_data_does_not_reuse_the_old_trials(training):
    _add_mails(training.engine, 20)
    training.main()
    _add_mails(training.engine, 2)
    # Worse than the first run, whose trial has predictions for only 20 mails.
    training.score = 0.5

    training.main()

    assert len(training.final_trainers[-1].train_dataset.texts) == 22


def test_report_rejects_a_trial_run_on_other_data(script):
    trial = SimpleNamespace(number=3, value=0.9, user_attrs={"out_of_fold_predictions": [0, 1]})

    with pytest.raises(ValueError, match="other data"):
        script.report(trial, [0, 1, 1], script.label_ids("default"), "default")


def test_main_migrates_a_database_from_before_folder_lists(training):
    with training.engine.begin() as connection:
        for column in ("account", "folder_list"):
            connection.execute(text(f"DROP INDEX ix_emails_{column}"))
            connection.execute(text(f"ALTER TABLE emails DROP COLUMN {column}"))
        connection.execute(
            text(
                "INSERT INTO emails (text, true_label, processed, is_eval_holdout, "
                "is_training_data) VALUES (:text, :label, 1, 0, 1)"
            ),
            [{"text": f"mail {index}", "label": DEFAULT_LABELS[index % 5]} for index in range(20)],
        )

    training.main()

    (final,) = training.final_trainers
    assert len(final.train_dataset.texts) == 20


def test_publish_without_gcs_fails_before_loading_data(training, monkeypatch, script):
    def load_data(*args, **kwargs):
        raise AssertionError("data loaded although the model cannot be published")

    monkeypatch.setattr(script.gcs, "enabled", lambda: False)
    monkeypatch.setattr(script, "load_data", load_data)

    with pytest.raises(SystemExit, match="GCS_BUCKET"):
        training.main("--publish")


class FakeBucket:
    """GCS objects in a dict, with the directory calls of src.storage.gcs."""

    def __init__(self, objects=None, archive_fails=False):
        self.objects = dict(objects or {})
        self.archive_fails = archive_fails

    def download_directory(self, prefix, destination):
        prefix = prefix.rstrip("/") + "/"
        names = [name for name in self.objects if name.startswith(prefix)]
        for name in names:
            path = Path(destination, name[len(prefix):])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.objects[name])
        return bool(names)

    def upload_directory(self, source, prefix):
        if self.archive_fails and "_history/" in prefix:
            return False
        for path in Path(source).rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(source).as_posix()
                self.objects[f"{prefix.rstrip('/')}/{relative_path}"] = path.read_bytes()
        return True


def _publish(script, monkeypatch, tmp_path, bucket):
    monkeypatch.setattr(script.gcs, "download_directory", bucket.download_directory)
    monkeypatch.setattr(script.gcs, "upload_directory", bucket.upload_directory)
    new_model = tmp_path / "new-model"
    new_model.mkdir()
    (new_model / "model.safetensors").write_bytes(b"new")
    script.publish(new_model, "models/distilbert_deployed")


def test_publish_archives_the_live_model_before_replacing_it(script, monkeypatch, tmp_path):
    bucket = FakeBucket({"models/distilbert_deployed/model.safetensors": b"live"})

    _publish(script, monkeypatch, tmp_path, bucket)

    assert bucket.objects["models/distilbert_deployed/model.safetensors"] == b"new"
    (archived,) = [name for name in bucket.objects if "_history/" in name]
    assert bucket.objects[archived] == b"live"
    # The archive name is the one RESTORE MODEL parses (src/ai/retrain.py).
    prefix, archive_name, _ = archived.rsplit("/", 2)
    assert prefix == "models/distilbert_deployed_history"
    datetime.strptime(archive_name, "%Y%m%d_%H%M%S_%f")


def test_publish_is_aborted_when_the_live_model_cannot_be_archived(script, monkeypatch, tmp_path):
    bucket = FakeBucket(
        {"models/distilbert_deployed/model.safetensors": b"live"}, archive_fails=True
    )

    with pytest.raises(RuntimeError, match="archive"):
        _publish(script, monkeypatch, tmp_path, bucket)

    assert bucket.objects == {"models/distilbert_deployed/model.safetensors": b"live"}


def test_first_publish_has_nothing_to_archive(script, monkeypatch, tmp_path):
    bucket = FakeBucket()

    _publish(script, monkeypatch, tmp_path, bucket)

    assert bucket.objects == {"models/distilbert_deployed/model.safetensors": b"new"}
