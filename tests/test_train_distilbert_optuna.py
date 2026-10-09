import importlib
import sys
import types
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
    ids = script.label_ids("rhein-cargo")
    assert ids["other"] == len(ids) - 1
    assert sorted(ids.values()) == list(range(11))


def test_load_data_only_uses_the_folder_lists_labels(db, script):
    db.add_all([
        Email(folder_list="rhein-cargo", true_label="dokumente", text="a", case_ref="VG:1"),
        Email(folder_list="rhein-cargo", true_label="other", text="b"),
        Email(folder_list="rhein-cargo", true_label=None, text="unlabelled"),
        Email(folder_list="default", true_label="other", text="other list"),
    ])
    db.commit()
    ids = script.label_ids("rhein-cargo")

    texts, labels, groups = script.load_data(db, "rhein-cargo", ids)

    assert texts == ["a", "b"]
    assert labels == [ids["dokumente"], ids["other"]]
    assert groups[0] == "VG:1" and groups[1].startswith("mail-")


def test_load_data_rejects_labels_outside_the_list(db, script):
    db.add(Email(folder_list="rhein-cargo", true_label="delivery", text="a"))
    db.commit()

    with pytest.raises(ValueError, match="delivery"):
        script.load_data(db, "rhein-cargo", script.label_ids("rhein-cargo"))


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
