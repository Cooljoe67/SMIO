import importlib
import sys
import types

import pytest

from src.db.models import Email

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
