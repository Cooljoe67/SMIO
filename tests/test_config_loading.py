import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import config
from src.storage import gcs

ROOT = Path(__file__).resolve().parents[1]


def _config_text(first_name):
    return f"""
folder_lists:
  default:
    delivery: {{}}
accounts:
  - id: primary
    provider: imap
    email: owner@test
    user_first_name: {first_name}
    folder_list: default
    imap: {{host: imap.test, username: owner@test, password: x}}
"""


BUNDLED = _config_text("Bundled")


class FakeBlob:
    def __init__(self, objects, name):
        self.objects = objects
        self.name = name

    def exists(self):
        return self.name in self.objects

    def download_as_text(self, encoding=None):
        return self.objects[self.name]

    def download_to_filename(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            file.write(self.objects[self.name])

    def upload_from_filename(self, filename):
        self.objects[self.name] = Path(filename).read_text(encoding="utf-8")


class FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self.objects, name)


@pytest.fixture
def bucket(monkeypatch, tmp_path):
    """A fake GCS bucket and a read-only bundled config, as in the image."""
    fake = FakeBucket()
    monkeypatch.setattr(gcs, "enabled", lambda: True)
    monkeypatch.setattr(gcs, "_bucket", lambda: fake)
    bundled = tmp_path / "smio.yaml"
    bundled.write_text(BUNDLED, encoding="utf-8")
    bundled.chmod(0o444)
    monkeypatch.setattr(config, "CONFIG_PATH", bundled)
    monkeypatch.setattr(config, "_config", None)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("SMIO_CONFIG_SOURCE", raising=False)
    return fake.objects


@pytest.mark.parametrize("variable, value", [("K_SERVICE", "smio"), ("SMIO_CONFIG_SOURCE", "gcs")])
def test_service_reads_the_gcs_config_without_writing_the_bundled_file(
    bucket, monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)
    bucket[config.CONFIG_GCS_OBJECT] = _config_text("Remote")

    assert config.reload_config().accounts[0].user_first_name == "Remote"
    assert config.CONFIG_PATH.read_text(encoding="utf-8") == BUNDLED


def test_scripts_read_the_local_file_and_leave_gcs_alone(bucket):
    bucket[config.CONFIG_GCS_OBJECT] = _config_text("Remote")

    assert config.reload_config().accounts[0].user_first_name == "Bundled"
    assert config.CONFIG_PATH.read_text(encoding="utf-8") == BUNDLED
    assert bucket == {config.CONFIG_GCS_OBJECT: _config_text("Remote")}


def test_scripts_never_upload_their_config(bucket):
    config.reload_config()

    assert bucket == {}


def test_service_uploads_the_bundled_config_when_gcs_has_none(bucket, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "smio")

    assert config.reload_config().accounts[0].user_first_name == "Bundled"
    assert bucket == {config.CONFIG_GCS_OBJECT: BUNDLED}


@pytest.mark.parametrize(
    "broken",
    ["folder_lists: [", BUNDLED.replace("folder_list: default", "folder_lists: default")],
    ids=["yaml", "schema"],
)
def test_broken_gcs_config_at_startup_falls_back_to_the_bundled_one(
    bucket, monkeypatch, caplog, broken
):
    monkeypatch.setenv("K_SERVICE", "smio")
    bucket[config.CONFIG_GCS_OBJECT] = broken

    with caplog.at_level(logging.ERROR, logger="src.config"):
        assert config.reload_config().accounts[0].user_first_name == "Bundled"
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_broken_gcs_config_later_keeps_the_previous_one(bucket, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "smio")
    previous = config.parse_config(_config_text("Previous"))
    monkeypatch.setattr(config, "_config", previous)
    bucket[config.CONFIG_GCS_OBJECT] = "folder_lists: ["

    assert config.reload_config() is previous


def test_gcs_bucket_is_read_when_used(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "smio-test")
    assert gcs.enabled()

    monkeypatch.setenv("GCS_BUCKET", "")
    assert not gcs.enabled()


def test_dotenv_is_loaded_before_the_environment_is_read():
    # A stand-in for load_dotenv that sets what a .env file would.
    script = (
        "import os, dotenv\n"
        "dotenv.load_dotenv = lambda *args, **kwargs: os.environ.update("
        "GCS_BUCKET='from-dotenv', GCS_CONFIG_OBJECT='from-dotenv.yaml')\n"
        "from src import config\n"
        "print(config.gcs.enabled(), config.CONFIG_GCS_OBJECT)\n"
    )
    env = {
        name: value for name, value in os.environ.items()
        if name not in ("GCS_BUCKET", "GCS_CONFIG_OBJECT")
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )

    assert result.stdout.split() == ["True", "from-dotenv.yaml"]
