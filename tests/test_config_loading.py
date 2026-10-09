import os
import subprocess
import sys
from pathlib import Path

from src.storage import gcs

ROOT = Path(__file__).resolve().parents[1]


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
