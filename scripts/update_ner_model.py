"""Upload a NER model release and roll it out to Cloud Run."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


def _run_gcloud(arguments, allow_failure=False):
    executable = shutil.which("gcloud.cmd" if os.name == "nt" else "gcloud")
    if executable is None:
        raise SystemExit("Google Cloud CLI (gcloud) is required.")
    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise SystemExit("Google Cloud CLI (gcloud) is required.") from None

    if result.returncode and not allow_failure:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"gcloud {' '.join(arguments)} failed: {details}")
    return result


def _validate_model(model_dir):
    has_weights = any(
        (model_dir / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    has_tokenizer = any(
        (model_dir / filename).is_file()
        for filename in ("tokenizer.json", "vocab.txt", "spiece.model")
    )
    if not (model_dir / "config.json").is_file() or not has_weights or not has_tokenizer:
        raise SystemExit(
            f"{model_dir} is not a complete NER model directory "
            "(expected config, weights, and tokenizer files)."
        )


def _read_current_pointer(pointer_uri):
    result = _run_gcloud(["storage", "cat", pointer_uri], allow_failure=True)
    if result.returncode == 0:
        json.loads(result.stdout)
        return result.stdout

    details = f"{result.stdout}\n{result.stderr}".casefold()
    if "no urls matched" in details or "matched no objects" in details:
        return None
    raise RuntimeError(f"Could not read current NER model pointer: {details.strip()}")


def _upload_pointer(pointer_uri, contents):
    with tempfile.TemporaryDirectory(prefix="smio-ner-release-") as temporary_dir:
        pointer_file = Path(temporary_dir) / "current.json"
        pointer_file.write_text(contents, encoding="utf-8")
        _run_gcloud(["storage", "cp", str(pointer_file), pointer_uri])


def _delete_release(release_uri):
    _run_gcloud(["storage", "rm", "--recursive", release_uri], allow_failure=True)


def _deploy_revision(service, project, region, bucket, prefix, revision):
    _run_gcloud([
        "run", "services", "update", service,
        f"--project={project}",
        f"--region={region}",
        f"--update-env-vars=GCS_BUCKET={bucket},GCS_NER_MODEL_PREFIX={prefix},SMIO_NER_MODEL_REVISION={revision}",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Upload a versioned NER model to GCS and roll it out to Cloud Run."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("NER_MODEL_PATH", "./models/ner-smio")),
        help="Trained NER model directory (default: NER_MODEL_PATH or ./models/ner-smio).",
    )
    parser.add_argument("--bucket", default=os.getenv("GCS_BUCKET", ""))
    parser.add_argument(
        "--prefix",
        default=os.getenv("GCS_NER_MODEL_PREFIX", "models/ner-smio"),
        help="Base GCS prefix for versioned NER releases.",
    )
    parser.add_argument(
        "--service",
        default=os.getenv("SMIO_SERVICE_NAME", "smio"),
        help="Cloud Run service to update.",
    )
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID", ""))
    parser.add_argument("--region", default=os.getenv("GCP_REGION", ""))
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Upload and select a model release without updating Cloud Run; print its revision ID.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    _validate_model(model_dir)
    if not args.bucket:
        raise SystemExit("Set GCS_BUCKET in .env or pass --bucket.")
    if not args.prefix:
        raise SystemExit("Set GCS_NER_MODEL_PREFIX in .env or pass --prefix.")
    if not args.upload_only and (not args.project or not args.region):
        raise SystemExit("Set GCP_PROJECT_ID and GCP_REGION in .env or pass them explicitly.")
    if shutil.which("gcloud.cmd" if os.name == "nt" else "gcloud") is None:
        raise SystemExit("Google Cloud CLI (gcloud) is required.")

    revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = args.prefix.rstrip("/")
    release_prefix = f"{prefix}/releases/{revision}"
    release_uri = f"gs://{args.bucket}/{release_prefix}"
    pointer_uri = f"gs://{args.bucket}/{prefix}/current.json"
    previous_pointer = _read_current_pointer(pointer_uri)
    new_pointer = json.dumps({"revision": revision, "prefix": release_prefix})

    try:
        for file_path in sorted(model_dir.iterdir()):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(model_dir).as_posix()
            _run_gcloud([
                "storage", "cp", str(file_path),
                f"{release_uri}/{relative_path}",
            ])
        _upload_pointer(pointer_uri, new_pointer)
    except Exception:
        _delete_release(release_uri)
        raise

    if args.upload_only:
        print(revision)
        return

    try:
        _deploy_revision(
            args.service,
            args.project,
            args.region,
            args.bucket,
            prefix,
            revision,
        )
    except Exception:
        try:
            if previous_pointer is None:
                _run_gcloud(["storage", "rm", pointer_uri], allow_failure=True)
            else:
                _upload_pointer(pointer_uri, previous_pointer)
        except Exception:
            print(
                f"Could not restore the previous NER pointer; kept release {release_uri} "
                "to avoid leaving the pointer broken.",
                file=sys.stderr,
            )
        else:
            print(
                f"Cloud Run update failed; kept release {release_uri} "
                "in case a new revision references it.",
                file=sys.stderr,
            )
        raise

    print(f"NER model revision {revision} deployed to {args.service}.")


if __name__ == "__main__":
    main()
