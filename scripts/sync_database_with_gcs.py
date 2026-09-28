"""Download the GCS SQLite database, pause for local edits, and optionally upload it."""

import argparse
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google.auth import default as get_default_credentials
from google.auth.exceptions import DefaultCredentialsError


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.storage import gcs  # noqa: E402


def _is_valid_database(path):
    connection = None
    try:
        connection = sqlite3.connect(str(path))
        result = connection.execute("PRAGMA integrity_check").fetchone()
        return result == ("ok",)
    except sqlite3.DatabaseError:
        return False
    finally:
        if connection is not None:
            connection.close()


def _database_fingerprint(path):
    fingerprint = hashlib.sha256()
    with path.open("rb") as database_file:
        for chunk in iter(lambda: database_file.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    return fingerprint.digest()


def _gcloud_executable():
    executable = shutil.which("gcloud.cmd" if os.name == "nt" else "gcloud")
    if executable is None:
        raise SystemExit("Google Cloud CLI (gcloud) is required to authenticate.")
    return executable


def _run_gcloud(executable, arguments, capture_output=False):
    try:
        return subprocess.run(
            [executable, *arguments],
            check=True,
            capture_output=capture_output,
            text=capture_output,
        )
    except FileNotFoundError:
        raise SystemExit("Google Cloud CLI (gcloud) is required to manage the scheduled job.") from None
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"gcloud command failed: {' '.join(arguments)}; local database was not replaced."
        ) from None


def _authenticate_with_gcloud(project):
    executable = _gcloud_executable()
    active_account = _run_gcloud(
        executable,
        ("auth", "list", "--filter=status:ACTIVE", "--format=value(account)"),
        capture_output=True,
    ).stdout.strip()
    if not active_account:
        print("Logging in to Google Cloud...")
        _run_gcloud(executable, ("auth", "login"))

    _run_gcloud(executable, ("config", "set", "project", project))
    try:
        get_default_credentials()
    except DefaultCredentialsError:
        print("Setting up Google Application Default Credentials...")
        _run_gcloud(executable, ("auth", "application-default", "login"))


def _pause_scheduler_job(job, location, project):
    executable = _gcloud_executable()
    state = _run_gcloud(
        executable,
        (
            "scheduler", "jobs", "describe", job,
            f"--location={location}", f"--project={project}", "--format=value(state)",
        ),
        capture_output=True,
    ).stdout.strip().upper()
    if state == "PAUSED":
        print("5 min job is already paused; it will remain paused.")
        return False
    if state != "ENABLED":
        raise SystemExit(f"5 min job has unexpected state {state!r}; database was not changed.")

    print("Stopping 5 min job...")
    _run_gcloud(
        executable,
        ("scheduler", "jobs", "pause", job, f"--location={location}", f"--project={project}"),
    )
    return True


def _resume_scheduler_job(job, location, project):
    print("Restarting 5 min job...")
    try:
        _run_gcloud(
            _gcloud_executable(),
            ("scheduler", "jobs", "resume", job, f"--location={location}", f"--project={project}"),
        )
    except SystemExit as error:
        print(f"Could not restart 5 min job automatically: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replace the local SQLite database with its GCS copy; optionally "
            "pause for edits and upload the edited database after confirmation."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(os.getenv("SQLITE_DB_PATH", "./smio.db")),
        help="Local SQLite database path (default: SQLITE_DB_PATH or ./smio.db).",
    )
    parser.add_argument(
        "--gcs-object",
        default=os.getenv("GCS_DATABASE_OBJECT", "databases/smio.db"),
        help="GCS object name (default: GCS_DATABASE_OBJECT or databases/smio.db).",
    )
    parser.add_argument(
        "--project",
        default=(
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("CLOUDSDK_CORE_PROJECT")
            or "smio-509409"
        ),
        help="Google Cloud project for gcloud authentication (default: deployment project).",
    )
    parser.add_argument(
        "--scheduler-job",
        default="smio-five-minute",
        help="Five-minute Cloud Scheduler job name (default: smio-five-minute).",
    )
    parser.add_argument(
        "--scheduler-location",
        default="europe-west3",
        help="Cloud Scheduler region (default: europe-west3).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and replace the local database, then exit without prompting to upload.",
    )
    args = parser.parse_args()

    if not gcs.enabled():
        raise SystemExit("GCS_BUCKET is not configured in the environment or .env file.")

    database_path = args.db_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    downloaded_fingerprint = None
    scheduler_paused = False
    try:
        if args.download_only:
            print("Download-only mode; leaving 5 min job running.")
        else:
            print("Authenticating with Google Cloud...")
            _authenticate_with_gcloud(args.project)
            scheduler_paused = _pause_scheduler_job(
                args.scheduler_job,
                args.scheduler_location,
                args.project,
            )

        print("Copying database from GCS...")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{database_path.name}.",
            suffix=".download",
            dir=database_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)

        try:
            downloaded = gcs.download_file(args.gcs_object, temporary_path)
        except DefaultCredentialsError:
            print("Google Cloud credentials are unavailable; starting gcloud login.")
            _authenticate_with_gcloud(args.project)
            try:
                downloaded = gcs.download_file(args.gcs_object, temporary_path)
            except DefaultCredentialsError:
                raise SystemExit(
                    "Google Cloud credentials are still unavailable; local database unchanged."
                ) from None
        if not downloaded:
            raise SystemExit(f"GCS database object not found: {args.gcs_object}")
        if not _is_valid_database(temporary_path):
            raise SystemExit("Downloaded file failed SQLite integrity_check; local database unchanged.")
        downloaded_fingerprint = _database_fingerprint(temporary_path)

        temporary_path.replace(database_path)
        print(f"Downloaded and verified the GCS database at {database_path}.")
        if args.download_only:
            return

        print(
            "The five-minute job will resume automatically when this script exits. "
            "If it is interrupted before then, resume it manually with:"
        )
        print(
            f"  gcloud scheduler jobs resume {args.scheduler_job} "
            f"--location={args.scheduler_location} --project={args.project}"
        )
        input("Make your local edits, then press Enter here to review the upload confirmation...")
        confirmation = input(
            f"Type uppercase Y to overwrite GCS object {args.gcs_object} with this database: "
        )
        if confirmation != "Y":
            print("Upload canceled; GCS was not changed.")
            return

        if _database_fingerprint(database_path) == downloaded_fingerprint:
            print("Database is unchanged since download; skipping write-back.")
            return

        if not _is_valid_database(database_path):
            raise SystemExit("Local database failed SQLite integrity_check; GCS was not changed.")
        print("Writing back DB...")
        if not gcs.upload_file(database_path, args.gcs_object):
            raise SystemExit("GCS upload failed; check credentials and logs.")
        print(f"Uploaded the edited database to GCS object {args.gcs_object}.")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if scheduler_paused:
            _resume_scheduler_job(
                args.scheduler_job,
                args.scheduler_location,
                args.project,
            )


if __name__ == "__main__":
    main()
