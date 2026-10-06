"""GCS synchronization for filesystem-backed runtime artifacts."""

import os
import shutil
from pathlib import Path, PurePosixPath


BUCKET_NAME = os.getenv("GCS_BUCKET")


def enabled():
    return bool(BUCKET_NAME)


def _bucket():
    if not BUCKET_NAME:
        raise RuntimeError("GCS_BUCKET must be set to use GCS storage")

    try:
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError(
            "google-cloud-storage is required when GCS_BUCKET is configured"
        ) from error

    return storage.Client().bucket(BUCKET_NAME)


def download_file(object_name, destination):
    """Download one object, returning whether it existed in GCS."""
    if not enabled():
        return False

    blob = _bucket().blob(object_name)
    if not blob.exists():
        return False

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination)
    return True


def upload_file(source, object_name):
    if not enabled():
        return False

    _bucket().blob(object_name).upload_from_filename(source)
    return True


def download_directory(prefix, destination):
    """Replace a local directory with objects found under a GCS prefix."""
    if not enabled():
        return False

    prefix = prefix.rstrip("/") + "/"
    blobs = list(_bucket().list_blobs(prefix=prefix))
    if not blobs:
        return False

    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for blob in blobs:
        relative_path = PurePosixPath(blob.name).relative_to(prefix)
        if not relative_path.name:
            continue
        local_path = destination.joinpath(*relative_path.parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
    return True


def upload_directory(source, prefix):
    if not enabled():
        return False

    source = Path(source)
    prefix = prefix.rstrip("/")
    for file_path in source.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(source).as_posix()
            _bucket().blob(f"{prefix}/{relative_path}").upload_from_filename(file_path)
    return True


def list_directory_prefixes(prefix):
    """List immediate child directories under a GCS object prefix."""
    if not enabled():
        return []

    object_prefix = prefix.rstrip("/") + "/"
    child_directories = set()
    for blob in _bucket().list_blobs(prefix=object_prefix):
        relative_name = blob.name[len(object_prefix):]
        if "/" in relative_name:
            child_directories.add(relative_name.split("/", 1)[0])
    return sorted(child_directories)


def delete_directory(prefix):
    """Delete all objects under one directory prefix."""
    if not enabled():
        return 0

    object_prefix = prefix.rstrip("/") + "/"
    blobs = list(_bucket().list_blobs(prefix=object_prefix))
    for blob in blobs:
        blob.delete()
    return len(blobs)