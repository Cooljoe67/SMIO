"""GCS synchronization for filesystem-backed runtime artifacts."""

import os
import shutil
from pathlib import Path, PurePosixPath


def _bucket_name():
    # Read on every call: .env may be loaded after this module is imported.
    return os.getenv("GCS_BUCKET")


def enabled():
    return bool(_bucket_name())


def syncs():
    """Whether runtime artifacts (database, config, models, command state) follow GCS.

    Only the deployed service does (Cloud Run sets K_SERVICE), or a local run
    that opts in with SMIO_GCS=1 while the service is idle. .env is loaded by
    every process, so GCS_BUCKET alone must not make local runs read or
    overwrite production. Explicit tools (--publish, the database sync script)
    use the bucket through enabled().
    """
    return enabled() and bool(os.getenv("K_SERVICE") or os.getenv("SMIO_GCS") == "1")


def _bucket():
    bucket_name = _bucket_name()
    if not bucket_name:
        raise RuntimeError("GCS_BUCKET must be set to use GCS storage")

    try:
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError(
            "google-cloud-storage is required when GCS_BUCKET is configured"
        ) from error

    return storage.Client().bucket(bucket_name)


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


def download_text(object_name):
    """An object's content as text, or None if it does not exist in GCS."""
    if not enabled():
        return None

    blob = _bucket().blob(object_name)
    if not blob.exists():
        return None
    return blob.download_as_text(encoding="utf-8")


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

# Versioned objects: every GCS object has a generation that changes with each
# write. Writing "only if the generation is still N" lets two service
# instances (e.g. the old and new revision during a deploy) share an object
# without overwriting each other.


class StaleObject(RuntimeError):
    """The object changed in GCS since the generation the caller knew."""


def generation(object_name):
    """The object's current generation, or None if it does not exist."""
    blob = _bucket().get_blob(object_name)
    return blob.generation if blob else None


def download_versioned(object_name, destination):
    """Replace destination with the object; returns its generation, or None if
    it does not exist. The file is swapped in whole, never half-written."""
    blob = _bucket().get_blob(object_name)
    if blob is None:
        return None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".download")
    blob.download_to_filename(partial, if_generation_match=blob.generation)
    os.replace(partial, destination)
    return blob.generation


def upload_versioned(source, object_name, expected_generation):
    """Upload unless the object changed since expected_generation (0: must not
    exist yet); returns the new generation or raises StaleObject."""
    from google.api_core.exceptions import PreconditionFailed

    blob = _bucket().blob(object_name)
    try:
        blob.upload_from_filename(source, if_generation_match=expected_generation)
    except PreconditionFailed as error:
        raise StaleObject(object_name) from error
    return blob.generation


def acquire_lease(object_name, holder, seconds):
    """Take a lease held in an object; its generation if taken, None if another
    holder's lease has not expired. An expired lease is taken over."""
    import json
    import time

    from google.api_core.exceptions import PreconditionFailed

    now = time.time()
    body = json.dumps({"holder": holder, "expires": now + seconds})
    blob = _bucket().blob(object_name)
    current = _bucket().get_blob(object_name)
    if current is not None:
        try:
            expires = float(json.loads(current.download_as_text()).get("expires", 0))
        except (ValueError, AttributeError):
            expires = 0
        if expires > now:
            return None
    try:
        blob.upload_from_string(
            body, content_type="application/json",
            if_generation_match=current.generation if current else 0,
        )
    except PreconditionFailed:
        # Someone else took it between the read and the write.
        return None
    return blob.generation


def release_lease(object_name, lease_generation):
    """Give a lease back, unless it expired and was taken over meanwhile."""
    from google.api_core.exceptions import NotFound, PreconditionFailed

    try:
        _bucket().blob(object_name).delete(if_generation_match=lease_generation)
    except (NotFound, PreconditionFailed):
        pass
