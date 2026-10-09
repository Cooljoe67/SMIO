"""Credentials from Google Secret Manager, referenced by secret name in smio.yaml."""

import os


_cache = {}


def _project():
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")
    if not project:
        import google.auth

        _, project = google.auth.default()
    if not project:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT to read secrets from Secret Manager")
    return project


def read_secret(name):
    """The latest version of a secret, cached for the life of the process."""
    if name not in _cache:
        # Imported here so scripts and tests that never read a secret run without it.
        try:
            from google.cloud import secretmanager
        except ImportError as error:
            raise RuntimeError(
                f"google-cloud-secret-manager is required to read the secret {name!r}"
            ) from error

        version = f"projects/{_project()}/secrets/{name}/versions/latest"
        response = secretmanager.SecretManagerServiceClient().access_secret_version(name=version)
        _cache[name] = response.payload.data.decode("utf-8")
    return _cache[name]


def invalidate(name):
    """Forget a cached secret, e.g. after a failed login, so it is read again."""
    _cache.pop(name, None)
