"""An in-memory GCS bucket with object generations, for src.storage.gcs."""


from google.api_core.exceptions import NotFound, PreconditionFailed


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        stored = bucket.objects.get(name)
        self.generation = stored[1] if stored else None

    def _check(self, if_generation_match):
        current = self.bucket.objects.get(self.name)
        if if_generation_match is not None and (current[1] if current else 0) != if_generation_match:
            raise PreconditionFailed("generation mismatch")

    def _store(self, data, if_generation_match):
        self._check(if_generation_match)
        self.bucket.next_generation += 1
        self.generation = self.bucket.next_generation
        self.bucket.objects[self.name] = (data, self.generation)

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        self._store(data.encode() if isinstance(data, str) else data, if_generation_match)

    def upload_from_filename(self, path, if_generation_match=None):
        with open(path, "rb") as source:
            self._store(source.read(), if_generation_match)

    def download_as_text(self, encoding="utf-8"):
        return self.bucket.objects[self.name][0].decode(encoding)

    def download_to_filename(self, path, if_generation_match=None):
        self._check(if_generation_match)
        with open(path, "wb") as target:
            target.write(self.bucket.objects[self.name][0])

    def exists(self):
        return self.name in self.bucket.objects

    def delete(self, if_generation_match=None):
        if self.name not in self.bucket.objects:
            raise NotFound(self.name)
        self._check(if_generation_match)
        del self.bucket.objects[self.name]


class FakeBucket:
    def __init__(self):
        self.objects = {}  # name -> (bytes, generation)
        self.next_generation = 100

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        return FakeBlob(self, name) if name in self.objects else None


def install(monkeypatch, gcs):
    bucket = FakeBucket()
    monkeypatch.setattr(gcs, "_bucket", lambda: bucket)
    return bucket


