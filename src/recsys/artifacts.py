"""Small serialization helpers shared by data, checkpoints and reports."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def atomic_path(path, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; choose another path or set overwrite=true")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # fails atomically if another writer won the race
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value, overwrite=False):
    with atomic_path(path, overwrite) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def read_json(path):
    return json.loads(Path(path).read_text())
