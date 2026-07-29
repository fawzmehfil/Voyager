"""Canonical, deterministic serialization helpers for replay artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: object) -> bytes:
    """Return Voyager's stable UTF-8 JSON representation."""

    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def canonical_gzip_bytes(value: object) -> bytes:
    """Return deterministic gzip data (including a stable header)."""

    return gzip.compress(canonical_json_bytes(value), compresslevel=9, mtime=0)


def write_json(path: Path, value: object, *, pretty: bool = True) -> None:
    payload = (
        (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()
        if pretty
        else canonical_json_bytes(value)
    )
    _atomic_write(path, payload)


def write_json_gz(path: Path, value: object) -> None:
    _atomic_write(path, canonical_gzip_bytes(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
