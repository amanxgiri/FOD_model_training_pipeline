"""Streaming SHA-256 helpers and atomic UTF-8 artifact writes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class HashingError(RuntimeError):
    """Raised when a file cannot be hashed or verified."""


class AtomicWriteError(RuntimeError):
    """Raised when an atomic artifact write cannot be completed."""


def atomic_replace_path(
    source: str | Path,
    destination: str | Path,
    *,
    attempts: int = 6,
    initial_delay_seconds: float = 0.05,
) -> None:
    """Atomically replace a path, retrying transient Windows access-denied locks."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if initial_delay_seconds < 0:
        raise ValueError("initial_delay_seconds cannot be negative")
    source_path = Path(source)
    destination_path = Path(destination)
    for attempt in range(attempts):
        try:
            os.replace(source_path, destination_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(initial_delay_seconds * (attempt + 1))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise HashingError(f"Cannot hash a missing or non-file path: {source}")

    digest = hashlib.sha256()
    try:
        with source.open("rb") as file_handle:
            while chunk := file_handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise HashingError(f"Unable to hash file {source}: {exc}") from exc
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected_sha256: str) -> bool:
    """Compare a file digest against one validated 64-character SHA-256 string."""

    normalized = expected_sha256.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return hmac.compare_digest(sha256_file(path), normalized)


def sha256_json(value: object) -> str:
    """Hash one canonical strict-JSON representation for stable fingerprints."""

    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise HashingError(f"Value cannot be canonicalized as strict JSON: {exc}") from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Atomically replace a file with bytes written and flushed in its destination directory."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        atomic_replace_path(temporary_path, destination)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise AtomicWriteError(f"Unable to atomically write {destination}: {exc}") from exc

    return destination


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Atomically write UTF-8 text with no platform-specific encoding dependency."""

    return atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(
    path: str | Path,
    value: object,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """Serialize stable, strict JSON and atomically replace the destination."""

    try:
        content = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
        )
    except (TypeError, ValueError) as exc:
        raise AtomicWriteError(f"Value cannot be serialized as strict JSON: {exc}") from exc
    return atomic_write_text(path, f"{content}\n")


def atomic_write_yaml(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Serialize a mapping as stable UTF-8 YAML and atomically replace the destination."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise AtomicWriteError("PyYAML is required to write YAML artifacts") from exc

    try:
        content = yaml.safe_dump(
            dict(value),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        )
    except yaml.YAMLError as exc:
        raise AtomicWriteError(f"Value cannot be serialized as YAML: {exc}") from exc
    return atomic_write_text(path, content)
