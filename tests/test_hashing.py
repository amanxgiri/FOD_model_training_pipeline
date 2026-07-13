"""Tests for file hashing and atomic artifact writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fod_yolo.hashing import (
    AtomicWriteError,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    sha256_file,
    verify_sha256,
)


def test_sha256_file_streams_a_known_digest(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"abc")

    digest = sha256_file(source, chunk_size=1)

    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert verify_sha256(source, digest.upper())


def test_verify_sha256_rejects_malformed_digest(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"data")

    with pytest.raises(ValueError, match="64-character"):
        verify_sha256(source, "not-a-digest")


def test_atomic_text_write_replaces_existing_content(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "artifact.txt"
    atomic_write_text(destination, "first")

    result = atomic_write_text(destination, "second")

    assert result == destination.resolve()
    assert destination.read_text(encoding="utf-8") == "second"
    assert list(destination.parent.glob("*.tmp")) == []


def test_atomic_json_is_stable_utf8_and_strict(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"

    atomic_write_json(destination, {"z": "FOD", "a": 1})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"a": 1, "z": "FOD"}
    assert destination.read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(AtomicWriteError, match="strict JSON"):
        atomic_write_json(destination, {"invalid": float("nan")})


def test_atomic_yaml_round_trips_mapping(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.yaml"

    atomic_write_yaml(destination, {"training": {"batch": 4}})

    assert yaml.safe_load(destination.read_text(encoding="utf-8")) == {"training": {"batch": 4}}
