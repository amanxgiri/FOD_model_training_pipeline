"""Tests for immutable run identity and dataset-safe resume behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fod_yolo.hashing import atomic_write_json, atomic_write_yaml
from fod_yolo.training import TrainingResumeError
from fod_yolo.training.resume import load_resume_context, resolved_config_for_checkpoint


def test_resume_rejects_changed_dataset_fingerprint(tmp_path: Path) -> None:
    checkpoint, run_directory = _failed_run(tmp_path, fingerprint="a" * 64)

    with pytest.raises(TrainingResumeError, match="fingerprint differs"):
        load_resume_context(
            checkpoint,
            dataset_fingerprint="b" * 64,
            allow_dataset_change=False,
        )

    context = load_resume_context(
        checkpoint,
        dataset_fingerprint="b" * 64,
        allow_dataset_change=True,
    )
    assert context.run_directory == run_directory
    assert context.run_id == run_directory.name


def test_resume_uses_original_resolved_configuration(tmp_path: Path) -> None:
    checkpoint, run_directory = _failed_run(tmp_path, fingerprint="a" * 64)

    assert resolved_config_for_checkpoint(checkpoint) == run_directory / "resolved_config.yaml"


def test_successful_run_cannot_be_resumed(tmp_path: Path) -> None:
    checkpoint, run_directory = _failed_run(tmp_path, fingerprint="a" * 64)
    metadata_path = run_directory / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "success"
    atomic_write_json(metadata_path, metadata)

    with pytest.raises(TrainingResumeError, match="successfully completed"):
        load_resume_context(
            checkpoint,
            dataset_fingerprint="a" * 64,
            allow_dataset_change=False,
        )


def _failed_run(tmp_path: Path, *, fingerprint: str) -> tuple[Path, Path]:
    run_directory = tmp_path / "runs" / "train" / "fixture_run"
    weights = run_directory / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "last.pt"
    checkpoint.write_bytes(b"partial-checkpoint")
    atomic_write_yaml(
        run_directory / "resolved_config.yaml",
        {
            "data": str(tmp_path / "data" / "fod_a.yaml"),
            "metadata": {"tags": ["fixture"]},
            "model": "yolo26n.pt",
            "training": {},
        },
    )
    atomic_write_json(
        run_directory / "run_metadata.json",
        {
            "dataset_fingerprint": fingerprint,
            "resume_history": [],
            "run_id": run_directory.name,
            "status": "failed",
        },
    )
    return checkpoint, run_directory
