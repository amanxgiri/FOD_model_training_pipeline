"""Fingerprint-safe Ultralytics checkpoint resume validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fod_yolo.training import TrainingResumeError
from fod_yolo.training.run_metadata import format_utc


@dataclass(frozen=True, slots=True)
class ResumeContext:
    """Validated original-run state for one resume operation."""

    checkpoint: Path
    run_directory: Path
    run_id: str
    resolved_config: Path
    metadata: dict[str, Any]


def resolved_config_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """Locate the immutable configuration belonging to a conventional last checkpoint."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise TrainingResumeError(f"Resume checkpoint does not exist: {checkpoint}")
    if checkpoint.name != "last.pt" or checkpoint.parent.name != "weights":
        raise TrainingResumeError("Resume requires a run's weights/last.pt checkpoint")
    resolved_config = checkpoint.parent.parent / "resolved_config.yaml"
    if not resolved_config.is_file():
        raise TrainingResumeError(f"Original resolved configuration is missing: {resolved_config}")
    return resolved_config


def load_resume_context(
    checkpoint_path: str | Path,
    *,
    dataset_fingerprint: str,
    allow_dataset_change: bool,
) -> ResumeContext:
    """Validate checkpoint layout, original identity, status, and dataset fingerprint."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise TrainingResumeError(f"Resume checkpoint does not exist: {checkpoint}")
    if checkpoint.name != "last.pt" or checkpoint.parent.name != "weights":
        raise TrainingResumeError("Resume requires a run's weights/last.pt checkpoint")
    run_directory = checkpoint.parent.parent
    metadata_path = run_directory / "run_metadata.json"
    resolved_config = run_directory / "resolved_config.yaml"
    metadata = _read_metadata(metadata_path)
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or run_id != run_directory.name:
        raise TrainingResumeError("Resume metadata run ID does not match its directory")
    if metadata.get("status") == "success":
        raise TrainingResumeError("A successfully completed run cannot be resumed")
    original_fingerprint = metadata.get("dataset_fingerprint")
    if not isinstance(original_fingerprint, str) or not original_fingerprint:
        raise TrainingResumeError("Original run metadata has no dataset fingerprint")
    if original_fingerprint != dataset_fingerprint and not allow_dataset_change:
        raise TrainingResumeError(
            "Dataset fingerprint differs from the original run; use --allow-dataset-change "
            "only after reviewing the provenance change"
        )
    if not resolved_config.is_file():
        raise TrainingResumeError(f"Original resolved configuration is missing: {resolved_config}")
    return ResumeContext(
        checkpoint=checkpoint,
        run_directory=run_directory,
        run_id=run_id,
        resolved_config=resolved_config,
        metadata=metadata,
    )


def append_resume_record(
    metadata: dict[str, Any],
    *,
    checkpoint: Path,
    resumed_at: datetime,
    allow_dataset_change: bool,
) -> None:
    """Append, rather than overwrite, the run's resume provenance."""

    history = metadata.setdefault("resume_history", [])
    if not isinstance(history, list):
        raise TrainingResumeError("Run metadata resume_history must be a list")
    history.append(
        {
            "allow_dataset_change": allow_dataset_change,
            "checkpoint": str(checkpoint),
            "resumed_at_utc": format_utc(resumed_at),
        }
    )


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingResumeError(f"Unable to read original run metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingResumeError(f"Run metadata root must be a mapping: {path}")
    return value
