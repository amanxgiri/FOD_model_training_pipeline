"""Stable training run identity and metadata state transitions."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fod_yolo.environment import GitReport
from fod_yolo.hashing import atomic_write_json

_RUN_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(UTC)


def format_utc(value: datetime) -> str:
    """Format a UTC timestamp for JSON metadata."""

    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def generate_run_id(name_prefix: str, git_commit: str | None, started_at: datetime) -> str:
    """Generate the specification-defined run identity."""

    prefix = _RUN_COMPONENT.sub("_", name_prefix).strip("._-")
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    short_sha = git_commit[:7] if git_commit else "nogit"
    return f"{prefix}_{timestamp}_{short_sha}"


def new_run_metadata(
    *,
    run_id: str,
    run_directory: Path,
    config_source: Path,
    dataset_yaml: Path,
    dataset_fingerprint: str | None,
    git: GitReport,
    started_at: datetime,
    allow_cpu: bool,
) -> dict[str, Any]:
    """Create metadata before any preflight or model execution can fail."""

    return {
        "allow_cpu": allow_cpu,
        "candidate_directory": None,
        "checkpoints": {},
        "config_source": str(config_source),
        "created_at_utc": format_utc(started_at),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_yaml": str(dataset_yaml),
        "ended_at_utc": None,
        "error": None,
        "git": git.to_dict(),
        "resume_history": [],
        "run_directory": str(run_directory),
        "run_id": run_id,
        "schema_version": "1.0",
        "status": "initializing",
    }


def write_run_metadata(run_directory: Path, metadata: dict[str, Any]) -> Path:
    """Atomically persist run metadata in its stable location."""

    return atomic_write_json(run_directory / "run_metadata.json", metadata)


def failure_details(error: BaseException, *, phase: str) -> dict[str, object]:
    """Return safe error metadata with an OOM-specific operator recommendation."""

    message = str(error)
    out_of_memory = "out of memory" in message.casefold()
    details: dict[str, object] = {
        "message": message,
        "out_of_memory": out_of_memory,
        "phase": phase,
        "type": type(error).__name__,
    }
    if out_of_memory:
        details["recommendation"] = (
            "Lower training.batch and retry. Keep training.imgsz=1280 unchanged for Phase 1."
        )
    return details
