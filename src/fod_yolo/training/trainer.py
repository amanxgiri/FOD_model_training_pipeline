"""Strict preflight, Ultralytics execution, and stable candidate artifact creation."""

from __future__ import annotations

import importlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

from fod_yolo.dataset.validate import StrictDatasetValidationError, validate_yolo_dataset
from fod_yolo.environment import (
    EnvironmentReport,
    EnvironmentValidationError,
    collect_environment_report,
    inspect_git,
    validate_environment,
    write_environment_freeze,
)
from fod_yolo.hashing import (
    atomic_replace_path,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    sha256_file,
)
from fod_yolo.paths import ProjectPaths
from fod_yolo.training import TrainingError, TrainingExecutionError
from fod_yolo.training.config import TrainingSettings, training_mode, validate_training_settings
from fod_yolo.training.resume import ResumeContext, append_resume_record
from fod_yolo.training.run_metadata import (
    failure_details,
    format_utc,
    generate_run_id,
    new_run_metadata,
    utc_now,
    write_run_metadata,
)


class YoloModel(Protocol):
    """Minimal Ultralytics model surface needed by the orchestrator."""

    def train(self, **kwargs: object) -> object:
        """Execute or resume training."""


YoloFactory = Callable[[str], YoloModel]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Successful training and candidate-artifact locations."""

    run_id: str
    run_directory: Path
    candidate_directory: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metadata_path: Path
    resumed: bool


def read_dataset_fingerprint(dataset_yaml: str | Path) -> str:
    """Read and validate the processed dataset fingerprint beside its YAML."""

    yaml_path = Path(dataset_yaml).expanduser().resolve()
    manifest_path = yaml_path.parent / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingExecutionError(
            f"Unable to read dataset manifest {manifest_path}: {exc}"
        ) from exc
    fingerprint = manifest.get("dataset_fingerprint") if isinstance(manifest, dict) else None
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint.casefold())
    ):
        raise TrainingExecutionError(
            f"Dataset manifest has no valid SHA-256 fingerprint: {manifest_path}"
        )
    return fingerprint


def run_training(
    settings: TrainingSettings,
    project_paths: ProjectPaths,
    *,
    allow_cpu: bool = False,
    resume: ResumeContext | None = None,
    allow_dataset_change: bool = False,
    yolo_factory: YoloFactory | None = None,
    environment_report: EnvironmentReport | None = None,
    dependency_freeze: tuple[str, ...] | None = None,
    now: Callable[[], datetime] = utc_now,
) -> TrainingResult:
    """Run strict preflight and training while preserving metadata on every failure."""

    validate_training_settings(settings)
    started_at = now()
    git = (
        environment_report.git
        if environment_report is not None
        else inspect_git(project_paths.project_root)
    )
    if resume is None:
        run_id = generate_run_id(settings.name_prefix, git.commit, started_at)
        run_directory = settings.project / run_id
        try:
            settings.project.mkdir(parents=True, exist_ok=True)
            run_directory.mkdir()
        except FileExistsError as exc:
            raise TrainingExecutionError(
                f"Generated run directory already exists; retry after the UTC second changes: "
                f"{run_directory}"
            ) from exc
        except OSError as exc:
            raise TrainingExecutionError(
                f"Unable to create training run directory {run_directory}: {exc}"
            ) from exc
        metadata = new_run_metadata(
            run_id=run_id,
            run_directory=run_directory,
            config_source=settings.source,
            dataset_yaml=settings.data,
            dataset_fingerprint=None,
            git=git,
            started_at=started_at,
            allow_cpu=allow_cpu,
        )
        if training_mode(settings) == "finetune":
            parent = Path(settings.model).expanduser().resolve()
            metadata["parent_checkpoint"] = {
                "path": str(parent),
                "sha256": sha256_file(parent),
                "size_bytes": parent.stat().st_size,
            }
        resolved_config = settings.resolved_config(
            run_id=run_id,
            device_override="cpu" if allow_cpu else None,
        )
    else:
        run_id = resume.run_id
        run_directory = resume.run_directory
        metadata = dict(resume.metadata)
        metadata["allow_cpu"] = allow_cpu
        metadata["ended_at_utc"] = None
        metadata["error"] = None
        resolved_config = None

    metadata_path = run_directory / "run_metadata.json"
    phase = "run_initialization"
    try:
        if resume is None:
            assert resolved_config is not None
            atomic_write_yaml(run_directory / "resolved_config.yaml", resolved_config)
        else:
            append_resume_record(
                metadata,
                checkpoint=resume.checkpoint,
                resumed_at=started_at,
                allow_dataset_change=allow_dataset_change,
            )
        metadata["status"] = "initializing"
        write_run_metadata(run_directory, metadata)

        phase = "dataset_preflight"
        dataset_fingerprint = read_dataset_fingerprint(settings.data)
        metadata["dataset_fingerprint"] = dataset_fingerprint
        if (
            resume is not None
            and resume.metadata.get("dataset_fingerprint") != dataset_fingerprint
            and not allow_dataset_change
        ):
            raise TrainingExecutionError("Dataset fingerprint changed after resume validation")
        if resume is not None:
            resume_record = metadata["resume_history"][-1]
            resume_record["original_dataset_fingerprint"] = resume.metadata.get(
                "dataset_fingerprint"
            )
            resume_record["resumed_dataset_fingerprint"] = dataset_fingerprint
        validate_yolo_dataset(settings.data, strict=True)

        phase = "environment_preflight"
        report = environment_report or collect_environment_report(
            project_root=project_paths.project_root,
            run_cuda_test=not allow_cpu,
        )
        atomic_write_json(run_directory / "environment.json", report.to_dict())
        if dependency_freeze is None:
            write_environment_freeze(run_directory / "requirements-freeze.txt")
        else:
            content = "\n".join(sorted(dependency_freeze))
            atomic_write_text(
                run_directory / "requirements-freeze.txt",
                f"{content}\n" if content else "",
            )
        validate_environment(
            report,
            require_cuda=not allow_cpu,
            require_model_check=False,
        )

        phase = "model_training"
        metadata["status"] = "running"
        metadata["training_started_at_utc"] = format_utc(now())
        write_run_metadata(run_directory, metadata)
        factory = yolo_factory or _default_yolo_factory
        checkpoint = str(resume.checkpoint if resume is not None else settings.model)
        model = factory(checkpoint)
        if resume is None:
            training_arguments = settings.ultralytics_arguments(
                run_id,
                device_override="cpu" if allow_cpu else None,
            )
            model.train(**cast(dict[str, object], training_arguments))
        else:
            resume_arguments: dict[str, object] = {"resume": True}
            if allow_cpu:
                resume_arguments["device"] = "cpu"
            model.train(**resume_arguments)

        phase = "checkpoint_finalization"
        best_checkpoint = run_directory / "weights" / "best.pt"
        last_checkpoint = run_directory / "weights" / "last.pt"
        checkpoint_metadata = _checkpoint_metadata(best_checkpoint, last_checkpoint)
        candidate_directory = _install_candidate(
            project_paths=project_paths,
            run_id=run_id,
            run_directory=run_directory,
            dataset_manifest=settings.data.parent / "dataset_manifest.json",
            dataset_fingerprint=dataset_fingerprint,
            checkpoints=checkpoint_metadata,
            parent_checkpoint=metadata.get("parent_checkpoint"),
        )
        metadata["candidate_directory"] = str(candidate_directory)
        metadata["checkpoints"] = checkpoint_metadata
        metadata["ended_at_utc"] = format_utc(now())
        metadata["error"] = None
        metadata["status"] = "success"
        write_run_metadata(run_directory, metadata)
        return TrainingResult(
            run_id=run_id,
            run_directory=run_directory,
            candidate_directory=candidate_directory,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            metadata_path=metadata_path,
            resumed=resume is not None,
        )
    except BaseException as exc:
        metadata["ended_at_utc"] = format_utc(now())
        metadata["error"] = failure_details(exc, phase=phase)
        metadata["status"] = "failed"
        if metadata["error"].get("out_of_memory"):
            metadata["error"]["batch"] = settings.training_arguments.get("batch")
            metadata["error"]["imgsz"] = settings.training_arguments.get("imgsz")
        write_run_metadata(run_directory, metadata)
        if isinstance(
            exc,
            (TrainingError, StrictDatasetValidationError, EnvironmentValidationError),
        ):
            raise
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise TrainingExecutionError(f"Training failed during {phase}: {exc}") from exc


def _default_yolo_factory(checkpoint: str) -> YoloModel:
    try:
        ultralytics = importlib.import_module("ultralytics")
        yolo_class = ultralytics.YOLO
    except (ImportError, AttributeError) as exc:
        raise TrainingExecutionError(f"Ultralytics YOLO is unavailable: {exc}") from exc
    return cast(YoloModel, yolo_class(checkpoint))


def _checkpoint_metadata(best: Path, last: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, checkpoint in (("best", best), ("last", last)):
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise TrainingExecutionError(f"Required non-empty checkpoint is missing: {checkpoint}")
        result[name] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        }
    return result


def _install_candidate(
    *,
    project_paths: ProjectPaths,
    run_id: str,
    run_directory: Path,
    dataset_manifest: Path,
    dataset_fingerprint: str,
    checkpoints: dict[str, dict[str, object]],
    parent_checkpoint: object = None,
) -> Path:
    candidates_root = project_paths.artifacts_root / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    destination = candidates_root / run_id
    if destination.exists():
        raise TrainingExecutionError(f"Candidate directory already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".tmp", dir=candidates_root))
    try:
        shutil.copy2(run_directory / "weights" / "best.pt", staging / "best.pt")
        shutil.copy2(run_directory / "weights" / "last.pt", staging / "last.pt")
        unique_best_filename = f"{run_id}_best.pt"
        shutil.copy2(run_directory / "weights" / "best.pt", staging / unique_best_filename)
        for name in ("best", "last"):
            copied_hash = sha256_file(staging / f"{name}.pt")
            if copied_hash != checkpoints[name]["sha256"]:
                raise TrainingExecutionError(
                    f"Candidate {name}.pt hash changed while copying into staging"
                )
        if sha256_file(staging / unique_best_filename) != checkpoints["best"]["sha256"]:
            raise TrainingExecutionError("Uniquely named best checkpoint failed hash verification")
        shutil.copy2(run_directory / "resolved_config.yaml", staging / "training_config.yaml")
        shutil.copy2(dataset_manifest, staging / "dataset_manifest.json")
        atomic_write_json(
            staging / "candidate_manifest.json",
            {
                "checkpoints": {
                    name: {
                        "filename": f"{name}.pt",
                        "sha256": value["sha256"],
                        "size_bytes": value["size_bytes"],
                    }
                    for name, value in checkpoints.items()
                },
                "dataset_fingerprint": dataset_fingerprint,
                "parent_checkpoint": parent_checkpoint,
                "unique_best_checkpoint": {
                    "filename": unique_best_filename,
                    "sha256": checkpoints["best"]["sha256"],
                    "size_bytes": checkpoints["best"]["size_bytes"],
                },
                "run_id": run_id,
                "schema_version": "1.0",
                "status": "training_complete",
            },
        )
        atomic_replace_path(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination
