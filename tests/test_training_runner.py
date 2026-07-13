"""Fake-backed integration tests for training success and failure artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from fod_yolo.dataset.pipeline import PreparationResult
from fod_yolo.environment import (
    EnvironmentReport,
    EnvironmentValidationError,
    GitReport,
    GpuReport,
    ModelCheckReport,
    NvidiaSmiReport,
    PackageReport,
)
from fod_yolo.hashing import sha256_file
from fod_yolo.paths import ProjectPaths
from fod_yolo.training import TrainingExecutionError
from fod_yolo.training.config import TrainingSettings
from fod_yolo.training.resume import load_resume_context
from fod_yolo.training.trainer import read_dataset_fingerprint, run_training

FIXED_TIME = datetime(2026, 7, 14, 2, 15, tzinfo=UTC)


class FakeYolo:
    """Write deterministic fake checkpoints or raise a configured error."""

    def __init__(self, checkpoint: str, *, failure: Exception | None = None) -> None:
        self.checkpoint = checkpoint
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def train(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        if kwargs.get("resume") is True:
            run_directory = Path(self.checkpoint).parent.parent
        else:
            run_directory = Path(str(kwargs["project"])) / str(kwargs["name"])
        weights = run_directory / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"best-checkpoint")
        (weights / "last.pt").write_bytes(b"last-checkpoint")
        return {"status": "fixture"}


def test_successful_training_writes_reproducible_run_and_candidate_artifacts(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    settings, paths = _settings_and_paths(tmp_path, prepared_tiny_dataset)
    fake_model = FakeYolo(settings.model)

    result = run_training(
        settings,
        paths,
        allow_cpu=True,
        yolo_factory=lambda checkpoint: fake_model,
        environment_report=_environment_report(cuda_available=False),
        dependency_freeze=("Pillow==12.3.0", "PyYAML==6.0.3"),
        now=lambda: FIXED_TIME,
    )

    assert result.run_id == "yolo26n_fod_phase1_1280_20260714T021500Z_abcdef1"
    assert fake_model.checkpoint == "yolo26n.pt"
    assert fake_model.calls[0]["imgsz"] == 1280
    assert fake_model.calls[0]["data"] == str(settings.data)
    assert fake_model.calls[0]["exist_ok"] is True
    assert fake_model.calls[0]["device"] == "cpu"
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "success"
    assert metadata["dataset_fingerprint"]
    assert metadata["git"]["dirty"] is True
    assert result.candidate_directory.joinpath("best.pt").is_file()
    candidate_manifest = json.loads(
        result.candidate_directory.joinpath("candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert candidate_manifest["checkpoints"]["best"]["sha256"] == sha256_file(
        result.best_checkpoint
    )
    assert result.run_directory.joinpath("resolved_config.yaml").is_file()
    resolved_config = yaml.safe_load(
        result.run_directory.joinpath("resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved_config["runtime"]["run_id"] == result.run_id
    assert resolved_config["runtime"]["ultralytics_arguments"]["device"] == "cpu"
    assert result.run_directory.joinpath("environment.json").is_file()
    assert result.run_directory.joinpath("requirements-freeze.txt").is_file()


def test_out_of_memory_failure_preserves_actionable_run_metadata(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    settings, paths = _settings_and_paths(tmp_path, prepared_tiny_dataset)
    fake_model = FakeYolo(settings.model, failure=RuntimeError("CUDA out of memory"))

    with pytest.raises(TrainingExecutionError, match="model_training"):
        run_training(
            settings,
            paths,
            allow_cpu=True,
            yolo_factory=lambda checkpoint: fake_model,
            environment_report=_environment_report(cuda_available=False),
            dependency_freeze=(),
            now=lambda: FIXED_TIME,
        )

    metadata_path = next(settings.project.glob("*/run_metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["error"]["out_of_memory"] is True
    assert metadata["error"]["imgsz"] == 1280
    assert metadata["error"]["batch"] == -1
    assert "Lower training.batch" in metadata["error"]["recommendation"]
    assert not paths.artifacts_root.joinpath("candidates").exists()


def test_cuda_preflight_failure_is_recorded_before_model_loading(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    settings, paths = _settings_and_paths(tmp_path, prepared_tiny_dataset)
    factory_called = False

    def factory(checkpoint: str) -> FakeYolo:
        nonlocal factory_called
        factory_called = True
        return FakeYolo(checkpoint)

    with pytest.raises(EnvironmentValidationError, match="CUDA is required"):
        run_training(
            settings,
            paths,
            allow_cpu=False,
            yolo_factory=factory,
            environment_report=_environment_report(cuda_available=False),
            dependency_freeze=(),
            now=lambda: FIXED_TIME,
        )

    assert factory_called is False
    metadata = json.loads(next(settings.project.glob("*/run_metadata.json")).read_text())
    assert metadata["status"] == "failed"
    assert metadata["error"]["phase"] == "environment_preflight"


def test_failed_run_resumes_in_place_with_original_identity(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    settings, paths = _settings_and_paths(tmp_path, prepared_tiny_dataset)
    failing_model = FakeYolo(settings.model, failure=RuntimeError("interrupted"))
    with pytest.raises(TrainingExecutionError):
        run_training(
            settings,
            paths,
            allow_cpu=True,
            yolo_factory=lambda checkpoint: failing_model,
            environment_report=_environment_report(cuda_available=False),
            dependency_freeze=(),
            now=lambda: FIXED_TIME,
        )

    run_directory = next(settings.project.iterdir())
    last_checkpoint = run_directory / "weights" / "last.pt"
    last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint.write_bytes(b"partial-checkpoint")
    context = load_resume_context(
        last_checkpoint,
        dataset_fingerprint=read_dataset_fingerprint(settings.data),
        allow_dataset_change=False,
    )
    resumed_model = FakeYolo(str(last_checkpoint))

    result = run_training(
        settings,
        paths,
        allow_cpu=True,
        resume=context,
        yolo_factory=lambda checkpoint: resumed_model,
        environment_report=_environment_report(cuda_available=False),
        dependency_freeze=(),
        now=lambda: FIXED_TIME,
    )

    assert result.resumed is True
    assert result.run_directory == run_directory
    assert resumed_model.calls == [{"resume": True, "device": "cpu"}]
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "success"
    assert metadata["resume_history"][0]["checkpoint"] == str(last_checkpoint)


def _settings_and_paths(
    tmp_path: Path,
    prepared_dataset: PreparationResult,
) -> tuple[TrainingSettings, ProjectPaths]:
    dataset_yaml = prepared_dataset.dataset_yaml
    paths = ProjectPaths(
        project_root=Path(__file__).resolve().parents[1],
        data_root=tmp_path / "data",
        runs_root=tmp_path / "runs",
        artifacts_root=tmp_path / "artifacts",
        reports_root=tmp_path / "reports",
        models_root=tmp_path / "models",
        configs_root=tmp_path / "configs",
    )
    settings = TrainingSettings(
        source=tmp_path / "train.yaml",
        model="yolo26n.pt",
        data=dataset_yaml,
        project=paths.runs_root / "train",
        name_prefix="yolo26n_fod_phase1_1280",
        training_arguments={
            "amp": True,
            "batch": -1,
            "cache": False,
            "close_mosaic": 10,
            "deterministic": True,
            "device": 0,
            "epochs": 100,
            "imgsz": 1280,
            "multi_scale": 0.0,
            "optimizer": "auto",
            "patience": 30,
            "plots": True,
            "pretrained": True,
            "rect": False,
            "save": True,
            "save_period": 10,
            "seed": 42,
            "val": True,
            "workers": 8,
        },
        metadata={"experiment_description": "fixture", "tags": ["test"]},
    )
    return settings, paths


def _environment_report(*, cuda_available: bool) -> EnvironmentReport:
    packages = {
        name: PackageReport(module=name, available=True, version="fixture", error=None)
        for name in ("opencv", "torch", "torchvision", "ultralytics")
    }
    return EnvironmentReport(
        generated_at_utc="2026-07-14T02:15:00Z",
        python_version="3.14.3",
        python_executable="python",
        python_implementation="CPython",
        operating_system="Windows",
        platform_release="fixture",
        machine="AMD64",
        processor="fixture",
        packages=packages,
        cuda_available=cuda_available,
        cuda_device_count=1 if cuda_available else 0,
        cuda_current_device=0 if cuda_available else None,
        torch_cuda_version="13.0" if cuda_available else None,
        cudnn_version=9999 if cuda_available else None,
        cuda_smoke_test="passed" if cuda_available else "failed",
        cuda_error=None if cuda_available else "CUDA unavailable",
        gpus=(GpuReport(0, "Fixture GPU", 1024, "9.0"),) if cuda_available else (),
        nvidia_smi=NvidiaSmiReport(False, None, None, (), "fixture"),
        git=GitReport(commit="abcdef1234567890", dirty=True, error=None),
        model_check=ModelCheckReport(None, False, None, None),
    )
