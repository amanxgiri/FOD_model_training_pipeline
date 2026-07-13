"""Tests for the fixed Phase 1 training configuration contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from fod_yolo.paths import ProjectPaths
from fod_yolo.training import TrainingConfigurationError
from fod_yolo.training.config import load_training_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_configuration_resolves_portable_paths(tmp_path: Path) -> None:
    paths = ProjectPaths.from_environment(
        project_root=PROJECT_ROOT,
        environment={
            "FOD_DATA_ROOT": str(tmp_path / "data"),
            "FOD_RUNS_ROOT": str(tmp_path / "runs"),
        },
    )

    settings = load_training_settings(
        PROJECT_ROOT / "configs" / "train_yolo26n_1280.yaml",
        paths,
        overrides=("training.epochs=5",),
    )

    assert (
        settings.data == tmp_path / "data" / "processed" / "fod_a_single_class_yolo" / "fod_a.yaml"
    )
    assert settings.project == tmp_path / "runs" / "train"
    assert settings.training_arguments["epochs"] == 5
    assert settings.training_arguments["imgsz"] == 1280


def test_phase1_image_size_cannot_be_overridden() -> None:
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT, environment={})

    with pytest.raises(TrainingConfigurationError, match="imgsz=1280"):
        load_training_settings(
            PROJECT_ROOT / "configs" / "train_yolo26n_1280.yaml",
            paths,
            overrides=("training.imgsz=640",),
        )
