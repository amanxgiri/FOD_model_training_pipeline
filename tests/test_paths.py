"""Tests for portable path resolution and containment checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from fod_yolo.paths import (
    PathConfigurationError,
    ProjectPaths,
    discover_project_root,
    ensure_within_root,
    resolve_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_root_is_discovered_from_a_nested_directory() -> None:
    assert discover_project_root(PROJECT_ROOT / "src" / "fod_yolo") == PROJECT_ROOT


def test_default_project_paths_are_repository_relative() -> None:
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT, environment={})

    assert paths.project_root == PROJECT_ROOT
    assert paths.data_root == PROJECT_ROOT / "data"
    assert paths.runs_root == PROJECT_ROOT / "runs"
    assert paths.artifacts_root == PROJECT_ROOT / "artifacts"
    assert paths.reports_root == PROJECT_ROOT / "reports"


def test_relative_environment_overrides_are_anchored_to_project() -> None:
    paths = ProjectPaths.from_environment(
        project_root=PROJECT_ROOT,
        environment={
            "FOD_DATA_ROOT": "workspace/data",
            "FOD_RUNS_ROOT": "workspace/runs",
            "FOD_ARTIFACTS_ROOT": "workspace/artifacts",
        },
    )

    assert paths.data_root == PROJECT_ROOT / "workspace" / "data"
    assert paths.runs_root == PROJECT_ROOT / "workspace" / "runs"
    assert paths.artifacts_root == PROJECT_ROOT / "workspace" / "artifacts"


def test_absolute_environment_override_is_an_explicit_external_root() -> None:
    external_data = (PROJECT_ROOT.parent / "training-device-data").resolve()

    paths = ProjectPaths.from_environment(
        project_root=PROJECT_ROOT,
        environment={"FOD_DATA_ROOT": str(external_data)},
    )

    assert paths.data_root == external_data


def test_configured_data_paths_follow_the_data_root_override() -> None:
    external_data = (PROJECT_ROOT.parent / "training-device-data").resolve()
    paths = ProjectPaths.from_environment(
        project_root=PROJECT_ROOT,
        environment={"FOD_DATA_ROOT": str(external_data)},
    )

    assert paths.resolve_data_path("data/raw/fod_a") == external_data / "raw" / "fod_a"
    assert paths.resolve_data_path("data/processed/fod_a") == external_data / "processed" / "fod_a"


def test_run_and_artifact_paths_follow_their_runtime_roots() -> None:
    paths = ProjectPaths.from_environment(
        project_root=PROJECT_ROOT,
        environment={
            "FOD_RUNS_ROOT": "external/runs",
            "FOD_ARTIFACTS_ROOT": "external/artifacts",
        },
    )

    assert paths.resolve_runs_path("runs/train") == PROJECT_ROOT / "external" / "runs" / "train"
    assert paths.resolve_artifacts_path("artifacts/candidates") == (
        PROJECT_ROOT / "external" / "artifacts" / "candidates"
    )


def test_resolve_path_anchors_relative_values() -> None:
    assert resolve_path("configs/dataset.yaml", relative_to=PROJECT_ROOT) == (
        PROJECT_ROOT / "configs" / "dataset.yaml"
    )


def test_containment_check_accepts_child_and_rejects_sibling() -> None:
    data_root = PROJECT_ROOT / "data"
    assert ensure_within_root(data_root / "raw", data_root) == data_root / "raw"

    with pytest.raises(PathConfigurationError, match="outside its allowed root"):
        ensure_within_root(PROJECT_ROOT / "models", data_root, description="dataset path")
