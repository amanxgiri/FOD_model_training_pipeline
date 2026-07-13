"""Tests for environment reports and explicit PyTorch installation commands."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from fod_yolo.environment import (
    EnvironmentValidationError,
    ModelCheckReport,
    PackageReport,
    build_torch_install_command,
    collect_environment_report,
    parse_nvidia_smi_output,
    resolve_torch_index_url,
    validate_environment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parse_nvidia_smi_output() -> None:
    rows = parse_nvidia_smi_output("0, NVIDIA RTX 4090, 24564, 590.10\n")

    assert len(rows) == 1
    assert rows[0].index == 0
    assert rows[0].name == "NVIDIA RTX 4090"
    assert rows[0].memory_total_mb == 24564
    assert rows[0].driver_version == "590.10"


def test_parse_nvidia_smi_output_rejects_malformed_rows() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_nvidia_smi_output("0, incomplete\n")


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("cpu", "https://download.pytorch.org/whl/cpu"),
        ("cu128", "https://download.pytorch.org/whl/cu128"),
        ("rocm6.3", "https://download.pytorch.org/whl/rocm6.3"),
    ),
)
def test_named_torch_profiles_resolve_to_official_index(profile: str, expected: str) -> None:
    assert resolve_torch_index_url(index_url=None, profile=profile) == expected


def test_torch_index_rejects_untrusted_hosts() -> None:
    with pytest.raises(ValueError, match="official PyTorch"):
        resolve_torch_index_url(index_url="https://example.com/whl/cu128", profile=None)


def test_torch_install_command_uses_selected_python_and_versions() -> None:
    command = build_torch_install_command(
        index_url="https://download.pytorch.org/whl/cpu",
        torch_version="2.9.0",
        torchvision_version="0.24.0",
        python_executable=sys.executable,
    )

    assert command[:4] == (sys.executable, "-m", "pip", "install")
    assert "torch==2.9.0" in command
    assert "torchvision==0.24.0" in command
    assert command[-1] == "https://download.pytorch.org/whl/cpu"


def test_environment_report_is_json_serializable_without_secrets() -> None:
    report = collect_environment_report(project_root=PROJECT_ROOT)
    serialized = report.to_dict()

    assert serialized["schema_version"] == "1.0"
    assert serialized["python"]["executable"] == sys.executable  # type: ignore[index]
    assert "KAGGLE_KEY" not in str(serialized)
    assert "GH_TOKEN" not in str(serialized)


def test_environment_validation_reports_missing_packages() -> None:
    report = collect_environment_report(project_root=PROJECT_ROOT)
    available_packages = {
        name: PackageReport(module=name, available=True, version="test", error=None)
        for name in ("opencv", "torch", "torchvision", "ultralytics")
    }
    available_packages["torch"] = PackageReport(
        module="torch",
        available=False,
        version=None,
        error="not installed",
    )
    incomplete = replace(
        report,
        packages=available_packages,
        model_check=ModelCheckReport(
            checkpoint="yolo26n.pt",
            attempted=True,
            passed=True,
            error=None,
        ),
    )

    with pytest.raises(EnvironmentValidationError, match="torch"):
        validate_environment(incomplete, require_cuda=False, require_model_check=True)
