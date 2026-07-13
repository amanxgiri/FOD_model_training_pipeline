"""Tests for Kaggle authentication, commands, and safe archive extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from fod_yolo.dataset import DatasetDownloadError, KaggleAuthenticationError
from fod_yolo.dataset.kaggle_client import (
    build_kaggle_download_command,
    download_and_extract_dataset,
    resolve_kaggle_authentication,
    safe_extract_zip,
)
from fod_yolo.hashing import atomic_write_json, sha256_file


def test_environment_credentials_are_accepted_without_retaining_secrets() -> None:
    authentication = resolve_kaggle_authentication(
        environment={"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "secret"}
    )

    assert authentication.method == "environment"
    assert authentication.config_file is None
    assert "secret" not in repr(authentication)


def test_partial_environment_credentials_are_rejected() -> None:
    with pytest.raises(KaggleAuthenticationError, match="must both be set"):
        resolve_kaggle_authentication(
            environment={"KAGGLE_USERNAME": "user"},
            home_directory=Path("missing-home"),
        )


def test_kaggle_json_credentials_are_discovered(tmp_path: Path) -> None:
    config_directory = tmp_path / "kaggle-config"
    config_directory.mkdir()
    (config_directory / "kaggle.json").write_text("{}", encoding="utf-8")

    authentication = resolve_kaggle_authentication(
        environment={"KAGGLE_CONFIG_DIR": str(config_directory)}
    )

    assert authentication.method == "kaggle_json"
    assert authentication.config_file == config_directory / "kaggle.json"


def test_versioned_kaggle_command_is_shell_free(tmp_path: Path) -> None:
    command = build_kaggle_download_command(
        dataset_slug="owner/dataset",
        output_directory=tmp_path,
        version="3",
        force=True,
    )

    assert command[:4] == ("kaggle", "datasets", "download", "-d")
    assert "owner/dataset/3" in command
    assert "--force" in command


def test_safe_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("../escape.txt", "unsafe")

    with pytest.raises(DatasetDownloadError, match="Unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "extracted")

    assert not (tmp_path / "escape.txt").exists()


def test_safe_zip_extraction_is_idempotent(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("wrapper/VOC/readme.txt", "fixture")
    destination = tmp_path / "extracted"

    assert safe_extract_zip(archive, destination)
    assert (destination / "wrapper" / "VOC" / "readme.txt").is_file()
    assert safe_extract_zip(archive, destination) is False


def test_valid_cached_download_skips_credentials_and_network(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    downloads = raw_root / "downloads"
    downloads.mkdir(parents=True)
    archive = downloads / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("wrapper/VOC/readme.txt", "fixture")
    atomic_write_json(
        raw_root / "source_manifest.json",
        {
            "archive_path": "downloads/dataset.zip",
            "archive_sha256": sha256_file(archive),
            "archive_size_bytes": archive.stat().st_size,
            "dataset_slug": "owner/dataset",
            "downloaded_at_utc": "2026-07-14T00:00:00Z",
            "kaggle_cli_version": "fixture",
            "requested_version": None,
            "resolved_version": None,
        },
    )

    first = download_and_extract_dataset(
        dataset_slug="owner/dataset",
        raw_root=raw_root,
        archive_name="dataset.zip",
        version=None,
        force_download=False,
        force_extract=False,
        environment={},
    )
    second = download_and_extract_dataset(
        dataset_slug="owner/dataset",
        raw_root=raw_root,
        archive_name="dataset.zip",
        version=None,
        force_download=False,
        force_extract=False,
        environment={},
    )

    assert first.downloaded is False
    assert first.extracted is True
    assert second.downloaded is False
    assert second.extracted is False
