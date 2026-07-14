"""Shared fixture-backed dataset pipeline setup."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fod_yolo.dataset.convert import ConversionOptions
from fod_yolo.dataset.pipeline import (
    DatasetSettings,
    PreparationResult,
    SplitOptions,
    prepare_dataset,
)
from fod_yolo.hashing import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY_VOC_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "tiny_voc"


@pytest.fixture
def tiny_voc_root() -> Path:
    """Return the immutable checked-in Pascal VOC fixture root."""

    return TINY_VOC_ROOT


@pytest.fixture
def tiny_dataset_settings(tmp_path: Path) -> DatasetSettings:
    """Create portable settings backed by a temporary copy of the VOC fixture."""

    raw_root = tmp_path / "data" / "raw" / "fod_a"
    extracted_voc = raw_root / "extracted" / "wrapper" / "VOC"
    shutil.copytree(TINY_VOC_ROOT, extracted_voc)
    atomic_write_json(
        raw_root / "source_manifest.json",
        {
            "archive_path": "downloads/fod_a.zip",
            "archive_sha256": "a" * 64,
            "archive_size_bytes": 123,
            "dataset_slug": "fixture/tiny-voc",
            "downloaded_at_utc": "2026-07-14T00:00:00Z",
            "kaggle_cli_version": "fixture",
            "requested_version": None,
            "resolved_version": None,
            "schema_version": "1.0",
        },
    )
    return DatasetSettings(
        kaggle_slug="fixture/tiny-voc",
        kaggle_version=None,
        raw_root=raw_root,
        processed_root=tmp_path / "data" / "processed" / "tiny_yolo",
        archive_name="fod_a.zip",
        force_download=False,
        force_extract=False,
        conversion=ConversionOptions(),
        split=SplitOptions(
            seed=42,
            validation_fraction=0.25,
            preserve_official_test=True,
            trainval_file=Path("ImageSets/Main/trainval.txt"),
            test_file=Path("ImageSets/Main/test.txt"),
        ),
    )


@pytest.fixture
def prepared_tiny_dataset(tiny_dataset_settings: DatasetSettings) -> PreparationResult:
    """Build and strictly validate a complete temporary YOLO fixture dataset."""

    return prepare_dataset(tiny_dataset_settings)
