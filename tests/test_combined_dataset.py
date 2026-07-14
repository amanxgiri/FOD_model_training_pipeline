"""Integration tests for real-plus-synthetic fine-tuning data preparation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fod_yolo.dataset.combine import CombinedDatasetSettings, prepare_combined_dataset
from fod_yolo.dataset.pipeline import PreparationResult


def test_combined_dataset_preserves_splits_and_prefixes_ids(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    synthetic_root = tmp_path / "synthetic" / "Data"
    _copy_processed_as_synthetic(prepared_tiny_dataset.processed_root, synthetic_root)
    settings = CombinedDatasetSettings(
        config_source=tmp_path / "finetune_dataset.yaml",
        runway_dataset_yaml=prepared_tiny_dataset.dataset_yaml,
        synthetic_root=synthetic_root,
        processed_root=tmp_path / "processed" / "combined",
        dataset_yaml_name="fod_combined.yaml",
        image_transfer_mode="hardlink",
        seed=42,
        runway_prefix="runway",
        synthetic_prefix="synthetic",
    )

    result = prepare_combined_dataset(settings)

    assert result.rebuilt is True
    assert result.validation_report.status == "pass"
    assert result.validation_report.counts == {
        "test_images": 2,
        "test_objects": 0,
        "train_images": 6,
        "train_objects": 8,
        "val_images": 2,
        "val_objects": 0,
    }
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["dataset_fingerprint"]) == 64
    assert set(manifest["source_datasets"]) == {"runway", "synthetic"}
    for split in ("train", "val", "test"):
        identifiers = manifest["splits"][split]
        assert any(identifier.startswith("runway__") for identifier in identifiers)
        assert any(identifier.startswith("synthetic__") for identifier in identifiers)
        assert len(identifiers) == len(set(identifiers))
    source_image = next((synthetic_root / "images" / "train").iterdir())
    combined_image = next(
        path
        for path in (result.processed_root / "images" / "train").iterdir()
        if path.stem == f"synthetic__{source_image.stem}"
    )
    assert source_image.stat().st_ino == combined_image.stat().st_ino


def test_complete_combined_dataset_is_reused(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    synthetic_root = tmp_path / "synthetic" / "Data"
    _copy_processed_as_synthetic(prepared_tiny_dataset.processed_root, synthetic_root)
    settings = CombinedDatasetSettings(
        config_source=tmp_path / "finetune_dataset.yaml",
        runway_dataset_yaml=prepared_tiny_dataset.dataset_yaml,
        synthetic_root=synthetic_root,
        processed_root=tmp_path / "processed" / "combined",
        dataset_yaml_name="fod_combined.yaml",
        image_transfer_mode="hardlink",
        seed=42,
        runway_prefix="runway",
        synthetic_prefix="synthetic",
    )

    prepare_combined_dataset(settings)
    reused = prepare_combined_dataset(settings)

    assert reused.rebuilt is False
    assert reused.validation_report.status == "pass"


def _copy_processed_as_synthetic(source: Path, destination: Path) -> None:
    for split in ("train", "val", "test"):
        shutil.copytree(source / "images" / split, destination / "images" / split)
        shutil.copytree(source / "labels" / split, destination / "labels" / split)
