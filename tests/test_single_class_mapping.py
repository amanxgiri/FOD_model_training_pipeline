"""Integration checks for explicit single-class label storage."""

from __future__ import annotations

import json

import pytest
import yaml

from fod_yolo.dataset.pipeline import DatasetSettings, PreparationResult, prepare_dataset


@pytest.mark.smoke
def test_tiny_voc_pipeline_builds_valid_single_class_dataset(
    prepared_tiny_dataset: PreparationResult,
) -> None:
    result = prepared_tiny_dataset

    assert result.rebuilt is True
    assert result.validation_report.status == "pass"
    dataset_yaml = yaml.safe_load(result.dataset_yaml.read_text(encoding="utf-8"))
    assert dataset_yaml["names"] == {0: "FOD"}

    labels = sorted(result.processed_root.glob("labels/*/*.txt"))
    assert len(labels) == 5
    for label in labels:
        for line in label.read_text(encoding="utf-8").splitlines():
            assert line.split()[0] == "0"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    statistics = json.loads(result.statistics_path.read_text(encoding="utf-8"))
    assert len(manifest["dataset_fingerprint"]) == 64
    assert manifest["counts"] == {"test_images": 1, "train_images": 3, "val_images": 1}
    assert statistics["annotation_repairs"]["dimension_mismatches"] == 1
    assert statistics["annotation_repairs"]["rejected_objects"] == 1


def test_empty_images_keep_empty_label_files(prepared_tiny_dataset: PreparationResult) -> None:
    empty_labels = [
        label
        for label in prepared_tiny_dataset.processed_root.glob("labels/*/*.txt")
        if not label.read_text(encoding="utf-8").strip()
    ]

    assert {label.stem for label in empty_labels} == {"image003", "image005"}


def test_incomplete_processed_directory_is_rebuilt_without_force(
    tiny_dataset_settings: DatasetSettings,
) -> None:
    processed_root = tiny_dataset_settings.processed_root
    processed_root.mkdir(parents=True)
    (processed_root / "validation_report.json").write_text("{}\n", encoding="utf-8")

    result = prepare_dataset(tiny_dataset_settings)

    assert result.rebuilt is True
    assert result.validation_report.status == "pass"
    assert result.manifest_path.is_file()
    report_text = (processed_root / "validation_report.json").read_text(encoding="utf-8")
    assert report_text.strip() != "{}"
