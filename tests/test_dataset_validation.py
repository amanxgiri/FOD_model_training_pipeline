"""Tests for processed-dataset validation and manifest checks."""

from __future__ import annotations

import pytest

from fod_yolo.dataset.pipeline import PreparationResult
from fod_yolo.dataset.validate import (
    StrictDatasetValidationError,
    validate_yolo_dataset,
)


def test_generated_fixture_passes_strict_validation(
    prepared_tiny_dataset: PreparationResult,
) -> None:
    report = validate_yolo_dataset(prepared_tiny_dataset.dataset_yaml, strict=True)

    assert report.status == "pass"
    assert report.class_ids_observed == (0,)
    assert sum(report.counts[key] for key in report.counts if key.endswith("_images")) == 5


def test_validator_reports_nonzero_class_id(
    prepared_tiny_dataset: PreparationResult,
) -> None:
    label = next(
        path
        for path in prepared_tiny_dataset.processed_root.glob("labels/*/*.txt")
        if path.read_text(encoding="utf-8").strip()
    )
    original = label.read_text(encoding="utf-8")
    label.write_text(f"1{original[1:]}", encoding="utf-8")

    report = validate_yolo_dataset(prepared_tiny_dataset.dataset_yaml)

    assert report.status == "fail"
    assert any("class ID 1" in error for error in report.errors)
    with pytest.raises(StrictDatasetValidationError):
        validate_yolo_dataset(prepared_tiny_dataset.dataset_yaml, strict=True)


def test_dimension_mismatch_uses_actual_image_size(
    prepared_tiny_dataset: PreparationResult,
) -> None:
    record = next(
        record for record in prepared_tiny_dataset.records if record.image_id == "image004"
    )

    assert record.dimension_mismatch is True
    assert (record.xml_width, record.xml_height) == (20, 20)
    assert (record.width, record.height) == (10, 10)
