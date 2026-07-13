"""Tests for safety metrics and threshold reference selection."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fod_yolo.evaluation.matcher import NormalizedBox
from fod_yolo.evaluation.threshold_sweep import (
    ImageDetections,
    build_thresholds,
    evaluate_threshold,
    select_threshold_references,
)


def test_threshold_metrics_include_false_negatives_and_small_objects() -> None:
    images = (
        ImageDetections(
            "image-a",
            (
                NormalizedBox(0.0, 0.0, 0.1, 0.1),
                NormalizedBox(0.5, 0.5, 1.0, 1.0),
            ),
            (
                NormalizedBox(0.0, 0.0, 0.1, 0.1, confidence=0.9),
                NormalizedBox(0.2, 0.2, 0.3, 0.3, confidence=0.8),
            ),
        ),
    )

    row = evaluate_threshold(
        images,
        confidence_threshold=0.5,
        iou_threshold=0.5,
        small_area_threshold=0.01,
    )

    assert row.true_positives == 1
    assert row.false_positives == 1
    assert row.false_negatives == 1
    assert row.precision == pytest.approx(0.5)
    assert row.recall == pytest.approx(0.5)
    assert row.false_negative_rate == pytest.approx(0.5)
    assert row.small_object_ground_truth_count == 1
    assert row.small_object_recall == pytest.approx(1.0)


def test_threshold_references_are_deterministic_on_ties() -> None:
    base = evaluate_threshold(
        (ImageDetections("empty", (), ()),),
        confidence_threshold=0.1,
        iou_threshold=0.5,
        small_area_threshold=0.01,
    )
    rows = (
        replace(base, threshold=0.1, recall=0.9, f1=0.7, false_positives_per_image=2.0),
        replace(base, threshold=0.2, recall=0.9, f1=0.8, false_positives_per_image=1.0),
        replace(base, threshold=0.3, recall=0.89, f1=0.8, false_positives_per_image=0.5),
    )

    references = select_threshold_references(rows)

    assert references["best_f1_threshold"] == 0.3
    assert references["max_recall_threshold"] == 0.2
    assert references["balanced_high_recall_threshold"] == 0.3


def test_threshold_builder_is_inclusive_and_inserts_default() -> None:
    assert build_thresholds(0.05, 0.15, 0.05, 0.12) == (0.05, 0.1, 0.12, 0.15)
