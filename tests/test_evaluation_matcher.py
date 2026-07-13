"""Hand-calculated tests for deterministic one-to-one matching."""

from __future__ import annotations

import pytest

from fod_yolo.evaluation.matcher import NormalizedBox, intersection_over_union, match_detections


def test_iou_matches_hand_calculated_overlap() -> None:
    first = NormalizedBox(0.0, 0.0, 0.5, 0.5)
    second = NormalizedBox(0.25, 0.25, 0.75, 0.75)

    assert intersection_over_union(first, second) == pytest.approx(1.0 / 7.0)


def test_matching_is_confidence_ordered_and_one_to_one() -> None:
    ground_truth = (
        NormalizedBox(0.0, 0.0, 0.4, 0.4),
        NormalizedBox(0.6, 0.6, 1.0, 1.0),
    )
    predictions = (
        NormalizedBox(0.0, 0.0, 0.4, 0.4, confidence=0.80),
        NormalizedBox(0.0, 0.0, 0.4, 0.4, confidence=0.95),
        NormalizedBox(0.6, 0.6, 1.0, 1.0, confidence=0.40),
    )

    result = match_detections(
        ground_truth,
        predictions,
        confidence_threshold=0.50,
        iou_threshold=0.50,
    )

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.matches[0].prediction_index == 1
    assert result.false_positive_indices == (0,)
    assert result.false_negative_indices == (1,)
