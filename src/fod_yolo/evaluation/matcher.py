"""Deterministic single-class one-to-one detection matching."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fod_yolo.evaluation import EvaluationDataError


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """Normalized XYXY box with optional confidence and source identity."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    confidence: float | None = None

    def validate(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise EvaluationDataError("Box coordinates must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise EvaluationDataError("Normalized box coordinates must be within [0, 1]")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise EvaluationDataError("Box must have positive width and height")
        if self.class_id != 0:
            raise EvaluationDataError(
                f"Single-class evaluation requires class 0, got {self.class_id}"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise EvaluationDataError("Prediction confidence must be within [0, 1]")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class DetectionMatch:
    """One prediction-to-ground-truth assignment."""

    prediction_index: int
    ground_truth_index: int
    iou: float


@dataclass(frozen=True, slots=True)
class ImageMatchResult:
    """Complete matching result for one image."""

    matches: tuple[DetectionMatch, ...]
    false_positive_indices: tuple[int, ...]
    false_negative_indices: tuple[int, ...]

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        return len(self.false_positive_indices)

    @property
    def false_negatives(self) -> int:
        return len(self.false_negative_indices)


def intersection_over_union(first: NormalizedBox, second: NormalizedBox) -> float:
    """Return axis-aligned IoU for two validated normalized boxes."""

    first.validate()
    second.validate()
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    union = first.area + second.area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_detections(
    ground_truth: tuple[NormalizedBox, ...],
    predictions: tuple[NormalizedBox, ...],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> ImageMatchResult:
    """Match descending-confidence predictions to the highest-IoU unmatched target."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be within (0, 1]")
    for box in ground_truth:
        box.validate()
    for box in predictions:
        box.validate()
        if box.confidence is None:
            raise EvaluationDataError("Every prediction must contain confidence")

    selected = [
        index
        for index, prediction in enumerate(predictions)
        if prediction.confidence is not None and prediction.confidence >= confidence_threshold
    ]
    selected.sort(key=lambda index: (-float(predictions[index].confidence or 0.0), index))
    unmatched_ground_truth = set(range(len(ground_truth)))
    matches: list[DetectionMatch] = []
    false_positives: list[int] = []
    for prediction_index in selected:
        candidates = [
            (intersection_over_union(predictions[prediction_index], ground_truth[target]), target)
            for target in sorted(unmatched_ground_truth)
        ]
        if not candidates:
            false_positives.append(prediction_index)
            continue
        best_iou, best_target = max(candidates, key=lambda item: (item[0], -item[1]))
        if best_iou >= iou_threshold:
            unmatched_ground_truth.remove(best_target)
            matches.append(DetectionMatch(prediction_index, best_target, best_iou))
        else:
            false_positives.append(prediction_index)
    return ImageMatchResult(
        matches=tuple(matches),
        false_positive_indices=tuple(false_positives),
        false_negative_indices=tuple(sorted(unmatched_ground_truth)),
    )
