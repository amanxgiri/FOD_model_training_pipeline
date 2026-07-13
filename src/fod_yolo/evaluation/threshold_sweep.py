"""Safety metrics and deterministic confidence-threshold reference selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal

from fod_yolo.evaluation.matcher import NormalizedBox, match_detections


@dataclass(frozen=True, slots=True)
class ImageDetections:
    """Ground truth and minimum-confidence predictions for one image."""

    image_id: str
    ground_truth: tuple[NormalizedBox, ...]
    predictions: tuple[NormalizedBox, ...]


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Project-controlled metrics at one confidence threshold."""

    threshold: float
    images: int
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_negative_rate: float
    false_positives_per_image: float
    images_with_false_negatives: int
    fraction_of_images_with_false_negatives: float
    small_object_ground_truth_count: int
    small_object_true_positives: int
    small_object_false_negatives: int
    small_object_recall: float
    small_object_false_negative_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def build_thresholds(start: float, stop: float, step: float, include: float) -> tuple[float, ...]:
    """Build an inclusive decimal-stable threshold series plus one required value."""

    if not 0.0 <= start <= stop <= 1.0 or step <= 0.0 or not 0.0 <= include <= 1.0:
        raise ValueError("Invalid confidence threshold sweep configuration")
    current = Decimal(str(start))
    end = Decimal(str(stop))
    increment = Decimal(str(step))
    values: set[Decimal] = {Decimal(str(include))}
    while current <= end:
        values.add(current)
        current += increment
    return tuple(float(value) for value in sorted(values))


def evaluate_threshold(
    images: tuple[ImageDetections, ...],
    *,
    confidence_threshold: float,
    iou_threshold: float,
    small_area_threshold: float,
) -> ThresholdMetrics:
    """Aggregate matching and safety metrics across every image."""

    true_positives = false_positives = false_negatives = predictions = 0
    images_with_false_negatives = 0
    small_ground_truth = small_true_positives = 0
    for image in images:
        result = match_detections(
            image.ground_truth,
            image.predictions,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        true_positives += result.true_positives
        false_positives += result.false_positives
        false_negatives += result.false_negatives
        predictions += result.true_positives + result.false_positives
        images_with_false_negatives += bool(result.false_negative_indices)
        small_indices = {
            index
            for index, box in enumerate(image.ground_truth)
            if box.area <= small_area_threshold + 1e-12
        }
        small_ground_truth += len(small_indices)
        matched_small = sum(match.ground_truth_index in small_indices for match in result.matches)
        small_true_positives += matched_small
    small_false_negatives = small_ground_truth - small_true_positives
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    return ThresholdMetrics(
        threshold=confidence_threshold,
        images=len(images),
        ground_truth=true_positives + false_negatives,
        predictions=predictions,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=_ratio(2.0 * precision * recall, precision + recall),
        false_negative_rate=_ratio(false_negatives, true_positives + false_negatives),
        false_positives_per_image=_ratio(false_positives, len(images)),
        images_with_false_negatives=images_with_false_negatives,
        fraction_of_images_with_false_negatives=_ratio(images_with_false_negatives, len(images)),
        small_object_ground_truth_count=small_ground_truth,
        small_object_true_positives=small_true_positives,
        small_object_false_negatives=small_false_negatives,
        small_object_recall=_ratio(small_true_positives, small_ground_truth),
        small_object_false_negative_rate=_ratio(small_false_negatives, small_ground_truth),
    )


def select_threshold_references(
    rows: tuple[ThresholdMetrics, ...],
    *,
    recall_tolerance: float = 0.01,
) -> dict[str, float]:
    """Select best-F1, highest max-recall, and balanced high-recall references."""

    if not rows:
        raise ValueError("At least one threshold row is required")
    best_f1 = max(rows, key=lambda row: (row.f1, row.threshold))
    maximum_recall = max(row.recall for row in rows)
    max_recall = max(
        (row for row in rows if abs(row.recall - maximum_recall) <= 1e-12),
        key=lambda row: row.threshold,
    )
    high_recall = [row for row in rows if row.recall >= maximum_recall - recall_tolerance]
    balanced = min(high_recall, key=lambda row: (row.false_positives_per_image, -row.threshold))
    return {
        "balanced_high_recall_threshold": balanced.threshold,
        "best_f1_threshold": best_f1.threshold,
        "max_recall_threshold": max_recall.threshold,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
