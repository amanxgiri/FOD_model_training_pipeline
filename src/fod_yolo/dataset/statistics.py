"""Processed-dataset statistics derived from conversion records."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable

from fod_yolo.dataset.convert import ConvertedImage


def build_dataset_statistics(
    records: Iterable[ConvertedImage],
    *,
    small_area_max: float = 0.01,
    medium_area_max: float = 0.10,
) -> dict[str, object]:
    """Aggregate split counts, source classes, dimensions, box areas, and repairs."""

    materialized = tuple(records)
    if not 0.0 < small_area_max < medium_area_max <= 1.0:
        raise ValueError("Area thresholds must satisfy 0 < small < medium <= 1")

    split_statistics: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_records = [
            record for record in materialized if record.split == split_name and record.included
        ]
        split_statistics[split_name] = {
            "empty_images": sum(not record.boxes for record in split_records),
            "images": len(split_records),
            "objects": sum(len(record.boxes) for record in split_records),
        }

    original_categories = Counter(
        class_name for record in materialized for class_name in record.original_classes
    )
    included_records = [record for record in materialized if record.included]
    areas = [box.area_ratio for record in included_records for box in record.boxes]
    widths = [float(record.width) for record in included_records]
    heights = [float(record.height) for record in included_records]
    total_objects = len(areas)

    return {
        "annotation_repairs": {
            "dimension_mismatches": sum(record.dimension_mismatch for record in materialized),
            "rejected_objects": sum(len(record.rejected_objects) for record in materialized),
        },
        "box_area_buckets": {
            "large": sum(area > medium_area_max for area in areas),
            "medium": sum(small_area_max < area <= medium_area_max for area in areas),
            "small": sum(area <= small_area_max for area in areas),
        },
        "box_area_ratio": _distribution(areas),
        "bucket_thresholds": {
            "medium_area_max": medium_area_max,
            "small_area_max": small_area_max,
        },
        "final_class_distribution": {"FOD": total_objects},
        "image_height": _distribution(heights),
        "image_width": _distribution(widths),
        "original_category_counts": dict(sorted(original_categories.items())),
        "schema_version": "1.0",
        "splits": split_statistics,
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "maximum": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "p95": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "minimum": ordered[0],
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
