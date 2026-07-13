"""Framework timing aggregation for reproducible evaluation reports."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LatencySummary:
    mean_preprocess_ms: float
    mean_inference_ms: float
    mean_postprocess_ms: float
    mean_end_to_end_ms: float
    median_end_to_end_ms: float
    p95_end_to_end_ms: float
    fps: float

    def to_dict(self) -> dict[str, float]:
        return {
            "fps": self.fps,
            "mean_end_to_end_ms": self.mean_end_to_end_ms,
            "mean_inference_ms": self.mean_inference_ms,
            "mean_postprocess_ms": self.mean_postprocess_ms,
            "mean_preprocess_ms": self.mean_preprocess_ms,
            "median_end_to_end_ms": self.median_end_to_end_ms,
            "p95_end_to_end_ms": self.p95_end_to_end_ms,
        }


def summarize_latency(samples: tuple[dict[str, float], ...]) -> LatencySummary:
    """Aggregate per-image Ultralytics stage timings in milliseconds."""

    if not samples:
        raise ValueError("Latency summary requires at least one sample")
    if any(
        not math.isfinite(value) or value < 0.0 for sample in samples for value in sample.values()
    ):
        raise ValueError("Latency samples must be finite and non-negative")
    preprocess = [sample.get("preprocess", 0.0) for sample in samples]
    inference = [sample.get("inference", 0.0) for sample in samples]
    postprocess = [sample.get("postprocess", 0.0) for sample in samples]
    end_to_end = [sum(values) for values in zip(preprocess, inference, postprocess, strict=True)]
    mean_end = statistics.fmean(end_to_end)
    ordered = sorted(end_to_end)
    return LatencySummary(
        mean_preprocess_ms=statistics.fmean(preprocess),
        mean_inference_ms=statistics.fmean(inference),
        mean_postprocess_ms=statistics.fmean(postprocess),
        mean_end_to_end_ms=mean_end,
        median_end_to_end_ms=statistics.median(ordered),
        p95_end_to_end_ms=_percentile(ordered, 0.95),
        fps=1000.0 / mean_end if mean_end > 0.0 else 0.0,
    )


def _percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
