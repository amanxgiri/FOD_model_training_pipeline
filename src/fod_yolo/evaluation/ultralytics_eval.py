"""Lazy Ultralytics validation and prediction-result normalization."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from fod_yolo.evaluation import EvaluationDataError, EvaluationExecutionError
from fod_yolo.evaluation.config import EvaluationSettings
from fod_yolo.evaluation.data import EvaluationImage
from fod_yolo.evaluation.latency import LatencySummary, summarize_latency
from fod_yolo.evaluation.matcher import NormalizedBox


class EvaluationModel(Protocol):
    def val(self, **kwargs: object) -> object: ...

    def predict(self, **kwargs: object) -> Iterable[object]: ...


ModelFactory = Callable[[str], EvaluationModel]


@dataclass(frozen=True, slots=True)
class StandardMetrics:
    precision: float
    recall: float
    map50: float
    map75: float
    map50_95: float

    def to_dict(self) -> dict[str, float]:
        return {
            "map50": self.map50,
            "map50_95": self.map50_95,
            "map75": self.map75,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class FrameworkEvaluation:
    standard_metrics: StandardMetrics
    predictions: dict[str, tuple[NormalizedBox, ...]]
    latency: LatencySummary
    peak_gpu_memory_bytes: int | None


def create_model(checkpoint: str | Path) -> EvaluationModel:
    """Load Ultralytics only for an actual evaluation command."""

    try:
        ultralytics = importlib.import_module("ultralytics")
        return cast(EvaluationModel, ultralytics.YOLO(str(checkpoint)))
    except (ImportError, AttributeError, OSError) as exc:
        raise EvaluationExecutionError(f"Unable to load Ultralytics model: {exc}") from exc


def run_framework_evaluation(
    model: EvaluationModel,
    *,
    dataset_yaml: Path,
    split: str,
    images: tuple[EvaluationImage, ...],
    settings: EvaluationSettings,
    framework_output: Path,
) -> FrameworkEvaluation:
    """Run Ultralytics val plus minimum-confidence predictions and timing capture."""

    framework_output.mkdir(parents=True, exist_ok=False)
    _reset_peak_gpu_memory()
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=settings.imgsz,
        device=settings.device,
        batch=settings.batch,
        half=settings.half,
        workers=settings.workers,
        plots=settings.plots,
        save_json=settings.save_json,
        save_txt=settings.save_txt,
        save_conf=settings.save_conf,
        max_det=settings.max_det,
        project=str(framework_output.parent),
        name=framework_output.name,
        exist_ok=True,
    )
    standard = _standard_metrics(metrics)
    for image in images[: settings.warmup_images]:
        tuple(
            model.predict(
                source=str(image.path),
                imgsz=settings.imgsz,
                device=settings.device,
                conf=settings.threshold_start,
                max_det=settings.max_det,
                half=settings.half,
                verbose=False,
                stream=True,
            )
        )
    results = model.predict(
        source=[str(image.path) for image in images],
        imgsz=settings.imgsz,
        device=settings.device,
        conf=settings.threshold_start,
        max_det=settings.max_det,
        half=settings.half,
        verbose=False,
        stream=True,
    )
    predictions: dict[str, tuple[NormalizedBox, ...]] = {}
    timing: list[dict[str, float]] = []
    for result in results:
        image_id, boxes, speed = _parse_prediction_result(result)
        if image_id in predictions:
            raise EvaluationDataError(f"Duplicate prediction result for image {image_id}")
        predictions[image_id] = boxes
        timing.append(speed)
    return FrameworkEvaluation(
        standard,
        predictions,
        summarize_latency(tuple(timing)),
        _peak_gpu_memory(),
    )


def _standard_metrics(metrics: object) -> StandardMetrics:
    box = getattr(metrics, "box", None)
    if box is None:
        raise EvaluationExecutionError("Ultralytics validation returned no box metrics")
    try:
        result = StandardMetrics(
            precision=float(box.mp),
            recall=float(box.mr),
            map50=float(box.map50),
            map75=float(box.map75),
            map50_95=float(box.map),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise EvaluationExecutionError(f"Invalid Ultralytics box metrics: {exc}") from exc
    if not all(math.isfinite(value) for value in result.to_dict().values()):
        raise EvaluationExecutionError("Ultralytics box metrics contain non-finite values")
    return result


def _parse_prediction_result(
    result: object,
) -> tuple[str, tuple[NormalizedBox, ...], dict[str, float]]:
    result_path = getattr(result, "path", None)
    if not isinstance(result_path, (str, Path)):
        raise EvaluationDataError("Prediction result has no source path")
    boxes_object = getattr(result, "boxes", None)
    boxes: list[NormalizedBox] = []
    if boxes_object is not None:
        coordinates = _to_list(getattr(boxes_object, "xyxyn", None))
        confidences = _to_list(getattr(boxes_object, "conf", None))
        classes = _to_list(getattr(boxes_object, "cls", None))
        if not (len(coordinates) == len(confidences) == len(classes)):
            raise EvaluationDataError("Prediction box arrays have inconsistent lengths")
        for coordinate, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
            if not isinstance(coordinate, list) or len(coordinate) != 4:
                raise EvaluationDataError("Prediction xyxyn row must contain four values")
            box = NormalizedBox(
                x1=_numeric(coordinate[0]),
                y1=_numeric(coordinate[1]),
                x2=_numeric(coordinate[2]),
                y2=_numeric(coordinate[3]),
                class_id=int(_numeric(class_id)),
                confidence=_numeric(confidence),
            )
            box.validate()
            boxes.append(box)
    raw_speed = getattr(result, "speed", None)
    if not isinstance(raw_speed, dict):
        raise EvaluationDataError("Prediction result has no timing dictionary")
    speed = {
        key: float(raw_speed.get(key, 0.0)) for key in ("preprocess", "inference", "postprocess")
    }
    return Path(result_path).stem, tuple(boxes), speed


def _to_list(value: object) -> list[object]:
    if value is None:
        return []
    cpu_method = getattr(value, "cpu", None)
    if callable(cpu_method):
        value = cpu_method()
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        converted = tolist_method()
        return list(converted) if isinstance(converted, list) else []
    return list(value) if isinstance(value, (list, tuple)) else []


def _numeric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationDataError(f"Prediction value is not numeric: {value!r}")
    return float(value)


def _reset_peak_gpu_memory() -> None:
    try:
        torch = importlib.import_module("torch")
        if bool(torch.cuda.is_available()):
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _peak_gpu_memory() -> int | None:
    try:
        torch = importlib.import_module("torch")
        if bool(torch.cuda.is_available()):
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None
    return None
