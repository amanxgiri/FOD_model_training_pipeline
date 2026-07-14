"""Stable FOD detector adapter over lazy Ultralytics frame prediction."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from fod_yolo.inference import DetectorError


class PredictionModel(Protocol):
    def predict(self, **kwargs: object) -> Iterable[object]: ...


ModelFactory = Callable[[str], PredictionModel]


@dataclass(frozen=True, slots=True)
class Detection:
    """One prediction in original-frame pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class FramePrediction:
    """Detections and Ultralytics stage timings for one video frame."""

    detections: tuple[Detection, ...]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class FODDetector:
    """Load one checkpoint and expose stable frame-level detections."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise DetectorError(f"Model checkpoint does not exist: {path}")
        factory = create_model if model_factory is None else model_factory
        try:
            self._model = factory(str(path))
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"Unable to load model checkpoint {path}: {exc}") from exc

    def predict_frame(
        self,
        frame: Any,
        *,
        imgsz: int,
        confidence: float,
        device: int | str,
        max_detections: int,
    ) -> FramePrediction:
        """Predict one in-memory frame without retaining other video frames."""

        try:
            results = tuple(
                self._model.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=confidence,
                    device=device,
                    max_det=max_detections,
                    verbose=False,
                    stream=True,
                )
            )
        except Exception as exc:
            raise DetectorError(f"Frame inference failed: {exc}") from exc
        if len(results) != 1:
            raise DetectorError(f"Expected one frame result; received {len(results)}")
        return _parse_result(results[0])


def create_model(checkpoint: str) -> PredictionModel:
    """Import Ultralytics only when a real detector is constructed."""

    try:
        ultralytics = importlib.import_module("ultralytics")
        return cast(PredictionModel, ultralytics.YOLO(checkpoint))
    except (ImportError, AttributeError, OSError) as exc:
        raise DetectorError(f"Unable to load Ultralytics model: {exc}") from exc


def _parse_result(result: object) -> FramePrediction:
    boxes_object = getattr(result, "boxes", None)
    detections: list[Detection] = []
    if boxes_object is not None:
        coordinates = _to_list(getattr(boxes_object, "xyxy", None))
        confidences = _to_list(getattr(boxes_object, "conf", None))
        classes = _to_list(getattr(boxes_object, "cls", None))
        if not (len(coordinates) == len(confidences) == len(classes)):
            raise DetectorError("Prediction box arrays have inconsistent lengths")
        names = _class_names(getattr(result, "names", {}))
        for coordinate, raw_confidence, raw_class in zip(
            coordinates,
            confidences,
            classes,
            strict=True,
        ):
            if not isinstance(coordinate, list) or len(coordinate) != 4:
                raise DetectorError("Prediction xyxy row must contain four values")
            class_id = int(_numeric(raw_class))
            detection = Detection(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=_numeric(raw_confidence),
                x1=_numeric(coordinate[0]),
                y1=_numeric(coordinate[1]),
                x2=_numeric(coordinate[2]),
                y2=_numeric(coordinate[3]),
            )
            _validate_detection(detection)
            detections.append(detection)
    speed = getattr(result, "speed", None)
    if not isinstance(speed, Mapping):
        raise DetectorError("Prediction result has no timing mapping")
    timings = tuple(_timing(speed, key) for key in ("preprocess", "inference", "postprocess"))
    return FramePrediction(tuple(detections), *timings)


def _class_names(value: object) -> dict[int, str]:
    if isinstance(value, Mapping):
        return {
            int(key): str(name)
            for key, name in value.items()
            if isinstance(key, (int, str)) and str(key).isdigit()
        }
    if isinstance(value, (list, tuple)):
        return {index: str(name) for index, name in enumerate(value)}
    return {}


def _validate_detection(detection: Detection) -> None:
    values = (
        detection.confidence,
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
    )
    if not all(math.isfinite(value) for value in values):
        raise DetectorError("Prediction contains non-finite values")
    if not 0.0 <= detection.confidence <= 1.0:
        raise DetectorError("Prediction confidence falls outside [0, 1]")
    if detection.x2 <= detection.x1 or detection.y2 <= detection.y1:
        raise DetectorError("Prediction bounding box is degenerate")


def _timing(speed: Mapping[object, object], key: str) -> float:
    value = _numeric(speed.get(key, 0.0))
    if value < 0.0:
        raise DetectorError("Prediction timing cannot be negative")
    return value


def _numeric(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DetectorError(f"Prediction value is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise DetectorError("Prediction value is not finite")
    return result


def _to_list(value: object) -> list[object]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(value) if isinstance(value, (list, tuple)) else [value]
