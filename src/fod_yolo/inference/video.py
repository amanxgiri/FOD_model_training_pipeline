"""Memory-efficient project-controlled annotated video inference."""

from __future__ import annotations

import csv
import importlib
import logging
import math
import re
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fod_yolo.hashing import atomic_write_json, atomic_write_yaml, sha256_file
from fod_yolo.inference import VideoInferenceError
from fod_yolo.inference.config import VideoInferenceSettings
from fod_yolo.inference.detector import Detection, FODDetector, FramePrediction

LOGGER = logging.getLogger("fod_yolo")

DETECTION_COLUMNS = (
    "video_name",
    "frame_index",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "x2",
    "y1",
    "y2",
    "box_width_pixels",
    "box_height_pixels",
    "box_area_pixels",
    "box_area_ratio",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
)
FRAME_COLUMNS = (
    "frame_index",
    "timestamp_seconds",
    "detections",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "end_to_end_ms",
)


class Detector(Protocol):
    def predict_frame(
        self,
        frame: Any,
        *,
        imgsz: int,
        confidence: float,
        device: int | str,
        max_detections: int,
    ) -> FramePrediction: ...


DetectorFactory = Callable[[Path], Detector]


@dataclass(frozen=True, slots=True)
class VideoInferenceResult:
    """Paths and final status for one video-inference run."""

    run_directory: Path
    summary_path: Path
    summary: dict[str, object]


def run_video_inference(
    settings: VideoInferenceSettings,
    *,
    detector_factory: DetectorFactory = FODDetector,
    cv2_module: Any | None = None,
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.perf_counter,
) -> VideoInferenceResult:
    """Stream a video through one detector and retain complete or partial artifacts."""

    if settings.source is None:
        raise VideoInferenceError("A source video is required")
    source = settings.source.expanduser().resolve()
    model = settings.model.expanduser().resolve()
    if not source.is_file():
        raise VideoInferenceError(f"Source video does not exist: {source}")
    if not model.is_file():
        raise VideoInferenceError(f"Model checkpoint does not exist: {model}")

    cv2 = _load_cv2() if cv2_module is None else cv2_module
    model_sha256 = sha256_file(model)
    timestamp = (now or (lambda: datetime.now(UTC)))()
    run_id = _run_id(source.stem, timestamp, model_sha256)
    run_directory = settings.output_root / run_id
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise VideoInferenceError(f"Unable to create inference run {run_directory}: {exc}") from exc

    summary_path = run_directory / "video_summary.json"
    annotated_path = run_directory / "annotated_video.mp4"
    detections_path = run_directory / "detections.csv"
    frame_metrics_path = run_directory / "frame_metrics.csv"
    detection_frames = run_directory / "detection_frames"
    atomic_write_yaml(run_directory / "inference_config.yaml", settings.to_dict())

    capture: Any | None = None
    writer: Any | None = None
    detection_handle: Any | None = None
    frame_handle: Any | None = None
    status = "failed"
    error: str | None = None
    failure: Exception | None = None
    source_metadata: dict[str, object] = {"path": str(source), "video_name": source.name}
    processed_frames = 0
    detection_frame_count = 0
    total_detections = 0
    confidences: list[float] = []
    inference_samples: list[float] = []
    processing_started = monotonic()

    try:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoInferenceError(f"OpenCV could not open source video: {source}")
        source_metadata = _source_metadata(capture, source, cv2)
        atomic_write_json(run_directory / "source_metadata.json", source_metadata)
        width = int(_metadata_number(source_metadata, "width"))
        height = int(_metadata_number(source_metadata, "height"))
        source_fps = _metadata_number(source_metadata, "fps")
        LOGGER.info(
            "Opened %s: %dx%d at %.3f FPS, frames=%s",
            source.name,
            width,
            height,
            source_fps,
            source_metadata["frame_count"],
        )
        start_frame = int((settings.start_time_seconds or 0.0) * source_fps)
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if settings.save_annotated_video:
            output_fps = source_fps / settings.frame_stride
            codec = cv2.VideoWriter_fourcc(*settings.output_codec)
            writer = cv2.VideoWriter(
                str(annotated_path),
                codec,
                output_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise VideoInferenceError(
                    f"OpenCV could not create annotated video with codec "
                    f"{settings.output_codec}: {annotated_path}"
                )
        if settings.save_detection_csv:
            detection_handle = detections_path.open("w", encoding="utf-8", newline="")
            detection_writer = csv.DictWriter(detection_handle, fieldnames=DETECTION_COLUMNS)
            detection_writer.writeheader()
        else:
            detection_writer = None
        frame_handle = frame_metrics_path.open("w", encoding="utf-8", newline="")
        frame_writer = csv.DictWriter(frame_handle, fieldnames=FRAME_COLUMNS)
        frame_writer.writeheader()
        if settings.save_detection_frames:
            detection_frames.mkdir()

        detector = detector_factory(model)
        processing_started = monotonic()
        frame_index = start_frame
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            timestamp_seconds = frame_index / source_fps
            if (
                settings.end_time_seconds is not None
                and timestamp_seconds > settings.end_time_seconds
            ):
                break
            if (frame_index - start_frame) % settings.frame_stride != 0:
                frame_index += 1
                continue

            frame_started = monotonic()
            prediction = detector.predict_frame(
                frame,
                imgsz=settings.imgsz,
                confidence=settings.confidence,
                device=settings.device,
                max_detections=settings.max_detections,
            )
            end_to_end_ms = (monotonic() - frame_started) * 1000.0
            processed_frames += 1
            inference_samples.append(prediction.inference_ms)
            if prediction.detections:
                detection_frame_count += 1
            annotated = frame.copy()
            for detection in prediction.detections:
                total_detections += 1
                confidences.append(detection.confidence)
                clipped = _clip_detection(detection, width, height)
                _draw_detection(annotated, clipped, cv2)
                if detection_writer is not None:
                    detection_writer.writerow(
                        _detection_row(
                            source.name,
                            frame_index,
                            timestamp_seconds,
                            clipped,
                            width,
                            height,
                            prediction,
                        )
                    )
            frame_writer.writerow(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": _decimal(timestamp_seconds),
                    "detections": len(prediction.detections),
                    "preprocess_ms": _decimal(prediction.preprocess_ms),
                    "inference_ms": _decimal(prediction.inference_ms),
                    "postprocess_ms": _decimal(prediction.postprocess_ms),
                    "end_to_end_ms": _decimal(end_to_end_ms),
                }
            )
            if writer is not None:
                writer.write(annotated)
            if settings.save_detection_frames and prediction.detections:
                frame_path = detection_frames / f"frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(frame_path), annotated):
                    raise VideoInferenceError(f"Unable to write detection frame: {frame_path}")
            if processed_frames % 100 == 0:
                if detection_handle is not None:
                    detection_handle.flush()
                frame_handle.flush()
                LOGGER.info(
                    "Video inference progress: frames=%d detections=%d",
                    processed_frames,
                    total_detections,
                )
            frame_index += 1
        status = "complete"
    except KeyboardInterrupt:
        status = "incomplete"
        error = "Interrupted by user"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        failure = exc
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if detection_handle is not None:
            detection_handle.close()
        if frame_handle is not None:
            frame_handle.close()

    elapsed_seconds = max(0.0, monotonic() - processing_started)
    summary = _build_summary(
        status=status,
        error=error,
        source_metadata=source_metadata,
        settings=settings,
        model_sha256=model_sha256,
        frames_processed=processed_frames,
        frames_with_detections=detection_frame_count,
        total_detections=total_detections,
        confidences=confidences,
        inference_samples=inference_samples,
        elapsed_seconds=elapsed_seconds,
        run_directory=run_directory,
    )
    atomic_write_json(summary_path, summary)
    if failure is not None:
        raise VideoInferenceError(
            f"Video inference failed; partial summary retained at {summary_path}: {failure}"
        ) from failure
    return VideoInferenceResult(run_directory, summary_path, summary)


def _load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise VideoInferenceError("OpenCV is required for video inference") from exc


def _source_metadata(capture: Any, source: Path, cv2: Any) -> dict[str, object]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0.0:
        raise VideoInferenceError("Source video has invalid dimensions or FPS metadata")
    codec = "".join(chr((codec_value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")
    return {
        "codec": codec or None,
        "duration_seconds": frame_count / fps if frame_count > 0 else None,
        "fps": fps,
        "frame_count": max(frame_count, 0),
        "height": height,
        "path": str(source),
        "video_name": source.name,
        "width": width,
    }


def _metadata_number(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoInferenceError(f"Source metadata {key!r} is not numeric")
    return float(value)


def _clip_detection(detection: Detection, width: int, height: int) -> Detection:
    x1 = min(max(detection.x1, 0.0), float(width - 1))
    y1 = min(max(detection.y1, 0.0), float(height - 1))
    x2 = min(max(detection.x2, x1 + 1.0), float(width))
    y2 = min(max(detection.y2, y1 + 1.0), float(height))
    return Detection(
        detection.class_id,
        detection.class_name,
        detection.confidence,
        x1,
        y1,
        x2,
        y2,
    )


def _draw_detection(frame: Any, detection: Detection, cv2: Any) -> None:
    top_left = (round(detection.x1), round(detection.y1))
    bottom_right = (round(detection.x2), round(detection.y2))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    label = f"{detection.class_name} {detection.confidence:.2f}"
    text_y = max(18, top_left[1] - 6)
    cv2.putText(
        frame,
        label,
        (top_left[0], text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _detection_row(
    video_name: str,
    frame_index: int,
    timestamp_seconds: float,
    detection: Detection,
    width: int,
    height: int,
    prediction: FramePrediction,
) -> dict[str, object]:
    return {
        "video_name": video_name,
        "frame_index": frame_index,
        "timestamp_seconds": _decimal(timestamp_seconds),
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": _decimal(detection.confidence),
        "x1": _decimal(detection.x1),
        "x2": _decimal(detection.x2),
        "y1": _decimal(detection.y1),
        "y2": _decimal(detection.y2),
        "box_width_pixels": _decimal(detection.width),
        "box_height_pixels": _decimal(detection.height),
        "box_area_pixels": _decimal(detection.area),
        "box_area_ratio": _decimal(detection.area / (width * height)),
        "preprocess_ms": _decimal(prediction.preprocess_ms),
        "inference_ms": _decimal(prediction.inference_ms),
        "postprocess_ms": _decimal(prediction.postprocess_ms),
    }


def _build_summary(
    *,
    status: str,
    error: str | None,
    source_metadata: Mapping[str, object],
    settings: VideoInferenceSettings,
    model_sha256: str,
    frames_processed: int,
    frames_with_detections: int,
    total_detections: int,
    confidences: list[float],
    inference_samples: list[float],
    elapsed_seconds: float,
    run_directory: Path,
) -> dict[str, object]:
    source_fps_value = source_metadata.get("fps")
    source_fps = float(source_fps_value) if isinstance(source_fps_value, (int, float)) else 0.0
    processing_fps = frames_processed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    return {
        "accuracy_metrics_available": False,
        "completion_status": status,
        "confidence": {
            "maximum": max(confidences) if confidences else None,
            "mean": statistics.fmean(confidences) if confidences else None,
            "median": statistics.median(confidences) if confidences else None,
            "threshold": settings.confidence,
        },
        "error": error,
        "evaluation_type": "inference-only",
        "frame_stride": settings.frame_stride,
        "frames_processed": frames_processed,
        "frames_with_detections": frames_with_detections,
        "imgsz": settings.imgsz,
        "latency_ms": {
            "mean_inference": (statistics.fmean(inference_samples) if inference_samples else None),
            "p95_inference": _percentile(inference_samples, 0.95),
        },
        "limitation": (
            "Unlabelled video cannot provide precision, recall, mAP, or false-negative rate."
        ),
        "model": {"path": str(settings.model), "sha256": model_sha256},
        "output_directory": str(run_directory),
        "processing": {
            "elapsed_seconds": elapsed_seconds,
            "fps": processing_fps,
            "real_time_ratio": processing_fps / source_fps if source_fps > 0.0 else None,
        },
        "source": dict(source_metadata),
        "source_fps": source_fps or None,
        "total_detections": total_detections,
        "total_frames_in_source": source_metadata.get("frame_count"),
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _decimal(value: float) -> str:
    return f"{value:.6f}"


def _run_id(source_stem: str, timestamp: datetime, model_sha256: str) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_stem).strip("_") or "video"
    utc = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_stem}_{utc}_{model_sha256[:8]}"
