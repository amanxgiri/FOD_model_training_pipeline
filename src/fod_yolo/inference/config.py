"""Typed configuration for reproducible streaming video inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from fod_yolo.config import ConfigValue, load_config
from fod_yolo.inference import InferenceConfigurationError
from fod_yolo.paths import ProjectPaths, resolve_path


@dataclass(frozen=True, slots=True)
class VideoInferenceSettings:
    """Validated runtime and artifact settings for one video."""

    config_source: Path
    model: Path
    source: Path | None
    imgsz: int
    confidence: float
    device: int | str
    frame_stride: int
    max_detections: int
    start_time_seconds: float | None
    end_time_seconds: float | None
    save_annotated_video: bool
    save_detection_csv: bool
    save_summary_json: bool
    save_detection_frames: bool
    stream: bool
    output_root: Path
    output_codec: str

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "device": self.device,
            "end_time_seconds": self.end_time_seconds,
            "frame_stride": self.frame_stride,
            "imgsz": self.imgsz,
            "max_detections": self.max_detections,
            "model": str(self.model),
            "output_codec": self.output_codec,
            "output_root": str(self.output_root),
            "save_annotated_video": self.save_annotated_video,
            "save_detection_csv": self.save_detection_csv,
            "save_detection_frames": self.save_detection_frames,
            "save_summary_json": self.save_summary_json,
            "source": str(self.source) if self.source is not None else None,
            "start_time_seconds": self.start_time_seconds,
            "stream": self.stream,
        }


def load_video_inference_settings(
    config_path: str | Path,
    project_paths: ProjectPaths,
    *,
    overrides: Iterable[str] = (),
) -> VideoInferenceSettings:
    """Load, resolve, and validate video inference configuration."""

    loaded = load_config(config_path, overrides=overrides)
    video = _mapping(loaded.values, "video")
    device = video.get("device")
    if isinstance(device, bool) or not isinstance(device, (int, str)):
        raise InferenceConfigurationError("video.device must be an integer or string")
    source_value = video.get("source")
    if source_value is not None and not isinstance(source_value, str):
        raise InferenceConfigurationError("video.source must be a path string or null")
    settings = VideoInferenceSettings(
        config_source=loaded.source,
        model=_resolve_model(_string(video, "model"), project_paths),
        source=(
            resolve_path(source_value, relative_to=project_paths.project_root)
            if source_value is not None and source_value.strip()
            else None
        ),
        imgsz=_integer(video, "imgsz"),
        confidence=_number(video, "confidence"),
        device=device,
        frame_stride=_integer(video, "frame_stride"),
        max_detections=_integer(video, "max_detections"),
        start_time_seconds=_optional_number(video, "start_time_seconds"),
        end_time_seconds=_optional_number(video, "end_time_seconds"),
        save_annotated_video=_boolean(video, "save_annotated_video"),
        save_detection_csv=_boolean(video, "save_detection_csv"),
        save_summary_json=_boolean(video, "save_summary_json"),
        save_detection_frames=_boolean(video, "save_detection_frames"),
        stream=_boolean(video, "stream"),
        output_root=project_paths.resolve_runs_path(_string(video, "output_root")),
        output_codec=_string(video, "output_codec"),
    )
    validate_video_inference_settings(settings)
    return settings


def validate_video_inference_settings(settings: VideoInferenceSettings) -> None:
    """Validate settings after configuration or explicit CLI replacement."""

    if settings.imgsz <= 0 or settings.frame_stride <= 0 or settings.max_detections <= 0:
        raise InferenceConfigurationError(
            "imgsz, frame_stride, and max_detections must be positive"
        )
    if not 0.0 <= settings.confidence <= 1.0:
        raise InferenceConfigurationError("video.confidence must be within [0, 1]")
    if settings.start_time_seconds is not None and settings.start_time_seconds < 0.0:
        raise InferenceConfigurationError("video.start_time_seconds cannot be negative")
    if settings.end_time_seconds is not None and settings.end_time_seconds <= 0.0:
        raise InferenceConfigurationError("video.end_time_seconds must be positive")
    if (
        settings.start_time_seconds is not None
        and settings.end_time_seconds is not None
        and settings.end_time_seconds <= settings.start_time_seconds
    ):
        raise InferenceConfigurationError("end time must be greater than start time")
    if len(settings.output_codec) != 4 or not settings.output_codec.isascii():
        raise InferenceConfigurationError("video.output_codec must contain four ASCII characters")
    if not settings.stream:
        raise InferenceConfigurationError("video.stream must remain true for bounded memory use")
    if not settings.save_summary_json:
        raise InferenceConfigurationError("video.save_summary_json must remain true")


def _resolve_model(value: str, paths: ProjectPaths) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0].casefold() == "runs":
        return paths.resolve_runs_path(candidate)
    if candidate.parts and candidate.parts[0].casefold() == "artifacts":
        return paths.resolve_artifacts_path(candidate)
    return resolve_path(candidate, relative_to=paths.project_root)


def _mapping(mapping: Mapping[str, ConfigValue], key: str) -> dict[str, ConfigValue]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise InferenceConfigurationError(f"Configuration key {key!r} must be a mapping")
    return value


def _string(mapping: Mapping[str, ConfigValue], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InferenceConfigurationError(f"video.{key} must be a non-empty string")
    return value.strip()


def _integer(mapping: Mapping[str, ConfigValue], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceConfigurationError(f"video.{key} must be an integer")
    return value


def _number(mapping: Mapping[str, ConfigValue], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceConfigurationError(f"video.{key} must be numeric")
    return float(value)


def _optional_number(mapping: Mapping[str, ConfigValue], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceConfigurationError(f"video.{key} must be numeric or null")
    return float(value)


def _boolean(mapping: Mapping[str, ConfigValue], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise InferenceConfigurationError(f"video.{key} must be boolean")
    return value
