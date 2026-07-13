"""Typed evaluation configuration for framework and project metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fod_yolo.config import ConfigValue, load_config
from fod_yolo.evaluation import EvaluationConfigurationError


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Validated evaluation, matching, sweep, and small-object settings."""

    source: Path
    imgsz: int
    device: int | str
    batch: int
    half: bool
    workers: int
    plots: bool
    save_json: bool
    save_txt: bool
    save_conf: bool
    max_det: int
    selection_split: str
    final_split: str
    iou_threshold: float
    class_agnostic: bool
    threshold_start: float
    threshold_stop: float
    threshold_step: float
    default_confidence: float
    small_area_ratio: float
    warmup_images: int


def load_evaluation_settings(config_path: str | Path) -> EvaluationSettings:
    """Load and validate the complete evaluation contract."""

    loaded = load_config(config_path)
    evaluation = _mapping(loaded.values, "evaluation")
    matching = _mapping(loaded.values, "matching")
    sweep = _mapping(loaded.values, "threshold_sweep")
    small = _mapping(loaded.values, "small_object")
    latency_value = loaded.values.get("latency", {"warmup_images": 1})
    if not isinstance(latency_value, dict):
        raise EvaluationConfigurationError("Configuration key 'latency' must be a mapping")
    device = evaluation.get("device")
    if isinstance(device, bool) or not isinstance(device, (int, str)):
        raise EvaluationConfigurationError("evaluation.device must be an integer or string")
    settings = EvaluationSettings(
        source=loaded.source,
        imgsz=_integer(evaluation, "imgsz"),
        device=device,
        batch=_integer(evaluation, "batch"),
        half=_boolean(evaluation, "half"),
        workers=_integer(evaluation, "workers"),
        plots=_boolean(evaluation, "plots"),
        save_json=_boolean(evaluation, "save_json"),
        save_txt=_boolean(evaluation, "save_txt"),
        save_conf=_boolean(evaluation, "save_conf"),
        max_det=_integer(evaluation, "max_det"),
        selection_split=_string(evaluation, "split_for_selection"),
        final_split=_string(evaluation, "final_split"),
        iou_threshold=_number(matching, "iou_threshold"),
        class_agnostic=_boolean(matching, "class_agnostic"),
        threshold_start=_number(sweep, "start"),
        threshold_stop=_number(sweep, "stop"),
        threshold_step=_number(sweep, "step"),
        default_confidence=_number(sweep, "include_default_confidence"),
        small_area_ratio=_number(small, "max_area_ratio"),
        warmup_images=_integer(latency_value, "warmup_images"),
    )
    _validate(settings, small)
    return settings


def _validate(settings: EvaluationSettings, small: Mapping[str, ConfigValue]) -> None:
    if settings.imgsz != 1280:
        raise EvaluationConfigurationError("Phase 1 evaluation requires evaluation.imgsz=1280")
    if settings.batch <= 0 or settings.workers < 0 or settings.max_det <= 0:
        raise EvaluationConfigurationError(
            "Batch/max_det must be positive and workers non-negative"
        )
    if settings.selection_split != "val" or settings.final_split != "test":
        raise EvaluationConfigurationError("Selection and final splits must be val and test")
    if not 0.0 < settings.iou_threshold <= 1.0:
        raise EvaluationConfigurationError("matching.iou_threshold must be within (0, 1]")
    if not settings.class_agnostic:
        raise EvaluationConfigurationError("Phase 1 single-class matching must be class agnostic")
    if not (
        0.0 <= settings.threshold_start <= settings.threshold_stop <= 1.0
        and settings.threshold_step > 0.0
        and 0.0 <= settings.default_confidence <= 1.0
    ):
        raise EvaluationConfigurationError("Invalid threshold sweep range")
    if _string(small, "definition") != "normalized_area_ratio":
        raise EvaluationConfigurationError("Small-object definition must be normalized_area_ratio")
    if not 0.0 < settings.small_area_ratio <= 1.0 or settings.warmup_images < 0:
        raise EvaluationConfigurationError("Invalid small-object or warm-up setting")


def _mapping(mapping: Mapping[str, ConfigValue], key: str) -> dict[str, ConfigValue]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise EvaluationConfigurationError(f"Configuration key {key!r} must be a mapping")
    return value


def _string(mapping: Mapping[str, ConfigValue], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationConfigurationError(f"Configuration key {key!r} must be a string")
    return value.strip()


def _integer(mapping: Mapping[str, ConfigValue], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationConfigurationError(f"Configuration key {key!r} must be an integer")
    return value


def _number(mapping: Mapping[str, ConfigValue], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationConfigurationError(f"Configuration key {key!r} must be numeric")
    return float(value)


def _boolean(mapping: Mapping[str, ConfigValue], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise EvaluationConfigurationError(f"Configuration key {key!r} must be boolean")
    return value
