"""Atomic end-to-end validation/test evaluation orchestration."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fod_yolo.dataset.validate import validate_yolo_dataset
from fod_yolo.environment import collect_environment_report
from fod_yolo.evaluation import EvaluationConfigurationError, EvaluationExecutionError
from fod_yolo.evaluation.config import EvaluationSettings
from fod_yolo.evaluation.data import combine_predictions, load_evaluation_images
from fod_yolo.evaluation.threshold_sweep import (
    ThresholdMetrics,
    build_thresholds,
    evaluate_threshold,
    select_threshold_references,
)
from fod_yolo.evaluation.ultralytics_eval import (
    ModelFactory,
    create_model,
    run_framework_evaluation,
)
from fod_yolo.hashing import (
    atomic_replace_path,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    sha256_file,
)
from fod_yolo.paths import ProjectPaths
from fod_yolo.training.trainer import read_dataset_fingerprint


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluation_directory: Path
    metrics_path: Path
    threshold_csv_path: Path
    metrics: dict[str, object]


def run_evaluation(
    *,
    model_path: str | Path,
    dataset_yaml: str | Path,
    split: str,
    settings: EvaluationSettings,
    project_paths: ProjectPaths,
    locked_threshold: float | None = None,
    model_factory: ModelFactory = create_model,
) -> EvaluationResult:
    """Run framework and project evaluation without using test data for selection."""

    model = Path(model_path).expanduser().resolve()
    data = Path(dataset_yaml).expanduser().resolve()
    if not model.is_file() or model.stat().st_size == 0:
        raise EvaluationConfigurationError(f"Model checkpoint is missing or empty: {model}")
    if split not in {settings.selection_split, settings.final_split}:
        raise EvaluationConfigurationError("Evaluation split must be val or test")
    if split == settings.final_split and locked_threshold is None:
        raise EvaluationConfigurationError("Test evaluation requires --locked-threshold")
    if locked_threshold is not None and not 0.0 <= locked_threshold <= 1.0:
        raise EvaluationConfigurationError("Locked threshold must be within [0, 1]")

    validate_yolo_dataset(data, strict=True)
    dataset_fingerprint = read_dataset_fingerprint(data)
    images = load_evaluation_images(data, split)
    run_id = _derive_run_id(model)
    target = project_paths.reports_root / run_id
    if split == settings.final_split:
        target = target / "test"
    if target.exists():
        raise EvaluationExecutionError(f"Evaluation directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        framework = run_framework_evaluation(
            model_factory(str(model)),
            dataset_yaml=data,
            split=split,
            images=images,
            settings=settings,
            framework_output=staging / "ultralytics",
        )
        detections = combine_predictions(images, framework.predictions)
        if split == settings.selection_split:
            thresholds = build_thresholds(
                settings.threshold_start,
                settings.threshold_stop,
                settings.threshold_step,
                settings.default_confidence,
            )
            if locked_threshold is not None and locked_threshold not in thresholds:
                thresholds = tuple(sorted((*thresholds, locked_threshold)))
            selected_threshold = (
                locked_threshold if locked_threshold is not None else settings.default_confidence
            )
            threshold_selection_allowed = True
        else:
            assert locked_threshold is not None
            thresholds = (locked_threshold,)
            selected_threshold = locked_threshold
            threshold_selection_allowed = False
        rows = tuple(
            evaluate_threshold(
                detections,
                confidence_threshold=threshold,
                iou_threshold=settings.iou_threshold,
                small_area_threshold=settings.small_area_ratio,
            )
            for threshold in thresholds
        )
        selected = _selected_row(rows, selected_threshold)
        references = (
            select_threshold_references(rows)
            if threshold_selection_allowed
            else {
                "balanced_high_recall_threshold": selected_threshold,
                "best_f1_threshold": selected_threshold,
                "max_recall_threshold": selected_threshold,
            }
        )
        environment = collect_environment_report(project_root=project_paths.project_root)
        metrics = _metrics_document(
            run_id=run_id,
            model=model,
            dataset_fingerprint=dataset_fingerprint,
            split=split,
            settings=settings,
            selected=selected,
            references=references,
            standard=framework.standard_metrics.to_dict(),
            latency=framework.latency.to_dict(),
            peak_gpu_memory_bytes=framework.peak_gpu_memory_bytes,
            threshold_selection_allowed=threshold_selection_allowed,
        )
        atomic_write_json(staging / "metrics.json", metrics)
        atomic_write_json(staging / "threshold_sweep.json", [row.to_dict() for row in rows])
        atomic_write_text(staging / "threshold_sweep.csv", _threshold_csv(rows))
        atomic_write_json(staging / "environment.json", environment.to_dict())
        atomic_write_yaml(
            staging / "resolved_evaluation_config.yaml",
            {
                "dataset_yaml": str(data),
                "locked_threshold": locked_threshold,
                "model": str(model),
                "settings_source": str(settings.source),
                "split": split,
            },
        )
        atomic_replace_path(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return EvaluationResult(
        evaluation_directory=target,
        metrics_path=target / "metrics.json",
        threshold_csv_path=target / "threshold_sweep.csv",
        metrics=metrics,
    )


def _derive_run_id(model: Path) -> str:
    if model.parent.name == "weights":
        return model.parent.parent.name
    return model.parent.name or model.stem


def _selected_row(rows: tuple[ThresholdMetrics, ...], threshold: float) -> ThresholdMetrics:
    try:
        return next(row for row in rows if abs(row.threshold - threshold) <= 1e-12)
    except StopIteration as exc:
        raise EvaluationConfigurationError(
            f"Selected threshold {threshold} is not present in the threshold sweep"
        ) from exc


def _metrics_document(
    *,
    run_id: str,
    model: Path,
    dataset_fingerprint: str,
    split: str,
    settings: EvaluationSettings,
    selected: ThresholdMetrics,
    references: dict[str, float],
    standard: dict[str, float],
    latency: dict[str, float],
    peak_gpu_memory_bytes: int | None,
    threshold_selection_allowed: bool,
) -> dict[str, object]:
    return {
        "confidence_threshold": selected.threshold,
        "counts": {
            "false_negatives": selected.false_negatives,
            "false_positives": selected.false_positives,
            "ground_truth": selected.ground_truth,
            "images": selected.images,
            "predictions": selected.predictions,
            "true_positives": selected.true_positives,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "imgsz": settings.imgsz,
        "latency": {**latency, "peak_gpu_memory_bytes": peak_gpu_memory_bytes},
        "matching_iou_threshold": settings.iou_threshold,
        "metrics": {
            "f1": selected.f1,
            "false_negative_rate": selected.false_negative_rate,
            "false_positives_per_image": selected.false_positives_per_image,
            "fraction_of_images_with_false_negatives": (
                selected.fraction_of_images_with_false_negatives
            ),
            "images_with_false_negatives": selected.images_with_false_negatives,
            "map50": standard["map50"],
            "map50_95": standard["map50_95"],
            "map75": standard["map75"],
            "precision": selected.precision,
            "recall": selected.recall,
            "small_object_false_negative_rate": selected.small_object_false_negative_rate,
            "small_object_ground_truth_count": selected.small_object_ground_truth_count,
            "small_object_recall": selected.small_object_recall,
            "small_object_true_positives": selected.small_object_true_positives,
            "small_object_false_negatives": selected.small_object_false_negatives,
            "ultralytics_precision": standard["precision"],
            "ultralytics_recall": standard["recall"],
        },
        "model_file_size_bytes": model.stat().st_size,
        "model_sha256": sha256_file(model),
        "run_id": run_id,
        "runtime": {
            "batch": settings.batch,
            "device": settings.device,
            "precision": "fp16" if settings.half else "fp32",
            "warmup_images": settings.warmup_images,
        },
        "schema_version": "1.0",
        "small_object_max_area_ratio": settings.small_area_ratio,
        "split": split,
        "threshold_references": references,
        "threshold_selection_allowed": threshold_selection_allowed,
    }


def _threshold_csv(rows: tuple[ThresholdMetrics, ...]) -> str:
    dictionaries = [row.to_dict() for row in rows]
    headers = list(dictionaries[0])
    lines = [",".join(headers)]
    lines.extend(",".join(str(row[header]) for header in headers) for row in dictionaries)
    return "\n".join(lines) + "\n"
