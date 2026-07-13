"""Processed YOLO ground-truth discovery for project-controlled evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from fod_yolo.dataset.discover import IMAGE_EXTENSIONS
from fod_yolo.evaluation import EvaluationDataError
from fod_yolo.evaluation.matcher import NormalizedBox
from fod_yolo.evaluation.threshold_sweep import ImageDetections


@dataclass(frozen=True, slots=True)
class EvaluationImage:
    image_id: str
    path: Path
    ground_truth: tuple[NormalizedBox, ...]


def load_evaluation_images(dataset_yaml: str | Path, split: str) -> tuple[EvaluationImage, ...]:
    """Load one processed split with its explicit class-0 YOLO labels."""

    yaml_path = Path(dataset_yaml).expanduser().resolve()
    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise EvaluationDataError(f"Unable to read dataset YAML {yaml_path}: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("path"), str):
        raise EvaluationDataError("Dataset YAML must contain a path")
    split_path = config.get(split)
    if not isinstance(split_path, str):
        raise EvaluationDataError(f"Dataset YAML does not define split {split!r}")
    root = Path(config["path"]).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()
    image_directory = (root / split_path).resolve()
    label_directory = root / "labels" / split
    images: list[EvaluationImage] = []
    for image_path in sorted(image_directory.iterdir()):
        if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_directory / f"{image_path.stem}.txt"
        images.append(
            EvaluationImage(
                image_id=image_path.stem,
                path=image_path,
                ground_truth=_read_label(label_path),
            )
        )
    if not images:
        raise EvaluationDataError(f"Evaluation split {split!r} contains no images")
    return tuple(images)


def combine_predictions(
    images: tuple[EvaluationImage, ...],
    predictions: dict[str, tuple[NormalizedBox, ...]],
) -> tuple[ImageDetections, ...]:
    """Require exactly one prediction entry for every expected image."""

    expected = {image.image_id for image in images}
    if set(predictions) != expected:
        missing = sorted(expected.difference(predictions))
        extra = sorted(set(predictions).difference(expected))
        raise EvaluationDataError(f"Prediction/image mismatch; missing={missing}, extra={extra}")
    return tuple(
        ImageDetections(image.image_id, image.ground_truth, predictions[image.image_id])
        for image in images
    )


def _read_label(path: Path) -> tuple[NormalizedBox, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvaluationDataError(f"Unable to read ground-truth label {path}: {exc}") from exc
    boxes: list[NormalizedBox] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise EvaluationDataError(f"Malformed ground truth {path}:{line_number}")
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise EvaluationDataError(f"Non-numeric ground truth {path}:{line_number}") from exc
        box = NormalizedBox(
            x_center - width / 2.0,
            y_center - height / 2.0,
            x_center + width / 2.0,
            y_center + height / 2.0,
            class_id,
        )
        box.validate()
        boxes.append(box)
    return tuple(boxes)
