"""Strict validation for processed single-class YOLO datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

from fod_yolo.dataset import DatasetValidationError
from fod_yolo.dataset.discover import IMAGE_EXTENSIONS
from fod_yolo.hashing import sha256_file


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Stable processed-dataset validation result."""

    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, int]
    class_ids_observed: tuple[int, ...]
    class_names: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Return the stable validation-report schema."""

        return {
            "class_ids_observed": list(self.class_ids_observed),
            "class_names": self.class_names,
            "counts": self.counts,
            "errors": list(self.errors),
            "schema_version": "1.0",
            "status": self.status,
            "warnings": list(self.warnings),
        }


class StrictDatasetValidationError(DatasetValidationError):
    """Strict-mode failure that retains the complete validation report."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(f"Processed dataset validation failed with {len(report.errors)} error(s)")


def validate_yolo_dataset(
    dataset_yaml: str | Path,
    *,
    strict: bool = False,
    check_duplicate_hashes: bool = False,
) -> ValidationReport:
    """Validate class mapping, image/label pairs, coordinates, splits, and manifest."""

    yaml_path = Path(dataset_yaml).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    counts = {
        "test_images": 0,
        "test_objects": 0,
        "train_images": 0,
        "train_objects": 0,
        "val_images": 0,
        "val_objects": 0,
    }
    class_ids: set[int] = set()
    class_names: dict[str, str] = {}

    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        errors.append(f"Unable to parse dataset YAML {yaml_path}: {exc}")
        return _finish(errors, warnings, counts, class_ids, class_names, strict)
    if not isinstance(config, dict):
        errors.append("Dataset YAML root must be a mapping")
        return _finish(errors, warnings, counts, class_ids, class_names, strict)

    raw_names = config.get("names")
    if isinstance(raw_names, dict):
        class_names = {str(key): str(value) for key, value in raw_names.items()}
    if class_names != {"0": "FOD"}:
        errors.append("Dataset YAML must define exactly one class: 0: FOD")

    raw_root = config.get("path")
    if not isinstance(raw_root, str) or not raw_root.strip():
        errors.append("Dataset YAML must define a non-empty path")
        return _finish(errors, warnings, counts, class_ids, class_names, strict)
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()

    split_ids: dict[str, set[str]] = {}
    image_hashes: dict[str, tuple[str, str]] = {}
    for split_name in ("train", "val", "test"):
        split_value = config.get(split_name)
        if not isinstance(split_value, str):
            errors.append(f"Dataset YAML is missing string path for split {split_name}")
            split_ids[split_name] = set()
            continue
        image_directory = (root / split_value).resolve()
        label_directory = root / "labels" / split_name
        images = _indexed_files(image_directory, IMAGE_EXTENSIONS, errors, "image")
        labels = _indexed_files(label_directory, (".txt",), errors, "label")
        split_ids[split_name] = set(images)
        counts[f"{split_name}_images"] = len(images)

        for image_id, image_path in images.items():
            _validate_image(image_path, errors)
            label_path = labels.get(image_id)
            if label_path is None:
                errors.append(f"Missing label for {split_name} image {image_path}")
                continue
            object_count = _validate_label(label_path, class_ids, errors)
            counts[f"{split_name}_objects"] += object_count

            if check_duplicate_hashes:
                digest = sha256_file(image_path)
                previous = image_hashes.get(digest)
                if previous is not None and previous[0] != split_name:
                    errors.append(
                        f"Duplicate image content across splits: {previous[0]}/{previous[1]} "
                        f"and {split_name}/{image_id}"
                    )
                else:
                    image_hashes[digest] = (split_name, image_id)

        for image_id, label_path in labels.items():
            if image_id not in images:
                errors.append(f"Label has no corresponding {split_name} image: {label_path}")

    _validate_disjoint_splits(split_ids, errors)
    _validate_manifest(root / "dataset_manifest.json", split_ids, counts, errors, warnings)
    if class_ids and class_ids != {0}:
        errors.append(f"Observed non-zero class IDs: {sorted(class_ids)}")
    return _finish(errors, warnings, counts, class_ids, class_names, strict)


def _indexed_files(
    directory: Path,
    extensions: tuple[str, ...],
    errors: list[str],
    kind: str,
) -> dict[str, Path]:
    if not directory.is_dir():
        errors.append(f"Missing {kind} directory: {directory}")
        return {}
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in extensions:
            continue
        if path.stem in indexed:
            errors.append(f"Duplicate {kind} stem {path.stem!r} in {directory}")
        else:
            indexed[path.stem] = path
    return indexed


def _validate_image(path: Path, errors: list[str]) -> None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        errors.append(f"Unreadable image {path}: {exc}")
        return
    if width <= 0 or height <= 0:
        errors.append(f"Image dimensions are not positive for {path}")


def _validate_label(path: Path, class_ids: set[int], errors: list[str]) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        errors.append(f"Unreadable label {path}: {exc}")
        return 0

    object_count = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"Label {path}:{line_number} must contain exactly five fields")
            continue
        try:
            class_value = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"Label {path}:{line_number} contains a non-numeric value")
            continue
        class_ids.add(class_value)
        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"Label {path}:{line_number} contains a non-finite coordinate")
            continue
        if class_value != 0:
            errors.append(f"Label {path}:{line_number} uses class ID {class_value}, expected 0")
        if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
            errors.append(f"Label {path}:{line_number} center is outside [0, 1]")
        if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
            errors.append(f"Label {path}:{line_number} size is outside (0, 1]")
        if (
            x_center - width / 2.0 < -1e-6
            or x_center + width / 2.0 > 1.0 + 1e-6
            or y_center - height / 2.0 < -1e-6
            or y_center + height / 2.0 > 1.0 + 1e-6
        ):
            errors.append(f"Label {path}:{line_number} box extends outside the image")
        object_count += 1
    return object_count


def _validate_disjoint_splits(split_ids: dict[str, set[str]], errors: list[str]) -> None:
    split_names = tuple(split_ids)
    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1 :]:
            overlap = sorted(split_ids[first_name].intersection(split_ids[second_name]))
            if overlap:
                errors.append(
                    f"Split overlap between {first_name} and {second_name}: {', '.join(overlap)}"
                )


def _validate_manifest(
    manifest_path: Path,
    split_ids: dict[str, set[str]],
    counts: dict[str, int],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not manifest_path.is_file():
        errors.append(f"Dataset manifest is missing: {manifest_path}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"Dataset manifest is unreadable: {exc}")
        return
    manifest_splits = manifest.get("splits")
    if not isinstance(manifest_splits, dict):
        errors.append("Dataset manifest is missing split membership")
        return
    manifest_counts = manifest.get("counts")
    if not isinstance(manifest_counts, dict):
        errors.append("Dataset manifest is missing count metadata")
        return
    for split_name, observed_ids in split_ids.items():
        expected = manifest_splits.get(split_name)
        if not isinstance(expected, list) or {str(value) for value in expected} != observed_ids:
            errors.append(f"Dataset manifest IDs do not match the {split_name} split")
        expected_count = manifest_counts.get(f"{split_name}_images")
        if expected_count != counts[f"{split_name}_images"]:
            errors.append(f"Dataset manifest image count does not match {split_name}")
    if manifest.get("dataset_fingerprint") is None:
        warnings.append("Dataset manifest does not contain a dataset fingerprint")


def _finish(
    errors: list[str],
    warnings: list[str],
    counts: dict[str, int],
    class_ids: set[int],
    class_names: dict[str, str],
    strict: bool,
) -> ValidationReport:
    report = ValidationReport(
        status="pass" if not errors else "fail",
        errors=tuple(errors),
        warnings=tuple(warnings),
        counts=counts,
        class_ids_observed=tuple(sorted(class_ids)),
        class_names=class_names,
    )
    if strict and report.errors:
        raise StrictDatasetValidationError(report)
    return report
