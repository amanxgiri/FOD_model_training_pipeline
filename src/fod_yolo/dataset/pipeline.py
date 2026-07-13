"""High-level dataset download, preparation, manifest, and validation workflows."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fod_yolo.config import ConfigMapping, ConfigValue, load_config
from fod_yolo.dataset import DatasetConversionError
from fod_yolo.dataset.convert import ConversionOptions, ConvertedImage, convert_voc_dataset
from fod_yolo.dataset.discover import discover_voc_root
from fod_yolo.dataset.kaggle_client import DownloadResult, download_and_extract_dataset
from fod_yolo.dataset.split import DatasetSplits, resolve_dataset_splits, write_split_files
from fod_yolo.dataset.statistics import build_dataset_statistics
from fod_yolo.dataset.validate import ValidationReport, validate_yolo_dataset
from fod_yolo.environment import inspect_git
from fod_yolo.hashing import (
    atomic_replace_path,
    atomic_write_json,
    atomic_write_yaml,
    sha256_file,
    sha256_json,
)
from fod_yolo.paths import ProjectPaths


@dataclass(frozen=True, slots=True)
class SplitOptions:
    """Deterministic split settings from dataset.yaml."""

    seed: int
    validation_fraction: float
    preserve_official_test: bool
    trainval_file: Path
    test_file: Path

    def to_dict(self) -> dict[str, object]:
        """Return stable manifest metadata."""

        return {
            "preserve_official_test": self.preserve_official_test,
            "seed": self.seed,
            "source_test_file": self.test_file.as_posix(),
            "source_trainval_file": self.trainval_file.as_posix(),
            "validation_fraction_of_trainval": self.validation_fraction,
        }


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Fully resolved dataset pipeline settings for the current machine."""

    kaggle_slug: str
    kaggle_version: str | None
    raw_root: Path
    processed_root: Path
    archive_name: str
    force_download: bool
    force_extract: bool
    conversion: ConversionOptions
    split: SplitOptions


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Outputs from an idempotent processed-dataset build."""

    processed_root: Path
    dataset_yaml: Path
    manifest_path: Path
    statistics_path: Path
    validation_report_path: Path
    validation_report: ValidationReport
    records: tuple[ConvertedImage, ...]
    rebuilt: bool


def load_dataset_settings(
    config_path: str | Path,
    project_paths: ProjectPaths,
    *,
    overrides: Iterable[str] = (),
) -> DatasetSettings:
    """Load and type-check the dataset-specific YAML contract."""

    config = load_config(config_path, overrides=overrides).values
    dataset = _mapping(config, "dataset")
    conversion = _mapping(config, "conversion")
    split = _mapping(config, "split")

    raw_root = project_paths.resolve_data_path(_string(dataset, "raw_root"))
    processed_root = project_paths.resolve_data_path(_string(dataset, "processed_root"))
    archive_name = Path(_string(dataset, "archive_name")).name
    if archive_name != _string(dataset, "archive_name") or not archive_name.lower().endswith(
        ".zip"
    ):
        raise DatasetConversionError("dataset.archive_name must be a simple .zip filename")

    version_value = dataset.get("kaggle_version")
    version = None if version_value is None else str(version_value)
    settings = DatasetSettings(
        kaggle_slug=_string(dataset, "kaggle_slug"),
        kaggle_version=version,
        raw_root=raw_root,
        processed_root=processed_root,
        archive_name=archive_name,
        force_download=_boolean(dataset, "force_download"),
        force_extract=_boolean(dataset, "force_extract"),
        conversion=ConversionOptions(
            target_class_id=_integer(conversion, "target_class_id"),
            target_class_name=_string(conversion, "target_class_name"),
            image_transfer_mode=_string(conversion, "image_transfer_mode"),
            keep_empty_images=_boolean(conversion, "keep_empty_images"),
            clip_boxes=_boolean(conversion, "clip_boxes"),
            reject_degenerate_boxes=_boolean(conversion, "reject_degenerate_boxes"),
            verify_image_dimensions=_boolean(conversion, "verify_image_dimensions"),
        ),
        split=SplitOptions(
            seed=_integer(split, "seed"),
            validation_fraction=_number(split, "validation_fraction_of_trainval"),
            preserve_official_test=_boolean(split, "preserve_official_test"),
            trainval_file=_relative_path(split, "source_trainval_file"),
            test_file=_relative_path(split, "source_test_file"),
        ),
    )
    settings.conversion.validate()
    return settings


def download_dataset(
    settings: DatasetSettings,
    *,
    force: bool = False,
    environment: Mapping[str, str] | None = None,
) -> DownloadResult:
    """Run the configured Kaggle download and safe extraction workflow."""

    return download_and_extract_dataset(
        dataset_slug=settings.kaggle_slug,
        raw_root=settings.raw_root,
        archive_name=settings.archive_name,
        version=settings.kaggle_version,
        force_download=force or settings.force_download,
        force_extract=force or settings.force_extract,
        environment=environment,
    )


def prepare_dataset(
    settings: DatasetSettings,
    *,
    force: bool = False,
) -> PreparationResult:
    """Discover, split, convert, validate, and atomically install a YOLO dataset."""

    final_root = settings.processed_root
    dataset_yaml = final_root / "fod_a.yaml"
    manifest_path = final_root / "dataset_manifest.json"
    statistics_path = final_root / "dataset_statistics.json"
    validation_path = final_root / "validation_report.json"
    if final_root.exists() and not force:
        report = validate_yolo_dataset(dataset_yaml, strict=True)
        return PreparationResult(
            processed_root=final_root,
            dataset_yaml=dataset_yaml,
            manifest_path=manifest_path,
            statistics_path=statistics_path,
            validation_report_path=validation_path,
            validation_report=report,
            records=(),
            rebuilt=False,
        )

    extracted_root = settings.raw_root / "extracted"
    voc_root = discover_voc_root(
        extracted_root,
        trainval_file=settings.split.trainval_file,
        test_file=settings.split.test_file,
    )
    source_manifest = _read_json(settings.raw_root / "source_manifest.json")
    if not source_manifest.get("archive_sha256"):
        raise DatasetConversionError(
            f"Source manifest is missing a valid archive SHA-256: "
            f"{settings.raw_root / 'source_manifest.json'}"
        )
    splits = resolve_dataset_splits(
        voc_root,
        trainval_file=settings.split.trainval_file,
        test_file=settings.split.test_file,
        validation_fraction=settings.split.validation_fraction,
        seed=settings.split.seed,
        preserve_official_test=settings.split.preserve_official_test,
    )

    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{final_root.name}.",
            suffix=".tmp",
            dir=final_root.parent,
        )
    )
    try:
        records = convert_voc_dataset(voc_root, staging, splits, settings.conversion)
        effective_splits = _included_splits(splits, records)
        split_files = write_split_files(effective_splits, staging / "splits")
        _write_dataset_yaml(staging / "fod_a.yaml", staging)
        statistics = build_dataset_statistics(records)
        atomic_write_json(staging / "dataset_statistics.json", statistics)
        manifest = _build_manifest(
            settings=settings,
            source_manifest=source_manifest,
            voc_root=voc_root,
            splits=effective_splits,
            split_files=split_files,
            records=records,
            dataset_yaml_path=staging / "fod_a.yaml",
        )
        atomic_write_json(staging / "dataset_manifest.json", manifest)
        validate_yolo_dataset(staging / "fod_a.yaml", strict=True)

        _write_dataset_yaml(staging / "fod_a.yaml", final_root)
        manifest["dataset_yaml_sha256"] = sha256_file(staging / "fod_a.yaml")
        atomic_write_json(staging / "dataset_manifest.json", manifest)
        _replace_directory(staging, final_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    report = validate_yolo_dataset(dataset_yaml, strict=True)
    atomic_write_json(validation_path, report.to_dict())
    return PreparationResult(
        processed_root=final_root,
        dataset_yaml=dataset_yaml,
        manifest_path=manifest_path,
        statistics_path=statistics_path,
        validation_report_path=validation_path,
        validation_report=report,
        records=records,
        rebuilt=True,
    )


def _write_dataset_yaml(path: Path, dataset_root: Path) -> None:
    atomic_write_yaml(
        path,
        {
            "names": {0: "FOD"},
            "path": str(dataset_root.resolve()),
            "test": "images/test",
            "train": "images/train",
            "val": "images/val",
        },
    )


def _build_manifest(
    *,
    settings: DatasetSettings,
    source_manifest: dict[str, object],
    voc_root: Path,
    splits: DatasetSplits,
    split_files: dict[str, dict[str, object]],
    records: tuple[ConvertedImage, ...],
    dataset_yaml_path: Path,
) -> dict[str, object]:
    included = [record for record in records if record.included]
    counts = {
        f"{split_name}_images": sum(record.split == split_name for record in included)
        for split_name in ("train", "val", "test")
    }
    split_hashes = {
        split_name: str(metadata["sha256"]) for split_name, metadata in split_files.items()
    }
    fingerprint_payload = {
        "conversion": settings.conversion.to_dict(),
        "source_archive_sha256": source_manifest["archive_sha256"],
        "split_hashes": split_hashes,
    }
    git = inspect_git(Path(__file__).resolve().parents[3])
    try:
        relative_voc_root = voc_root.relative_to(settings.raw_root).as_posix()
    except ValueError:
        relative_voc_root = str(voc_root)
    return {
        "class_mapping": {"0": "FOD"},
        "conversion": settings.conversion.to_dict(),
        "conversion_git_commit": git.commit,
        "counts": counts,
        "dataset_fingerprint": sha256_json(fingerprint_payload),
        "dataset_yaml_sha256": sha256_file(dataset_yaml_path),
        "rejected_objects": [
            rejected.to_dict() for record in records for rejected in record.rejected_objects
        ],
        "schema_version": "1.0",
        "source_manifest": source_manifest,
        "split_configuration": settings.split.to_dict(),
        "split_files": {
            split_name: {
                "count": metadata["count"],
                "path": f"splits/{split_name}.txt",
                "sha256": metadata["sha256"],
            }
            for split_name, metadata in split_files.items()
        },
        "splits": {
            "test": list(splits.test),
            "train": list(splits.train),
            "val": list(splits.val),
        },
        "split_strategy": splits.strategy,
        "split_warnings": list(splits.warnings),
        "voc_root": relative_voc_root,
    }


def _included_splits(
    original: DatasetSplits,
    records: tuple[ConvertedImage, ...],
) -> DatasetSplits:
    included = {record.image_id for record in records if record.included}
    return DatasetSplits(
        train=tuple(identifier for identifier in original.train if identifier in included),
        val=tuple(identifier for identifier in original.val if identifier in included),
        test=tuple(identifier for identifier in original.test if identifier in included),
        strategy=original.strategy,
        seed=original.seed,
        warnings=original.warnings,
    )


def _replace_directory(staging: Path, target: Path) -> None:
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(f".{target.name}.backup-{uuid4().hex}")
        atomic_replace_path(target, backup)
    try:
        atomic_replace_path(staging, target)
    except OSError:
        if backup is not None and backup.exists() and not target.exists():
            atomic_replace_path(backup, target)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetConversionError(f"Unable to read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetConversionError(f"JSON root must be a mapping: {path}")
    return value


def _mapping(config: ConfigMapping, key: str) -> dict[str, ConfigValue]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise DatasetConversionError(f"Configuration key {key!r} must be a mapping")
    return value


def _string(mapping: Mapping[str, ConfigValue], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetConversionError(f"Configuration key {key!r} must be a non-empty string")
    return value.strip()


def _boolean(mapping: Mapping[str, ConfigValue], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise DatasetConversionError(f"Configuration key {key!r} must be a boolean")
    return value


def _integer(mapping: Mapping[str, ConfigValue], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetConversionError(f"Configuration key {key!r} must be an integer")
    return value


def _number(mapping: Mapping[str, ConfigValue], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetConversionError(f"Configuration key {key!r} must be numeric")
    return float(value)


def _relative_path(mapping: Mapping[str, ConfigValue], key: str) -> Path:
    value = Path(_string(mapping, key))
    if value.is_absolute() or ".." in value.parts:
        raise DatasetConversionError(f"Configuration key {key!r} must be a safe relative path")
    return value
