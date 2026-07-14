"""Build a fingerprinted real-plus-synthetic YOLO fine-tuning dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import yaml

from fod_yolo.config import ConfigValue, load_config
from fod_yolo.dataset import DatasetCombinationError
from fod_yolo.dataset.discover import IMAGE_EXTENSIONS
from fod_yolo.dataset.validate import ValidationReport, validate_yolo_dataset
from fod_yolo.hashing import (
    atomic_replace_path,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    sha256_file,
    sha256_json,
)
from fod_yolo.paths import ProjectPaths

LOGGER = logging.getLogger("fod_yolo")
_SAFE_PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True, slots=True)
class CombinedDatasetSettings:
    """Resolved source and output contract for fine-tuning data."""

    config_source: Path
    runway_dataset_yaml: Path
    synthetic_root: Path
    processed_root: Path
    dataset_yaml_name: str
    image_transfer_mode: str
    seed: int
    runway_prefix: str
    synthetic_prefix: str


@dataclass(frozen=True, slots=True)
class SourceItem:
    """One paired YOLO source image and label."""

    image_id: str
    image_path: Path
    label_path: Path
    object_count: int


@dataclass(frozen=True, slots=True)
class IndexedSource:
    """Validated split membership and fingerprint for one source dataset."""

    name: str
    prefix: str
    root: Path
    splits: dict[str, tuple[SourceItem, ...]]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class CombinedDatasetResult:
    """Installed combined dataset and strict validation result."""

    processed_root: Path
    dataset_yaml: Path
    manifest_path: Path
    validation_report_path: Path
    validation_report: ValidationReport
    rebuilt: bool


def load_combined_dataset_settings(
    config_path: str | Path,
    project_paths: ProjectPaths,
    *,
    overrides: Iterable[str] = (),
) -> CombinedDatasetSettings:
    """Load portable real/synthetic dataset combination settings."""

    loaded = load_config(config_path, overrides=overrides)
    sources = _mapping(loaded.values, "sources")
    output = _mapping(loaded.values, "output")
    split = _mapping(loaded.values, "split")
    prefixes = _mapping(split, "source_prefixes")
    preserve = split.get("preserve_source_splits")
    if preserve is not True:
        raise DatasetCombinationError("split.preserve_source_splits must be true")
    settings = CombinedDatasetSettings(
        config_source=loaded.source,
        runway_dataset_yaml=project_paths.resolve_data_path(
            _string(sources, "runway_dataset_yaml")
        ),
        synthetic_root=project_paths.resolve_data_path(_string(sources, "synthetic_root")),
        processed_root=project_paths.resolve_data_path(_string(output, "processed_root")),
        dataset_yaml_name=_string(output, "dataset_yaml_name"),
        image_transfer_mode=_string(output, "image_transfer_mode"),
        seed=_integer(split, "seed"),
        runway_prefix=_string(prefixes, "runway"),
        synthetic_prefix=_string(prefixes, "synthetic"),
    )
    _validate_settings(settings)
    return settings


def prepare_combined_dataset(
    settings: CombinedDatasetSettings,
    *,
    force: bool = False,
) -> CombinedDatasetResult:
    """Validate both sources and atomically build one prefixed YOLO dataset."""

    dataset_yaml = settings.processed_root / settings.dataset_yaml_name
    manifest_path = settings.processed_root / "dataset_manifest.json"
    validation_path = settings.processed_root / "validation_report.json"
    required = (dataset_yaml, manifest_path, validation_path)
    if settings.processed_root.exists() and not force and all(path.is_file() for path in required):
        report = validate_yolo_dataset(dataset_yaml, strict=True)
        return CombinedDatasetResult(
            settings.processed_root,
            dataset_yaml,
            manifest_path,
            validation_path,
            report,
            False,
        )

    runway_root, runway_fingerprint = _validated_runway_source(settings.runway_dataset_yaml)
    LOGGER.info("Indexing original runway source dataset")
    runway = _index_source("runway", settings.runway_prefix, runway_root, runway_fingerprint)
    LOGGER.info("Indexing and fingerprinting synthetic source dataset")
    synthetic = _index_source(
        "synthetic",
        settings.synthetic_prefix,
        settings.synthetic_root,
        None,
    )
    sources = (runway, synthetic)
    _assert_no_source_content_leakage(sources)

    settings.processed_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{settings.processed_root.name}.",
            suffix=".tmp",
            dir=settings.processed_root.parent,
        )
    )
    try:
        split_ids, counts = _materialize_sources(
            sources,
            staging,
            transfer_mode=settings.image_transfer_mode,
        )
        split_files = _write_split_files(staging, split_ids)
        _write_dataset_yaml(staging / settings.dataset_yaml_name, staging)
        manifest = _manifest(settings, sources, split_ids, counts, split_files, staging)
        atomic_write_json(staging / "dataset_manifest.json", manifest)
        validate_yolo_dataset(staging / settings.dataset_yaml_name, strict=True)

        _write_dataset_yaml(staging / settings.dataset_yaml_name, settings.processed_root)
        manifest["dataset_yaml_sha256"] = sha256_file(staging / settings.dataset_yaml_name)
        atomic_write_json(staging / "dataset_manifest.json", manifest)
        _replace_directory(staging, settings.processed_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    report = validate_yolo_dataset(dataset_yaml, strict=True)
    atomic_write_json(validation_path, report.to_dict())
    return CombinedDatasetResult(
        settings.processed_root,
        dataset_yaml,
        manifest_path,
        validation_path,
        report,
        True,
    )


def _validated_runway_source(dataset_yaml: Path) -> tuple[Path, str]:
    validate_yolo_dataset(dataset_yaml, strict=True)
    try:
        config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        manifest = json.loads((dataset_yaml.parent / "dataset_manifest.json").read_text("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise DatasetCombinationError(f"Unable to read runway source metadata: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("path"), str):
        raise DatasetCombinationError("Runway dataset YAML has no valid path")
    root = Path(config["path"]).expanduser()
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    fingerprint = manifest.get("dataset_fingerprint") if isinstance(manifest, dict) else None
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise DatasetCombinationError("Runway dataset manifest has no valid fingerprint")
    return root.resolve(), fingerprint


def _index_source(
    name: str,
    prefix: str,
    root: Path,
    trusted_fingerprint: str | None,
) -> IndexedSource:
    splits: dict[str, tuple[SourceItem, ...]] = {}
    seen_ids: dict[str, str] = {}
    digest = hashlib.sha256()
    processed = 0
    for split in ("train", "val", "test"):
        images = _index_files(root / "images" / split, IMAGE_EXTENSIONS, "image")
        labels = _index_files(root / "labels" / split, (".txt",), "label")
        missing_labels = sorted(set(images).difference(labels))
        missing_images = sorted(set(labels).difference(images))
        if missing_labels or missing_images:
            raise DatasetCombinationError(
                f"{name}/{split} pairing failed; missing labels={missing_labels[:10]}, "
                f"missing images={missing_images[:10]}"
            )
        items: list[SourceItem] = []
        for image_id in sorted(images):
            folded = image_id.casefold()
            previous = seen_ids.get(folded)
            if previous is not None:
                raise DatasetCombinationError(
                    f"Source {name} reuses image ID {image_id!r} across {previous} and {split}"
                )
            seen_ids[folded] = split
            object_count = _validate_label(labels[image_id])
            item = SourceItem(image_id, images[image_id], labels[image_id], object_count)
            items.append(item)
            if trusted_fingerprint is None:
                _update_source_digest(digest, split, item)
            processed += 1
            if processed % 5000 == 0:
                LOGGER.info("Indexed %d %s images", processed, name)
        if not items:
            raise DatasetCombinationError(f"Source {name} has an empty {split} split")
        splits[split] = tuple(items)
    fingerprint = trusted_fingerprint or digest.hexdigest()
    LOGGER.info(
        "Source %s ready: train=%d val=%d test=%d fingerprint=%s",
        name,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        fingerprint,
    )
    return IndexedSource(name, prefix, root, splits, fingerprint)


def _index_files(directory: Path, extensions: tuple[str, ...], kind: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise DatasetCombinationError(f"Missing {kind} directory: {directory}")
    indexed: dict[str, Path] = {}
    casefolded: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in extensions:
            continue
        folded = path.stem.casefold()
        if folded in casefolded:
            raise DatasetCombinationError(
                f"Duplicate case-insensitive {kind} stem in {directory}: "
                f"{casefolded[folded]}, {path.stem}"
            )
        casefolded[folded] = path.stem
        indexed[path.stem] = path.resolve()
    return indexed


def _validate_label(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetCombinationError(f"Unable to read label {path}: {exc}") from exc
    objects = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise DatasetCombinationError(f"Label {path}:{line_number} must have five fields")
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise DatasetCombinationError(
                f"Label {path}:{line_number} contains a non-numeric value"
            ) from exc
        if class_id != 0:
            raise DatasetCombinationError(
                f"Label {path}:{line_number} uses class {class_id}; expected class 0 FOD"
            )
        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            raise DatasetCombinationError(f"Label {path}:{line_number} is non-finite")
        if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
            raise DatasetCombinationError(f"Label {path}:{line_number} center is outside [0, 1]")
        if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
            raise DatasetCombinationError(f"Label {path}:{line_number} size is outside (0, 1]")
        if (
            x_center - width / 2.0 < -1e-6
            or x_center + width / 2.0 > 1.0 + 1e-6
            or y_center - height / 2.0 < -1e-6
            or y_center + height / 2.0 > 1.0 + 1e-6
        ):
            raise DatasetCombinationError(f"Label {path}:{line_number} box exceeds the image")
        objects += 1
    return objects


def _update_source_digest(digest: AnyHash, split: str, item: SourceItem) -> None:
    for value in (split, item.image_id, sha256_file(item.image_path), sha256_file(item.label_path)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")


class AnyHash(Protocol):
    """Minimal structural hash interface for strict typing."""

    def update(self, value: bytes) -> None: ...


def _assert_no_source_content_leakage(sources: tuple[IndexedSource, ...]) -> None:
    memberships: dict[str, str] = {}
    for source in sources:
        for split, items in source.splits.items():
            for item in items:
                key = f"{source.prefix}__{item.image_id}".casefold()
                previous = memberships.get(key)
                if previous is not None:
                    raise DatasetCombinationError(
                        f"Combined ID collision {key!r} between {previous} "
                        f"and {source.name}/{split}"
                    )
                memberships[key] = f"{source.name}/{split}"


def _materialize_sources(
    sources: tuple[IndexedSource, ...],
    staging: Path,
    *,
    transfer_mode: str,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    split_ids: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    counts = {
        f"{split}_{kind}": 0 for split in ("train", "val", "test") for kind in ("images", "objects")
    }
    transferred = 0
    total = sum(len(items) for source in sources for items in source.splits.values())
    for split in ("train", "val", "test"):
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
        for source in sources:
            for item in source.splits[split]:
                combined_id = f"{source.prefix}__{item.image_id}"
                image_destination = (
                    staging / "images" / split / f"{combined_id}{item.image_path.suffix.casefold()}"
                )
                label_destination = staging / "labels" / split / f"{combined_id}.txt"
                _transfer(item.image_path, image_destination, transfer_mode)
                _transfer(item.label_path, label_destination, transfer_mode)
                split_ids[split].append(combined_id)
                counts[f"{split}_images"] += 1
                counts[f"{split}_objects"] += item.object_count
                transferred += 1
                if transferred % 5000 == 0 or transferred == total:
                    LOGGER.info("Materialized %d/%d combined images", transferred, total)
    return split_ids, counts


def _transfer(source: Path, destination: Path, mode: str) -> None:
    try:
        if mode == "hardlink":
            os.link(source, destination)
        elif mode == "copy":
            shutil.copy2(source, destination)
        else:
            raise DatasetCombinationError(f"Unsupported transfer mode: {mode}")
    except OSError as exc:
        raise DatasetCombinationError(
            f"Unable to {mode} {source} to {destination}: {exc}. "
            "Use output.image_transfer_mode=copy if the sources are on another filesystem."
        ) from exc


def _write_split_files(staging: Path, split_ids: dict[str, list[str]]) -> dict[str, str]:
    split_root = staging / "splits"
    hashes: dict[str, str] = {}
    for split, identifiers in split_ids.items():
        path = split_root / f"{split}.txt"
        atomic_write_text(path, "".join(f"{identifier}\n" for identifier in identifiers))
        hashes[split] = sha256_file(path)
    return hashes


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


def _manifest(
    settings: CombinedDatasetSettings,
    sources: tuple[IndexedSource, ...],
    split_ids: dict[str, list[str]],
    counts: dict[str, int],
    split_hashes: dict[str, str],
    staging: Path,
) -> dict[str, object]:
    source_metadata = {
        source.name: {
            "fingerprint": source.fingerprint,
            "prefix": source.prefix,
            "root": str(source.root),
            "split_counts": {split: len(items) for split, items in source.splits.items()},
        }
        for source in sources
    }
    fingerprint = sha256_json(
        {
            "seed": settings.seed,
            "source_fingerprints": {source.name: source.fingerprint for source in sources},
            "split_hashes": split_hashes,
        }
    )
    return {
        "class_mapping": {"0": "FOD"},
        "combination": {
            "preserve_source_splits": True,
            "randomization": "Ultralytics seeded training sampler",
            "seed": settings.seed,
        },
        "counts": counts,
        "dataset_fingerprint": fingerprint,
        "dataset_yaml_sha256": sha256_file(staging / settings.dataset_yaml_name),
        "schema_version": "1.0",
        "source_datasets": source_metadata,
        "split_files": {
            split: {
                "count": len(split_ids[split]),
                "path": f"splits/{split}.txt",
                "sha256": split_hashes[split],
            }
            for split in ("train", "val", "test")
        },
        "splits": split_ids,
    }


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


def _validate_settings(settings: CombinedDatasetSettings) -> None:
    if settings.dataset_yaml_name != "fod_combined.yaml":
        raise DatasetCombinationError("output.dataset_yaml_name must be fod_combined.yaml")
    if settings.image_transfer_mode not in {"hardlink", "copy"}:
        raise DatasetCombinationError("output.image_transfer_mode must be hardlink or copy")
    if settings.seed != 42:
        raise DatasetCombinationError("Fine-tuning dataset split seed must be 42")
    for prefix in (settings.runway_prefix, settings.synthetic_prefix):
        if _SAFE_PREFIX.fullmatch(prefix) is None:
            raise DatasetCombinationError(f"Unsafe source prefix: {prefix!r}")
    if settings.runway_prefix.casefold() == settings.synthetic_prefix.casefold():
        raise DatasetCombinationError("Source prefixes must be distinct")


def _mapping(mapping: Mapping[str, ConfigValue], key: str) -> dict[str, ConfigValue]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise DatasetCombinationError(f"Configuration key {key!r} must be a mapping")
    return value


def _string(mapping: Mapping[str, ConfigValue], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetCombinationError(f"Configuration key {key!r} must be a string")
    return value.strip()


def _integer(mapping: Mapping[str, ConfigValue], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetCombinationError(f"Configuration key {key!r} must be an integer")
    return value
