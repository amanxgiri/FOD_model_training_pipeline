"""Deterministic Pascal VOC train, validation, and test split construction."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fod_yolo.dataset import DatasetDiscoveryError, DatasetSplitError
from fod_yolo.dataset.discover import find_source_image
from fod_yolo.hashing import atomic_write_text, sha256_file


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Immutable deterministic split membership and provenance."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    strategy: str
    seed: int
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return JSON-serializable split membership."""

        return {
            "seed": self.seed,
            "strategy": self.strategy,
            "test": list(self.test),
            "train": list(self.train),
            "val": list(self.val),
            "warnings": list(self.warnings),
        }


def read_split_ids(path: str | Path) -> tuple[str, ...]:
    """Read, normalize, and duplicate-check one VOC split file."""

    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetSplitError(f"Unable to read split file {source}: {exc}") from exc

    identifiers = tuple(_normalize_id(line) for line in lines if line.strip())
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise DatasetSplitError(f"Duplicate IDs in split file {source}: {', '.join(duplicates)}")
    return identifiers


def resolve_dataset_splits(
    voc_root: str | Path,
    *,
    trainval_file: str | Path,
    test_file: str | Path,
    validation_fraction: float,
    seed: int,
    preserve_official_test: bool,
) -> DatasetSplits:
    """Preserve the official test split or create a deterministic 70/15/15 fallback."""

    root = Path(voc_root).expanduser().resolve()
    if not 0.0 < validation_fraction < 1.0:
        raise DatasetSplitError("validation_fraction must be between zero and one")

    trainval_path = root / trainval_file
    test_path = root / test_file
    if preserve_official_test and trainval_path.is_file() and test_path.is_file():
        trainval_ids = read_split_ids(trainval_path)
        test_ids = read_split_ids(test_path)
        _assert_disjoint({"trainval": trainval_ids, "test": test_ids})
        if len(trainval_ids) < 2:
            raise DatasetSplitError("Official trainval split must contain at least two images")

        shuffled = sorted(trainval_ids)
        random.Random(seed).shuffle(shuffled)
        validation_count = _bounded_count(len(shuffled), validation_fraction)
        val_ids = tuple(sorted(shuffled[:validation_count]))
        train_ids = tuple(sorted(shuffled[validation_count:]))
        splits = DatasetSplits(
            train=train_ids,
            val=val_ids,
            test=tuple(sorted(test_ids)),
            strategy="official_test_trainval_80_20",
            seed=seed,
        )
    else:
        all_ids = sorted(path.stem for path in (root / "Annotations").glob("*.xml"))
        if len(all_ids) < 3:
            raise DatasetSplitError("Fallback 70/15/15 split requires at least three images")
        random.Random(seed).shuffle(all_ids)
        test_count = _bounded_count(len(all_ids), 0.15)
        validation_count = _bounded_count(len(all_ids) - test_count, 0.15 / 0.85)
        test_ids = tuple(sorted(all_ids[:test_count]))
        val_ids = tuple(sorted(all_ids[test_count : test_count + validation_count]))
        train_ids = tuple(sorted(all_ids[test_count + validation_count :]))
        warning = (
            "Configured official split files were unavailable; used a deterministic "
            "70/15/15 image-level fallback"
        )
        splits = DatasetSplits(
            train=train_ids,
            val=val_ids,
            test=test_ids,
            strategy="fallback_70_15_15",
            seed=seed,
            warnings=(warning,),
        )

    verify_split_membership(root, splits)
    return splits


def verify_split_membership(voc_root: str | Path, splits: DatasetSplits) -> None:
    """Require disjoint splits with one annotation and image for every ID."""

    root = Path(voc_root).expanduser().resolve()
    memberships = {"train": splits.train, "val": splits.val, "test": splits.test}
    _assert_disjoint(memberships)
    for split_name, identifiers in memberships.items():
        if not identifiers:
            raise DatasetSplitError(f"Resolved {split_name} split is empty")
        for image_id in identifiers:
            annotation_path = root / "Annotations" / f"{image_id}.xml"
            if not annotation_path.is_file():
                raise DatasetSplitError(
                    f"Missing annotation for {split_name} image ID {image_id}: {annotation_path}"
                )
            try:
                find_source_image(root, image_id)
            except DatasetDiscoveryError as exc:
                raise DatasetSplitError(
                    f"Missing or ambiguous image for {split_name} ID {image_id}: {exc}"
                ) from exc


def write_split_files(
    splits: DatasetSplits,
    output_directory: str | Path,
) -> dict[str, dict[str, object]]:
    """Atomically write final split ID lists and return paths, counts, and hashes."""

    output_root = Path(output_directory).expanduser().resolve()
    result: dict[str, dict[str, object]] = {}
    for split_name, identifiers in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        path = output_root / f"{split_name}.txt"
        atomic_write_text(path, "".join(f"{identifier}\n" for identifier in identifiers))
        result[split_name] = {
            "count": len(identifiers),
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return result


def _normalize_id(raw_value: str) -> str:
    normalized = raw_value.strip().replace("\\", "/")
    identifier = PurePosixPath(normalized).stem
    if not identifier or identifier in {".", ".."}:
        raise DatasetSplitError(f"Invalid image ID in split file: {raw_value!r}")
    return identifier


def _assert_disjoint(memberships: dict[str, tuple[str, ...]]) -> None:
    names = tuple(memberships)
    for index, first_name in enumerate(names):
        first = set(memberships[first_name])
        for second_name in names[index + 1 :]:
            overlap = sorted(first.intersection(memberships[second_name]))
            if overlap:
                raise DatasetSplitError(
                    f"Split overlap between {first_name} and {second_name}: {', '.join(overlap)}"
                )


def _bounded_count(total: int, fraction: float) -> int:
    return min(total - 1, max(1, round(total * fraction)))
