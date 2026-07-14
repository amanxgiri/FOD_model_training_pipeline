"""Tests for deterministic, disjoint dataset split construction."""

from __future__ import annotations

import shutil
from pathlib import Path

from fod_yolo.dataset.split import resolve_dataset_splits


def test_official_test_split_is_preserved_and_reproducible(tiny_voc_root: Path) -> None:
    arguments = {
        "trainval_file": Path("ImageSets/Main/trainval.txt"),
        "test_file": Path("ImageSets/Main/test.txt"),
        "validation_fraction": 0.25,
        "seed": 42,
        "preserve_official_test": True,
    }

    first = resolve_dataset_splits(tiny_voc_root, **arguments)
    second = resolve_dataset_splits(tiny_voc_root, **arguments)

    assert first == second
    assert first.test == ("image005",)
    assert len(first.train) == 3
    assert len(first.val) == 1
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train).isdisjoint(first.test)


def test_duplicate_source_ids_are_deduplicated_and_recorded(
    tiny_voc_root: Path,
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "VOC"
    shutil.copytree(tiny_voc_root, copied_root)
    trainval_path = copied_root / "ImageSets" / "Main" / "trainval.txt"
    with trainval_path.open("a", encoding="utf-8") as split_file:
        split_file.write("image001\nimage003\nimage001\n")

    splits = resolve_dataset_splits(
        copied_root,
        trainval_file=Path("ImageSets/Main/trainval.txt"),
        test_file=Path("ImageSets/Main/test.txt"),
        validation_fraction=0.25,
        seed=42,
        preserve_official_test=True,
    )

    assert len(splits.train) + len(splits.val) == 4
    assert len(set(splits.train + splits.val)) == 4
    assert splits.warnings == (
        "Source trainval split contained duplicate IDs; retained one occurrence "
        "of each: image001, image003",
    )


def test_official_test_membership_wins_when_source_lists_overlap(
    tiny_voc_root: Path,
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "VOC"
    shutil.copytree(tiny_voc_root, copied_root)
    trainval_path = copied_root / "ImageSets" / "Main" / "trainval.txt"
    with trainval_path.open("a", encoding="utf-8") as split_file:
        split_file.write("image005\n")

    splits = resolve_dataset_splits(
        copied_root,
        trainval_file=Path("ImageSets/Main/trainval.txt"),
        test_file=Path("ImageSets/Main/test.txt"),
        validation_fraction=0.25,
        seed=42,
        preserve_official_test=True,
    )

    assert splits.test == ("image005",)
    assert "image005" not in splits.train + splits.val
    assert set(splits.train).isdisjoint(splits.val)
    assert splits.warnings == (
        "Source trainval and test splits overlapped; preserved official test membership "
        "and removed the IDs from trainval: image005",
    )
