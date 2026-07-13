"""Tests for deterministic, disjoint dataset split construction."""

from __future__ import annotations

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
