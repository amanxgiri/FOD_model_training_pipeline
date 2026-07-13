"""Tests for typed and defensive Pascal VOC parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from fod_yolo.dataset import VocParseError
from fod_yolo.dataset.voc import parse_voc_annotation


def test_voc_parser_reads_objects_and_flags(tiny_voc_root: Path) -> None:
    annotation = parse_voc_annotation(tiny_voc_root / "Annotations" / "image001.xml")

    assert annotation.image_id == "image001"
    assert annotation.filename == "image001.ppm"
    assert (annotation.width, annotation.height, annotation.depth) == (10, 10, 3)
    assert len(annotation.objects) == 1
    assert annotation.objects[0].original_class_name == "bolt"
    assert annotation.objects[0].bbox.xmin == 1.0
    assert annotation.objects[0].difficult is False


def test_voc_parser_rejects_invalid_numeric_values(tmp_path: Path) -> None:
    annotation = tmp_path / "invalid.xml"
    annotation.write_text(
        "<annotation><filename>x.ppm</filename><size><width>bad</width>"
        "<height>10</height></size></annotation>",
        encoding="utf-8",
    )

    with pytest.raises(VocParseError, match="Invalid integer"):
        parse_voc_annotation(annotation)


def test_voc_parser_rejects_entity_declarations(tmp_path: Path) -> None:
    annotation = tmp_path / "unsafe.xml"
    annotation.write_text(
        '<!DOCTYPE x [<!ENTITY secret "value">]><annotation></annotation>',
        encoding="utf-8",
    )

    with pytest.raises(VocParseError, match="not allowed"):
        parse_voc_annotation(annotation)
