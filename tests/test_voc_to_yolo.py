"""Tests for VOC bounding-box normalization and clipping."""

from __future__ import annotations

import pytest

from fod_yolo.dataset import DatasetConversionError
from fod_yolo.dataset.convert import convert_bbox_to_yolo
from fod_yolo.dataset.voc import BoundingBox


def test_known_voc_box_converts_to_expected_yolo_coordinates() -> None:
    converted = convert_bbox_to_yolo(
        BoundingBox(xmin=10, ymin=20, xmax=30, ymax=60),
        image_width=100,
        image_height=100,
        clip=True,
    )

    assert converted == pytest.approx((0.2, 0.4, 0.2, 0.4))


def test_out_of_bounds_box_is_clipped() -> None:
    converted = convert_bbox_to_yolo(
        BoundingBox(xmin=80, ymin=80, xmax=120, ymax=120),
        image_width=100,
        image_height=100,
        clip=True,
    )

    assert converted == pytest.approx((0.9, 0.9, 0.2, 0.2))


def test_degenerate_box_is_rejected() -> None:
    with pytest.raises(DatasetConversionError, match="degenerate"):
        convert_bbox_to_yolo(
            BoundingBox(xmin=20, ymin=10, xmax=20, ymax=30),
            image_width=100,
            image_height=100,
            clip=True,
        )
