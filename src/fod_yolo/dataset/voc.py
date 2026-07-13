"""Typed and defensive Pascal VOC XML parsing."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from fod_yolo.dataset import VocParseError

MAX_XML_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pascal VOC bounding-box coordinates in source-image pixels."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True, slots=True)
class VocObject:
    """One original dataset object before single-class remapping."""

    original_class_name: str
    bbox: BoundingBox
    difficult: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class VocAnnotation:
    """One parsed Pascal VOC annotation."""

    image_id: str
    filename: str
    width: int
    height: int
    depth: int | None
    objects: tuple[VocObject, ...]
    xml_path: Path


def parse_voc_annotation(xml_path: str | Path) -> VocAnnotation:
    """Parse one VOC XML file and reject malformed or unsafe content."""

    source = Path(xml_path).expanduser().resolve()
    if not source.is_file():
        raise VocParseError(f"VOC annotation does not exist: {source}")

    try:
        file_size = source.stat().st_size
        if file_size > MAX_XML_BYTES:
            raise VocParseError(
                f"VOC annotation exceeds the {MAX_XML_BYTES}-byte safety limit: {source}"
            )
        content = source.read_bytes()
    except OSError as exc:
        raise VocParseError(f"Unable to read VOC annotation {source}: {exc}") from exc

    upper_content = content.upper()
    if b"<!DOCTYPE" in upper_content or b"<!ENTITY" in upper_content:
        raise VocParseError(f"DTD and entity declarations are not allowed in {source}")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise VocParseError(f"Invalid VOC XML in {source}: {exc}") from exc

    filename = _required_text(root, "filename", source)
    size = root.find("size")
    if size is None:
        raise VocParseError(f"Missing required <size> element in {source}")
    width = _required_int(size, "width", source)
    height = _required_int(size, "height", source)
    if width <= 0 or height <= 0:
        raise VocParseError(f"Image dimensions must be positive in {source}: {width}x{height}")
    depth = _optional_int(size, "depth", source)

    objects: list[VocObject] = []
    for object_element in root.findall("object"):
        class_name = _required_text(object_element, "name", source)
        box_element = object_element.find("bndbox")
        if box_element is None:
            raise VocParseError(f"Object {class_name!r} is missing <bndbox> in {source}")
        bbox = BoundingBox(
            xmin=_required_float(box_element, "xmin", source),
            ymin=_required_float(box_element, "ymin", source),
            xmax=_required_float(box_element, "xmax", source),
            ymax=_required_float(box_element, "ymax", source),
        )
        if not all(math.isfinite(value) for value in bbox_values(bbox)):
            raise VocParseError(f"Object {class_name!r} has non-finite coordinates in {source}")
        objects.append(
            VocObject(
                original_class_name=class_name,
                bbox=bbox,
                difficult=bool(_optional_int(object_element, "difficult", source) or 0),
                truncated=bool(_optional_int(object_element, "truncated", source) or 0),
            )
        )

    return VocAnnotation(
        image_id=source.stem,
        filename=filename,
        width=width,
        height=height,
        depth=depth,
        objects=tuple(objects),
        xml_path=source,
    )


def bbox_values(bbox: BoundingBox) -> tuple[float, float, float, float]:
    """Return coordinates in their canonical order."""

    return bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax


def _required_text(element: ET.Element, tag: str, source: Path) -> str:
    child = element.find(tag)
    value = child.text.strip() if child is not None and child.text else ""
    if not value:
        raise VocParseError(f"Missing required <{tag}> value in {source}")
    return value


def _required_int(element: ET.Element, tag: str, source: Path) -> int:
    value = _required_text(element, tag, source)
    try:
        return int(value)
    except ValueError as exc:
        raise VocParseError(f"Invalid integer <{tag}>={value!r} in {source}") from exc


def _optional_int(element: ET.Element, tag: str, source: Path) -> int | None:
    child = element.find(tag)
    if child is None or child.text is None or not child.text.strip():
        return None
    try:
        return int(child.text.strip())
    except ValueError as exc:
        raise VocParseError(f"Invalid integer <{tag}>={child.text!r} in {source}") from exc


def _required_float(element: ET.Element, tag: str, source: Path) -> float:
    value = _required_text(element, tag, source)
    try:
        return float(value)
    except ValueError as exc:
        raise VocParseError(f"Invalid number <{tag}>={value!r} in {source}") from exc
