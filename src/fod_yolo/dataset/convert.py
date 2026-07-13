"""Pascal VOC to explicit single-class YOLO dataset conversion."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from fod_yolo.dataset import DatasetConversionError
from fod_yolo.dataset.discover import find_source_image
from fod_yolo.dataset.split import DatasetSplits
from fod_yolo.dataset.voc import BoundingBox, VocAnnotation, parse_voc_annotation
from fod_yolo.hashing import atomic_write_text


@dataclass(frozen=True, slots=True)
class ConversionOptions:
    """Validated behavior for one VOC-to-YOLO conversion run."""

    target_class_id: int = 0
    target_class_name: str = "FOD"
    image_transfer_mode: str = "copy"
    keep_empty_images: bool = True
    clip_boxes: bool = True
    reject_degenerate_boxes: bool = True
    verify_image_dimensions: bool = True

    def validate(self) -> None:
        """Reject unsupported options before conversion writes begin."""

        if self.target_class_id != 0 or self.target_class_name != "FOD":
            raise DatasetConversionError("Phase 1 conversion requires exactly class 0: FOD")
        if self.image_transfer_mode not in {"copy", "hardlink", "symlink"}:
            raise DatasetConversionError("image_transfer_mode must be copy, hardlink, or symlink")
        if not self.keep_empty_images:
            raise DatasetConversionError("Phase 1 conversion requires empty images to be retained")
        if not self.clip_boxes:
            raise DatasetConversionError("Phase 1 conversion requires boxes to be clipped")
        if not self.reject_degenerate_boxes:
            raise DatasetConversionError(
                "Phase 1 conversion requires degenerate boxes to be rejected"
            )

    def to_dict(self) -> dict[str, object]:
        """Return stable manifest metadata."""

        return {
            "clip_boxes": self.clip_boxes,
            "image_transfer_mode": self.image_transfer_mode,
            "keep_empty_images": self.keep_empty_images,
            "reject_degenerate_boxes": self.reject_degenerate_boxes,
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name,
            "verify_image_dimensions": self.verify_image_dimensions,
        }


@dataclass(frozen=True, slots=True)
class YoloBox:
    """One normalized single-class YOLO box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    original_class_name: str

    @property
    def area_ratio(self) -> float:
        """Return normalized box area."""

        return self.width * self.height

    def as_label_line(self) -> str:
        """Serialize with stable six-decimal coordinate precision."""

        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


@dataclass(frozen=True, slots=True)
class RejectedObject:
    """An invalid source object retained in conversion diagnostics."""

    image_id: str
    original_class_name: str
    xml_path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Return stable diagnostic metadata."""

        return {
            "image_id": self.image_id,
            "original_class_name": self.original_class_name,
            "reason": self.reason,
            "xml_path": self.xml_path,
        }


@dataclass(frozen=True, slots=True)
class ConvertedImage:
    """Conversion outcome for one source image and annotation."""

    split: str
    image_id: str
    source_image: str
    output_image: str | None
    width: int
    height: int
    xml_width: int
    xml_height: int
    original_classes: tuple[str, ...]
    boxes: tuple[YoloBox, ...]
    rejected_objects: tuple[RejectedObject, ...]
    dimension_mismatch: bool
    included: bool

    def to_dict(self) -> dict[str, object]:
        """Return detailed conversion metadata."""

        return {
            "boxes": len(self.boxes),
            "dimension_mismatch": self.dimension_mismatch,
            "height": self.height,
            "image_id": self.image_id,
            "included": self.included,
            "original_classes": list(self.original_classes),
            "output_image": self.output_image,
            "rejected_objects": [item.to_dict() for item in self.rejected_objects],
            "source_image": self.source_image,
            "split": self.split,
            "width": self.width,
            "xml_height": self.xml_height,
            "xml_width": self.xml_width,
        }


def convert_bbox_to_yolo(
    bbox: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    clip: bool,
) -> tuple[float, float, float, float]:
    """Validate, optionally clip, and normalize one VOC bounding box."""

    if image_width <= 0 or image_height <= 0:
        raise DatasetConversionError("Image dimensions must be positive")
    coordinates = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
    if not all(math.isfinite(value) for value in coordinates):
        raise DatasetConversionError("Bounding box contains NaN or infinite coordinates")

    xmin, ymin, xmax, ymax = coordinates
    if clip:
        xmin = min(max(xmin, 0.0), float(image_width))
        xmax = min(max(xmax, 0.0), float(image_width))
        ymin = min(max(ymin, 0.0), float(image_height))
        ymax = min(max(ymax, 0.0), float(image_height))
    if xmax <= xmin or ymax <= ymin:
        raise DatasetConversionError("Bounding box is degenerate after clipping")

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    normalized = (x_center, y_center, width, height)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in normalized):
        raise DatasetConversionError("Normalized bounding box falls outside [0, 1]")
    if width <= 0.0 or height <= 0.0:
        raise DatasetConversionError("Normalized bounding box has non-positive size")
    if (
        x_center - width / 2.0 < -1e-9
        or x_center + width / 2.0 > 1.0 + 1e-9
        or y_center - height / 2.0 < -1e-9
        or y_center + height / 2.0 > 1.0 + 1e-9
    ):
        raise DatasetConversionError("Normalized bounding box extends outside the image")
    return normalized


def convert_annotation(
    annotation: VocAnnotation,
    image_path: str | Path,
    *,
    split: str,
    output_root: str | Path,
    options: ConversionOptions,
) -> ConvertedImage:
    """Convert and write one annotation/image pair into a processed split."""

    options.validate()
    source_image = Path(image_path).expanduser().resolve()
    actual_width, actual_height = _read_image_dimensions(source_image)
    if options.verify_image_dimensions:
        width, height = actual_width, actual_height
    else:
        width, height = annotation.width, annotation.height
    dimension_mismatch = (actual_width, actual_height) != (annotation.width, annotation.height)

    boxes: list[YoloBox] = []
    rejected: list[RejectedObject] = []
    for source_object in annotation.objects:
        try:
            x_center, y_center, box_width, box_height = convert_bbox_to_yolo(
                source_object.bbox,
                image_width=width,
                image_height=height,
                clip=options.clip_boxes,
            )
        except DatasetConversionError as exc:
            rejected.append(
                RejectedObject(
                    image_id=annotation.image_id,
                    original_class_name=source_object.original_class_name,
                    xml_path=str(annotation.xml_path),
                    reason=str(exc),
                )
            )
            continue
        boxes.append(
            YoloBox(
                class_id=options.target_class_id,
                x_center=x_center,
                y_center=y_center,
                width=box_width,
                height=box_height,
                original_class_name=source_object.original_class_name,
            )
        )

    included = bool(boxes) or options.keep_empty_images
    output_image: Path | None = None
    if included:
        processed_root = Path(output_root).expanduser().resolve()
        output_image = (
            processed_root
            / "images"
            / split
            / f"{annotation.image_id}{source_image.suffix.lower()}"
        )
        output_label = processed_root / "labels" / split / f"{annotation.image_id}.txt"
        _transfer_image(source_image, output_image, options.image_transfer_mode)
        content = "".join(f"{box.as_label_line()}\n" for box in boxes)
        atomic_write_text(output_label, content)

    return ConvertedImage(
        split=split,
        image_id=annotation.image_id,
        source_image=str(source_image),
        output_image=str(output_image) if output_image is not None else None,
        width=width,
        height=height,
        xml_width=annotation.width,
        xml_height=annotation.height,
        original_classes=tuple(item.original_class_name for item in annotation.objects),
        boxes=tuple(boxes),
        rejected_objects=tuple(rejected),
        dimension_mismatch=dimension_mismatch,
        included=included,
    )


def convert_voc_dataset(
    voc_root: str | Path,
    output_root: str | Path,
    splits: DatasetSplits,
    options: ConversionOptions,
) -> tuple[ConvertedImage, ...]:
    """Convert every resolved split member without changing split identity."""

    root = Path(voc_root).expanduser().resolve()
    records: list[ConvertedImage] = []
    for split_name, identifiers in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        for image_id in identifiers:
            annotation = parse_voc_annotation(root / "Annotations" / f"{image_id}.xml")
            image_path = find_source_image(
                root,
                image_id,
                annotation_filename=annotation.filename,
            )
            records.append(
                convert_annotation(
                    annotation,
                    image_path,
                    split=split_name,
                    output_root=output_root,
                    options=options,
                )
            )
    return tuple(records)


def _read_image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise DatasetConversionError(f"Unreadable source image {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise DatasetConversionError(f"Image dimensions must be positive for {path}")
    return int(width), int(height)


def _transfer_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "copy":
            shutil.copy2(source, destination)
        elif mode == "hardlink":
            os.link(source, destination)
        elif mode == "symlink":
            destination.symlink_to(source)
        else:
            raise DatasetConversionError(f"Unsupported image transfer mode: {mode}")
    except OSError as exc:
        raise DatasetConversionError(
            f"Unable to {mode} source image {source} to {destination}: {exc}"
        ) from exc
