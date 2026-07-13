"""Pascal VOC root and source-image discovery."""

from __future__ import annotations

from pathlib import Path

from fod_yolo.dataset import DatasetDiscoveryError

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm")


def discover_voc_root(
    extracted_root: str | Path,
    *,
    trainval_file: str | Path = "ImageSets/Main/trainval.txt",
    test_file: str | Path = "ImageSets/Main/test.txt",
) -> Path:
    """Locate one unambiguous Pascal VOC root beneath an extracted archive."""

    root = Path(extracted_root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetDiscoveryError(f"Extracted dataset directory does not exist: {root}")

    candidates = sorted(
        {
            annotations.parent
            for annotations in root.rglob("Annotations")
            if annotations.is_dir()
            and (annotations.parent / "JPEGImages").is_dir()
            and (annotations.parent / "ImageSets" / "Main").is_dir()
        },
        key=lambda path: str(path).lower(),
    )
    if not candidates:
        raise DatasetDiscoveryError(
            f"No Pascal VOC root containing Annotations, JPEGImages, and ImageSets/Main "
            f"was found under {root}"
        )
    if len(candidates) == 1:
        return candidates[0]

    matching_splits = [
        candidate
        for candidate in candidates
        if (candidate / trainval_file).is_file() and (candidate / test_file).is_file()
    ]
    if len(matching_splits) == 1:
        return matching_splits[0]

    rendered = "\n".join(f"- {candidate}" for candidate in candidates)
    raise DatasetDiscoveryError(
        "Multiple Pascal VOC roots were found and the configured split files did not "
        f"identify exactly one candidate:\n{rendered}"
    )


def find_source_image(
    voc_root: str | Path,
    image_id: str,
    *,
    annotation_filename: str | None = None,
) -> Path:
    """Resolve exactly one image for an annotation ID or declared filename."""

    image_root = Path(voc_root).expanduser().resolve() / "JPEGImages"
    if annotation_filename:
        declared = image_root / Path(annotation_filename).name
        if declared.is_file():
            return declared.resolve()

    matches = sorted(
        path.resolve()
        for extension in IMAGE_EXTENSIONS
        for path in image_root.glob(f"{image_id}{extension}")
        if path.is_file()
    )
    if not matches:
        casefolded_id = image_id.casefold()
        matches = sorted(
            path.resolve()
            for path in image_root.iterdir()
            if path.is_file()
            and path.suffix.casefold() in IMAGE_EXTENSIONS
            and path.stem.casefold() == casefolded_id
        )
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(str(path) for path in matches)
        raise DatasetDiscoveryError(
            f"Expected exactly one source image for ID {image_id!r}; found {detail}"
        )
    return matches[0]
