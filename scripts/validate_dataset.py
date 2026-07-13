"""Validate a processed single-class YOLO dataset and write its report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_yolo.dataset.validate import (  # noqa: E402
    StrictDatasetValidationError,
    validate_yolo_dataset,
)
from fod_yolo.hashing import AtomicWriteError, atomic_write_json  # noqa: E402
from fod_yolo.logging_utils import configure_logging  # noqa: E402
from fod_yolo.paths import ProjectPaths, resolve_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the processed-dataset validation CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Processed YOLO dataset YAML.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--duplicate-hashes", action="store_true")
    parser.add_argument(
        "--output",
        help="Validation report path; defaults beside the dataset YAML.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, persist the complete report, and return the dataset exit code."""

    args = build_parser().parse_args(argv)
    logger = configure_logging(level=args.log_level)
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT)
    dataset_yaml = paths.resolve_data_path(args.data)
    output_path = (
        resolve_path(args.output, relative_to=PROJECT_ROOT)
        if args.output
        else dataset_yaml.parent / "validation_report.json"
    )
    try:
        report = validate_yolo_dataset(
            dataset_yaml,
            strict=args.strict,
            check_duplicate_hashes=args.duplicate_hashes,
        )
    except StrictDatasetValidationError as exc:
        report = exc.report

    try:
        atomic_write_json(output_path, report.to_dict())
    except AtomicWriteError as exc:
        logger.error("Validation report could not be written: %s", exc)
        return 5

    if report.status != "pass":
        logger.error(
            "Dataset validation failed with %d error(s); report: %s",
            len(report.errors),
            output_path,
        )
        return 5
    logger.info("Dataset validation passed; report: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
