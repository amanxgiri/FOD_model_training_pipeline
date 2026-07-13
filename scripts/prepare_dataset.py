"""Convert the extracted FOD-A Pascal VOC dataset into validated single-class YOLO data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_yolo.config import ConfigError  # noqa: E402
from fod_yolo.dataset import DatasetError  # noqa: E402
from fod_yolo.dataset.pipeline import (  # noqa: E402
    load_dataset_settings,
    prepare_dataset,
)
from fod_yolo.logging_utils import configure_logging  # noqa: E402
from fod_yolo.paths import ProjectPaths  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset preparation CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run discovery, split, conversion, manifest, statistics, and validation."""

    args = build_parser().parse_args(argv)
    logger = configure_logging(level=args.log_level)
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT)
    try:
        settings = load_dataset_settings(
            args.config,
            paths,
            overrides=args.overrides,
        )
        result = prepare_dataset(settings, force=args.force)
    except (ConfigError, ValueError) as exc:
        logger.error("Dataset configuration is invalid: %s", exc)
        return 2
    except DatasetError as exc:
        logger.error("Dataset preparation failed: %s", exc)
        return 5

    logger.info(
        "Processed dataset ready at %s (rebuilt=%s)",
        result.processed_root,
        result.rebuilt,
    )
    logger.info("Strict validation status: %s", result.validation_report.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
