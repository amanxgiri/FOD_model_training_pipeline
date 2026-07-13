"""Inspect and validate the current Python, package, GPU, and model environment."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_yolo.environment import (  # noqa: E402
    ENVIRONMENT_EXIT_CODE,
    EnvironmentValidationError,
    collect_environment_report,
    format_environment_summary,
    validate_environment,
)
from fod_yolo.hashing import AtomicWriteError, atomic_write_json  # noqa: E402
from fod_yolo.logging_utils import configure_logging  # noqa: E402
from fod_yolo.paths import ProjectPaths, resolve_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the environment-check command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Require CUDA and run a CUDA tensor smoke test.",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip resolving the configured YOLO checkpoint.",
    )
    parser.add_argument("--model", default="yolo26n.pt", help="Checkpoint to resolve.")
    parser.add_argument(
        "--output",
        default="reports/environment_report.json",
        help="JSON report path, relative to the project root unless absolute.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run environment inspection, persist the report, and enforce requirements."""

    args = build_parser().parse_args(argv)
    logger = configure_logging(level=args.log_level)
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT)
    output_path = resolve_path(args.output, relative_to=paths.project_root)
    checkpoint = None if args.skip_model_check else args.model

    logger.info("Inspecting the active Python environment")
    report = collect_environment_report(
        project_root=paths.project_root,
        run_cuda_test=args.require_cuda,
        model_checkpoint=checkpoint,
    )
    try:
        atomic_write_json(output_path, report.to_dict())
    except AtomicWriteError as exc:
        logger.error("Environment report could not be written: %s", exc)
        return ENVIRONMENT_EXIT_CODE

    print(format_environment_summary(report))
    logger.info("Environment report written to %s", output_path)
    try:
        validate_environment(
            report,
            require_cuda=args.require_cuda,
            require_model_check=not args.skip_model_check,
        )
    except EnvironmentValidationError as exc:
        logger.error("Environment validation failed: %s", exc)
        return ENVIRONMENT_EXIT_CODE

    logger.info("Environment validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
