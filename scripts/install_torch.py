"""Install PyTorch for the active interpreter from an explicit official wheel index."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_yolo.environment import (  # noqa: E402
    ENVIRONMENT_EXIT_CODE,
    TorchInstallationError,
    build_torch_install_command,
    install_torch_packages,
    resolve_torch_index_url,
)
from fod_yolo.hashing import AtomicWriteError, atomic_write_json  # noqa: E402
from fod_yolo.logging_utils import configure_logging  # noqa: E402
from fod_yolo.paths import ProjectPaths, resolve_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit PyTorch installation command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--index-url",
        help="Official https://download.pytorch.org/whl/... index URL.",
    )
    source.add_argument(
        "--profile",
        help="Named official wheel profile such as cpu, cu128, rocm6.3, or xpu.",
    )
    parser.add_argument("--torch-version", help="Optional exact torch version.")
    parser.add_argument("--torchvision-version", help="Optional exact torchvision version.")
    parser.add_argument(
        "--report",
        default="reports/torch_install_report.json",
        help="Successful-installation JSON report path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the validated pip command without installing anything.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the wheel source and install into the active Python interpreter."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(level=args.log_level)

    try:
        index_url = resolve_torch_index_url(
            index_url=args.index_url,
            profile=args.profile,
        )
        command = build_torch_install_command(
            index_url=index_url,
            torch_version=args.torch_version,
            torchvision_version=args.torchvision_version,
            python_executable=sys.executable,
        )
    except ValueError as exc:
        parser.error(str(exc))

    display_command = subprocess.list2cmdline(command)
    if args.dry_run:
        print(display_command)
        return 0

    logger.info("Installing torch and torchvision into %s", sys.executable)
    logger.info("Using official PyTorch index %s", index_url)
    try:
        report = install_torch_packages(command)
    except TorchInstallationError as exc:
        logger.error("PyTorch installation failed: %s", exc)
        return ENVIRONMENT_EXIT_CODE

    report["index_url"] = index_url
    report["requested_profile"] = args.profile
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT)
    report_path = resolve_path(args.report, relative_to=paths.project_root)
    try:
        atomic_write_json(report_path, report)
    except AtomicWriteError as exc:
        logger.error("Installation succeeded, but its report could not be written: %s", exc)
        return ENVIRONMENT_EXIT_CODE

    logger.info("PyTorch installation completed; report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
