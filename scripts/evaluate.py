"""Evaluate a trained model with Ultralytics and project-controlled safety metrics."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_yolo.bootstrap import EnvironmentFileError, load_project_environment  # noqa: E402
from fod_yolo.config import ConfigError  # noqa: E402
from fod_yolo.dataset.validate import StrictDatasetValidationError  # noqa: E402
from fod_yolo.evaluation import EvaluationConfigurationError, EvaluationError  # noqa: E402
from fod_yolo.evaluation.config import load_evaluation_settings  # noqa: E402
from fod_yolo.evaluation.runner import run_evaluation  # noqa: E402
from fod_yolo.logging_utils import configure_logging  # noqa: E402
from fod_yolo.paths import ProjectPaths, resolve_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--config", default="configs/evaluate.yaml")
    parser.add_argument("--locked-threshold", type=float)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_project_environment(PROJECT_ROOT)
    except EnvironmentFileError as exc:
        print(f"Environment configuration is invalid: {exc}", file=sys.stderr)
        return 2
    logger = configure_logging(level=args.log_level)
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT)
    try:
        settings = load_evaluation_settings(args.config)
        result = run_evaluation(
            model_path=_resolve_model_path(args.model, paths),
            dataset_yaml=paths.resolve_data_path(args.data),
            split=args.split,
            settings=settings,
            project_paths=paths,
            locked_threshold=args.locked_threshold,
        )
    except (ConfigError, EvaluationConfigurationError) as exc:
        logger.error("Evaluation configuration is invalid: %s", exc)
        return 2
    except StrictDatasetValidationError as exc:
        logger.error("Dataset validation failed with %d error(s)", len(exc.report.errors))
        return 5
    except EvaluationError as exc:
        logger.error("Evaluation failed: %s", exc)
        return 8
    except Exception as exc:
        logger.error("Unexpected evaluation failure: %s: %s", type(exc).__name__, exc)
        return 8
    logger.info("Evaluation completed: %s", result.evaluation_directory)
    return 0


def _resolve_model_path(value: str, paths: ProjectPaths) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0].casefold() == "runs":
        return paths.resolve_runs_path(candidate)
    if candidate.parts and candidate.parts[0].casefold() == "artifacts":
        return paths.resolve_artifacts_path(candidate)
    return resolve_path(candidate, relative_to=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
