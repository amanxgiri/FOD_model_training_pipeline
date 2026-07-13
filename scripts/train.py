"""Train or safely resume the Phase 1 YOLO26n detector."""

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
from fod_yolo.dataset.validate import StrictDatasetValidationError  # noqa: E402
from fod_yolo.environment import EnvironmentValidationError  # noqa: E402
from fod_yolo.logging_utils import configure_logging  # noqa: E402
from fod_yolo.paths import ProjectPaths  # noqa: E402
from fod_yolo.training import (  # noqa: E402
    TrainingConfigurationError,
    TrainingError,
    TrainingResumeError,
)
from fod_yolo.training.config import load_training_settings  # noqa: E402
from fod_yolo.training.resume import (  # noqa: E402
    load_resume_context,
    resolved_config_for_checkpoint,
)
from fod_yolo.training.trainer import read_dataset_fingerprint, run_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the training and resume CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_yolo26n_1280.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", help="Existing runs/train/<run>/weights/last.pt checkpoint.")
    parser.add_argument("--allow-dataset-change", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve configuration, execute training, and return the required exit code."""

    args = build_parser().parse_args(argv)
    logger = configure_logging(level=args.log_level)
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT)
    resume_context = None
    try:
        if args.resume:
            if args.overrides:
                raise TrainingConfigurationError(
                    "--set overrides are not accepted with --resume; the original resolved "
                    "configuration is authoritative"
                )
            original_config = resolved_config_for_checkpoint(args.resume)
            settings = load_training_settings(original_config, paths)
            fingerprint = read_dataset_fingerprint(settings.data)
            resume_context = load_resume_context(
                args.resume,
                dataset_fingerprint=fingerprint,
                allow_dataset_change=args.allow_dataset_change,
            )
        else:
            if args.allow_dataset_change:
                raise TrainingConfigurationError(
                    "--allow-dataset-change is valid only together with --resume"
                )
            settings = load_training_settings(args.config, paths, overrides=args.overrides)
    except (ConfigError, TrainingConfigurationError) as exc:
        logger.error("Training configuration is invalid: %s", exc)
        return 2
    except TrainingResumeError as exc:
        logger.error("Training resume validation failed: %s", exc)
        return 7
    except TrainingError as exc:
        logger.error("Training preflight failed: %s", exc)
        return 7

    try:
        result = run_training(
            settings,
            paths,
            allow_cpu=args.allow_cpu,
            resume=resume_context,
            allow_dataset_change=args.allow_dataset_change,
        )
    except StrictDatasetValidationError as exc:
        logger.error("Strict dataset validation failed with %d error(s)", len(exc.report.errors))
        return 5
    except EnvironmentValidationError as exc:
        logger.error("Environment or CUDA validation failed: %s", exc)
        return 6
    except TrainingError as exc:
        logger.error("Training failed: %s", exc)
        return 7
    except Exception as exc:
        logger.error("Unexpected training failure: %s: %s", type(exc).__name__, exc)
        return 7

    logger.info("Training completed for run %s", result.run_id)
    logger.info("Ultralytics run directory: %s", result.run_directory)
    logger.info("Stable candidate artifacts: %s", result.candidate_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
