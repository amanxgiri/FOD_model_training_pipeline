"""Run streaming FOD inference and create an annotated video plus statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fod_yolo.bootstrap import EnvironmentFileError, load_project_environment  # noqa: E402
from fod_yolo.config import ConfigError  # noqa: E402
from fod_yolo.inference import (  # noqa: E402
    InferenceConfigurationError,
    InferenceError,
)
from fod_yolo.inference.config import (  # noqa: E402
    load_video_inference_settings,
    validate_video_inference_settings,
)
from fod_yolo.inference.video import run_video_inference  # noqa: E402
from fod_yolo.logging_utils import configure_logging  # noqa: E402
from fod_yolo.paths import ProjectPaths, resolve_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained best.pt checkpoint.")
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument("--config", default="configs/video_inference.yaml")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--device")
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--output-root")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--save-detection-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_project_environment(PROJECT_ROOT)
    except EnvironmentFileError as exc:
        print(f"Environment configuration is invalid: {exc}", file=sys.stderr)
        return 2
    logger = configure_logging(
        level=args.log_level,
        log_file=PROJECT_ROOT / "logs" / "infer_video.log",
    )
    paths = ProjectPaths.from_environment(project_root=PROJECT_ROOT)
    try:
        settings = load_video_inference_settings(
            args.config,
            paths,
            overrides=args.overrides,
        )
        settings = replace(
            settings,
            model=_resolve_model(args.model, paths),
            source=resolve_path(args.source, relative_to=PROJECT_ROOT),
            imgsz=settings.imgsz if args.imgsz is None else args.imgsz,
            confidence=settings.confidence if args.conf is None else args.conf,
            device=settings.device if args.device is None else _device(args.device),
            frame_stride=(
                settings.frame_stride if args.frame_stride is None else args.frame_stride
            ),
            start_time_seconds=(
                settings.start_time_seconds if args.start_time is None else args.start_time
            ),
            end_time_seconds=(
                settings.end_time_seconds if args.end_time is None else args.end_time
            ),
            output_root=(
                settings.output_root
                if args.output_root is None
                else paths.resolve_runs_path(args.output_root)
            ),
            save_annotated_video=(
                settings.save_annotated_video if args.save_video is None else args.save_video
            ),
            save_detection_csv=(
                settings.save_detection_csv if args.save_csv is None else args.save_csv
            ),
            save_detection_frames=(
                settings.save_detection_frames
                if args.save_detection_frames is None
                else args.save_detection_frames
            ),
        )
        validate_video_inference_settings(settings)
        result = run_video_inference(settings)
    except (ConfigError, InferenceConfigurationError) as exc:
        logger.error("Video inference configuration is invalid: %s", exc)
        return 2
    except InferenceError as exc:
        logger.error("Video inference failed: %s", exc)
        return 9
    except Exception as exc:
        logger.error("Unexpected video inference failure: %s: %s", type(exc).__name__, exc)
        return 9

    print(json.dumps(result.summary, indent=2, sort_keys=True))
    logger.info("Inference artifacts written to %s", result.run_directory)
    return 0 if result.summary["completion_status"] == "complete" else 130


def _resolve_model(value: str, paths: ProjectPaths) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0].casefold() == "runs":
        return paths.resolve_runs_path(candidate)
    if candidate.parts and candidate.parts[0].casefold() == "artifacts":
        return paths.resolve_artifacts_path(candidate)
    return resolve_path(candidate, relative_to=PROJECT_ROOT)


def _device(value: str) -> int | str:
    normalized = value.strip()
    try:
        return int(normalized)
    except ValueError:
        return normalized


if __name__ == "__main__":
    raise SystemExit(main())
