"""Typed Phase 1 training configuration and portable path resolution."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from fod_yolo.config import ConfigMapping, ConfigValue, load_config
from fod_yolo.paths import ProjectPaths, resolve_path
from fod_yolo.training import TrainingConfigurationError

_SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REQUIRED_TRAINING_KEYS = {
    "amp",
    "batch",
    "cache",
    "close_mosaic",
    "deterministic",
    "device",
    "epochs",
    "exist_ok",
    "imgsz",
    "multi_scale",
    "name",
    "optimizer",
    "patience",
    "plots",
    "pretrained",
    "project",
    "rect",
    "save",
    "save_period",
    "seed",
    "val",
    "workers",
}


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Validated and machine-resolved configuration for a training run."""

    source: Path
    model: str
    data: Path
    project: Path
    name_prefix: str
    training_arguments: dict[str, ConfigValue]
    metadata: dict[str, ConfigValue]

    def resolved_config(
        self,
        *,
        run_id: str | None = None,
        device_override: str | None = None,
    ) -> ConfigMapping:
        """Return the stable YAML snapshot written before model execution."""

        training = dict(self.training_arguments)
        training["project"] = str(self.project)
        training["name"] = self.name_prefix
        training["exist_ok"] = False
        if device_override is not None:
            training["device"] = device_override
        resolved: ConfigMapping = {
            "data": str(self.data),
            "metadata": dict(self.metadata),
            "model": self.model,
            "training": training,
        }
        if run_id is not None:
            resolved["runtime"] = {
                "run_id": run_id,
                "ultralytics_arguments": self.ultralytics_arguments(
                    run_id,
                    device_override=device_override,
                ),
            }
        return resolved

    def ultralytics_arguments(
        self,
        run_id: str,
        *,
        device_override: str | None = None,
    ) -> dict[str, ConfigValue]:
        """Build exact arguments for a new run in its pre-created directory."""

        arguments = dict(self.training_arguments)
        arguments.update(
            {
                "data": str(self.data),
                "exist_ok": True,
                "name": run_id,
                "project": str(self.project),
            }
        )
        if device_override is not None:
            arguments["device"] = device_override
        return arguments


def load_training_settings(
    config_path: str | Path,
    project_paths: ProjectPaths,
    *,
    overrides: Iterable[str] = (),
    model_override: str | Path | None = None,
) -> TrainingSettings:
    """Load, validate, and relocate the Phase 1 training configuration."""

    loaded = load_config(config_path, overrides=overrides)
    config = loaded.values
    configured_model = _string(config, "model")
    model = (
        configured_model
        if model_override is None
        else str(_resolve_model_checkpoint(model_override, project_paths))
    )
    data = project_paths.resolve_data_path(_string(config, "data"))
    training = _mapping(config, "training")
    metadata = _mapping(config, "metadata")

    missing = sorted(_REQUIRED_TRAINING_KEYS.difference(training))
    if missing:
        raise TrainingConfigurationError(
            f"Training configuration is missing required keys: {', '.join(missing)}"
        )
    name = _string(training, "name")
    if _SAFE_RUN_NAME.fullmatch(name) is None:
        raise TrainingConfigurationError(
            "training.name must contain only letters, numbers, dots, underscores, and hyphens"
        )
    project = project_paths.resolve_runs_path(_string(training, "project"))
    if _boolean(training, "exist_ok"):
        raise TrainingConfigurationError(
            "training.exist_ok must be false; the orchestrator owns unique run directories"
        )
    arguments = {
        key: value for key, value in training.items() if key not in {"project", "name", "exist_ok"}
    }
    settings = TrainingSettings(
        source=loaded.source,
        model=model,
        data=data,
        project=project,
        name_prefix=name,
        training_arguments=arguments,
        metadata=metadata,
    )
    validate_training_settings(settings)
    return settings


def validate_training_settings(settings: TrainingSettings) -> None:
    """Enforce the fixed Phase 1 baseline while retaining explicit configurability."""

    mode = training_mode(settings)
    if mode == "baseline":
        if Path(settings.model).name != "yolo26n.pt":
            raise TrainingConfigurationError(
                "Phase 1 baseline training requires the yolo26n.pt checkpoint"
            )
    elif mode == "finetune":
        checkpoint = Path(settings.model).expanduser().resolve()
        if checkpoint.name != "best.pt":
            raise TrainingConfigurationError(
                "Fine-tuning requires --init-checkpoint pointing to a best.pt checkpoint"
            )
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise TrainingConfigurationError(
                f"Fine-tuning checkpoint is missing or empty: {checkpoint}"
            )
    else:
        raise TrainingConfigurationError(f"Unsupported metadata.training_mode: {mode!r}")
    arguments = settings.training_arguments
    _positive_integer(arguments, "epochs")
    if _integer(arguments, "imgsz") != 1280:
        raise TrainingConfigurationError("Phase 1 training requires training.imgsz=1280")
    batch = _integer(arguments, "batch")
    if batch == 0 or batch < -1:
        raise TrainingConfigurationError("training.batch must be -1 or a positive integer")
    if _integer(arguments, "workers") < 0:
        raise TrainingConfigurationError("training.workers cannot be negative")
    if _integer(arguments, "patience") < 0:
        raise TrainingConfigurationError("training.patience cannot be negative")
    if _integer(arguments, "seed") != 42:
        raise TrainingConfigurationError("Phase 1 training requires training.seed=42")
    if _integer(arguments, "save_period") != 10:
        raise TrainingConfigurationError("Phase 1 training requires training.save_period=10")
    if _integer(arguments, "close_mosaic") != 10:
        raise TrainingConfigurationError("Phase 1 training requires training.close_mosaic=10")
    for key in (
        "amp",
        "deterministic",
        "plots",
        "pretrained",
        "save",
        "val",
    ):
        if _boolean(arguments, key) is not True:
            raise TrainingConfigurationError(f"Phase 1 training requires training.{key}=true")
    for key in ("cache", "rect"):
        _boolean(arguments, key)
    multi_scale = arguments.get("multi_scale")
    if isinstance(multi_scale, bool) or not isinstance(multi_scale, (int, float)):
        raise TrainingConfigurationError("training.multi_scale must be numeric")
    if float(multi_scale) != 0.0:
        raise TrainingConfigurationError("Phase 1 training requires training.multi_scale=0.0")
    _string(arguments, "optimizer")
    device = arguments.get("device")
    if isinstance(device, bool) or not isinstance(device, (str, int, list)):
        raise TrainingConfigurationError("training.device must be an integer, string, or list")


def training_mode(settings: TrainingSettings) -> str:
    """Return the explicit training mode, defaulting legacy configs to baseline."""

    value = settings.metadata.get("training_mode", "baseline")
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError("metadata.training_mode must be a non-empty string")
    return value.strip().lower()


def _resolve_model_checkpoint(value: str | Path, paths: ProjectPaths) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0].casefold() == "runs":
        return paths.resolve_runs_path(candidate)
    if candidate.parts and candidate.parts[0].casefold() == "artifacts":
        return paths.resolve_artifacts_path(candidate)
    return resolve_path(candidate, relative_to=paths.project_root)


def _mapping(mapping: Mapping[str, ConfigValue], key: str) -> dict[str, ConfigValue]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"Configuration key {key!r} must be a mapping")
    return value


def _string(mapping: Mapping[str, ConfigValue], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(f"Configuration key {key!r} must be a non-empty string")
    return value.strip()


def _integer(mapping: Mapping[str, ConfigValue], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigurationError(f"Configuration key {key!r} must be an integer")
    return value


def _positive_integer(mapping: Mapping[str, ConfigValue], key: str) -> int:
    value = _integer(mapping, key)
    if value <= 0:
        raise TrainingConfigurationError(f"Configuration key {key!r} must be positive")
    return value


def _boolean(mapping: Mapping[str, ConfigValue], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise TrainingConfigurationError(f"Configuration key {key!r} must be a boolean")
    return value
