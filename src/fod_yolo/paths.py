"""Portable project path discovery, environment overrides, and safety checks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class PathConfigurationError(ValueError):
    """Raised when a project path cannot be resolved safely."""


def discover_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing this project's package and metadata."""

    candidate = Path.cwd() if start is None else Path(start)
    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "src" / "fod_yolo").is_dir():
            return directory

    raise PathConfigurationError(
        f"Could not find the project root from {candidate}; expected pyproject.toml "
        "and src/fod_yolo"
    )


def resolve_path(path: str | Path, *, relative_to: str | Path) -> Path:
    """Resolve a path, anchoring relative values to a supplied directory."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(relative_to).expanduser() / candidate
    return candidate.resolve()


def ensure_within_root(
    path: str | Path,
    root: str | Path,
    *,
    description: str = "path",
) -> Path:
    """Return a resolved path only when it is contained by the allowed root."""

    resolved_root = Path(root).expanduser().resolve()
    resolved_path = Path(path).expanduser().resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PathConfigurationError(
            f"Resolved {description} is outside its allowed root: "
            f"{resolved_path} (root: {resolved_root})"
        ) from exc
    return resolved_path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved locations used by the pipeline on the current machine."""

    project_root: Path
    data_root: Path
    runs_root: Path
    artifacts_root: Path
    reports_root: Path
    models_root: Path
    configs_root: Path

    @classmethod
    def from_environment(
        cls,
        *,
        project_root: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> ProjectPaths:
        """Resolve defaults and optional FOD path overrides without creating them."""

        root = discover_project_root(project_root)
        env = os.environ if environment is None else environment

        return cls(
            project_root=root,
            data_root=_environment_path(env, "FOD_DATA_ROOT", "data", root),
            runs_root=_environment_path(env, "FOD_RUNS_ROOT", "runs", root),
            artifacts_root=_environment_path(env, "FOD_ARTIFACTS_ROOT", "artifacts", root),
            reports_root=resolve_path("reports", relative_to=root),
            models_root=resolve_path("models", relative_to=root),
            configs_root=resolve_path("configs", relative_to=root),
        )

    def create_runtime_directories(self) -> None:
        """Create the configured generated-output roots when a command needs them."""

        for directory in (
            self.data_root,
            self.runs_root,
            self.artifacts_root,
            self.reports_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_data_path(self, configured_path: str | Path) -> Path:
        """Resolve a configured data path while honoring FOD_DATA_ROOT relocation."""

        candidate = Path(configured_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        parts = candidate.parts
        if parts and parts[0].casefold() == "data":
            return self.data_root.joinpath(*parts[1:]).resolve()
        return resolve_path(candidate, relative_to=self.project_root)


def _environment_path(
    environment: Mapping[str, str],
    variable_name: str,
    default: str,
    project_root: Path,
) -> Path:
    raw_value = environment.get(variable_name)
    selected = default if raw_value is None or not raw_value.strip() else raw_value
    return resolve_path(selected, relative_to=project_root)
