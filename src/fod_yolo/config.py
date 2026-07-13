"""Safe YAML configuration loading and command-line-style overrides."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import yaml

ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = ConfigScalar | list["ConfigValue"] | dict[str, "ConfigValue"]
ConfigMapping: TypeAlias = dict[str, ConfigValue]


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    """A parsed configuration and the resolved file from which it was loaded."""

    source: Path
    values: ConfigMapping

    def as_dict(self) -> ConfigMapping:
        """Return a deep copy that callers may mutate safely."""

        return deepcopy(self.values)


def parse_config_text(text: str, *, source: str | Path = "<memory>") -> ConfigMapping:
    """Parse YAML text into the project's supported configuration value types."""

    source_label = str(source)
    try:
        raw_config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML configuration in {source_label}: {exc}") from exc

    normalized = _normalize_value(raw_config, location="root")
    if not isinstance(normalized, dict):
        raise ConfigError(f"Configuration {source_label} must contain a YAML mapping at the root")
    return normalized


def load_config(
    config_path: str | Path,
    *,
    overrides: Iterable[str] = (),
) -> LoadedConfig:
    """Load one UTF-8 YAML file and apply validated dotted-key overrides."""

    source = Path(config_path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")

    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"Unable to read configuration file {source}: {exc}") from exc

    values = parse_config_text(text, source=source)
    return LoadedConfig(source=source, values=apply_overrides(values, overrides))


def apply_overrides(
    config: Mapping[str, ConfigValue],
    overrides: Iterable[str],
) -> ConfigMapping:
    """Apply ``section.key=value`` overrides without mutating the input mapping.

    Override values use YAML scalar/list syntax, so values such as ``false``,
    ``null``, ``4``, and ``[phase1, local]`` retain their intended types. Every
    dotted key must already exist to turn misspellings into actionable failures.
    """

    normalized = _normalize_value(dict(config), location="root")
    if not isinstance(normalized, dict):  # Defensive; mappings normalize to dictionaries.
        raise ConfigError("Configuration root must be a mapping")
    resolved = normalized

    for assignment in overrides:
        key_path, value = _parse_override(assignment)
        cursor: ConfigMapping = resolved

        for key in key_path[:-1]:
            if key not in cursor:
                raise ConfigError(f"Unknown configuration override key: {'.'.join(key_path)}")
            child = cursor[key]
            if not isinstance(child, dict):
                raise ConfigError(f"Cannot descend through non-mapping configuration key: {key}")
            cursor = child

        final_key = key_path[-1]
        if final_key not in cursor:
            raise ConfigError(f"Unknown configuration override key: {'.'.join(key_path)}")
        cursor[final_key] = value

    return resolved


def _parse_override(assignment: str) -> tuple[list[str], ConfigValue]:
    if "=" not in assignment:
        raise ConfigError(
            f"Invalid configuration override {assignment!r}; expected dotted.key=value"
        )

    raw_key, raw_value = assignment.split("=", maxsplit=1)
    key_path = [part.strip() for part in raw_key.split(".")]
    if not key_path or any(not part for part in key_path):
        raise ConfigError(f"Invalid configuration override key: {raw_key!r}")
    if not raw_value.strip():
        raise ConfigError(
            f"Configuration override {raw_key!r} has no value; use null explicitly if intended"
        )

    try:
        parsed_value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML value in override {assignment!r}: {exc}") from exc
    return key_path, _normalize_value(parsed_value, location=f"override {raw_key!r}")


def _normalize_value(value: object, *, location: str) -> ConfigValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"Non-finite number is not allowed at {location}")
        return value
    if isinstance(value, list):
        return [
            _normalize_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: ConfigMapping = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"Configuration key at {location} must be a string: {key!r}")
            child_location = f"{location}.{key}" if location else key
            normalized[key] = _normalize_value(item, location=child_location)
        return normalized
    raise ConfigError(f"Unsupported configuration value at {location}: {type(value).__name__}")
