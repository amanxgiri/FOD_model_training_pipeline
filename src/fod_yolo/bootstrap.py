"""Minimal secret-safe project .env loading for top-level commands."""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class EnvironmentFileError(ValueError):
    """Raised when the local .env file cannot be parsed safely."""


def load_project_environment(
    project_root: str | Path,
    *,
    environment: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> tuple[str, ...]:
    """Load an optional ignored .env file without logging or returning secret values."""

    destination = os.environ if environment is None else environment
    env_path = Path(project_root).expanduser().resolve() / ".env"
    if not env_path.is_file():
        return ()
    try:
        content = env_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise EnvironmentFileError(f"Unable to read environment file {env_path}: {exc}") from exc
    if len(content.encode("utf-8")) > 1024 * 1024:
        raise EnvironmentFileError(f"Environment file is unexpectedly large: {env_path}")

    loaded: list[str] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise EnvironmentFileError(f"Malformed .env assignment at line {line_number}")
        raw_name, raw_value = line.split("=", maxsplit=1)
        name = raw_name.strip()
        if _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise EnvironmentFileError(f"Invalid .env variable name at line {line_number}")
        value = _parse_value(raw_value.strip(), line_number=line_number)
        if override or name not in destination:
            destination[name] = value
            loaded.append(name)
    return tuple(loaded)


def _parse_value(value: str, *, line_number: int) -> str:
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        if len(value) < 2 or value[-1] != value[0]:
            raise EnvironmentFileError(f"Unterminated quoted .env value at line {line_number}")
        return value[1:-1]
    return value.split(" #", maxsplit=1)[0].rstrip()
