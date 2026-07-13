"""Project logging with UTC timestamps, rotation, and secret redaction."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

SECRET_ENVIRONMENT_VARIABLES = ("KAGGLE_KEY", "GH_TOKEN")
_MANAGED_HANDLER_NAME = "fod_yolo-managed"
_SECRET_ASSIGNMENT_PATTERN = re.compile(r"(?i)\b(KAGGLE_KEY|GH_TOKEN)\s*=\s*([^\s,;]+)")


class UtcRedactingFormatter(logging.Formatter):
    """Format human-readable UTC logs while removing configured secret values."""

    def __init__(self, *, secret_values: Iterable[str] = ()) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._secret_values = tuple(
            sorted((value for value in secret_values if value), key=len, reverse=True)
        )

    def formatTime(  # noqa: N802 - logging.Formatter defines this public name.
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        """Return an ISO-8601 UTC timestamp with millisecond precision."""

        del datefmt
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def format(self, record: logging.LogRecord) -> str:
        """Format then redact the complete message, including exception text."""

        rendered = super().format(record)
        for secret_value in self._secret_values:
            rendered = rendered.replace(secret_value, "[REDACTED]")
        return _SECRET_ASSIGNMENT_PATTERN.sub(r"\1=[REDACTED]", rendered)


def configure_logging(
    *,
    logger_name: str = "fod_yolo",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    secret_values: Iterable[str] = (),
    console_stream: TextIO | None = None,
) -> logging.Logger:
    """Configure one idempotent project logger with console and optional rotation."""

    resolved_level = _resolve_level(level)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    if backup_count < 0:
        raise ValueError("backup_count cannot be negative")

    environment_secrets = (
        os.environ.get(variable_name, "") for variable_name in SECRET_ENVIRONMENT_VARIABLES
    )
    formatter = UtcRedactingFormatter(secret_values=(*environment_secrets, *tuple(secret_values)))
    logger = logging.getLogger(logger_name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    for handler in tuple(logger.handlers):
        if handler.name == _MANAGED_HANDLER_NAME:
            logger.removeHandler(handler)
            handler.close()

    console_handler = logging.StreamHandler(console_stream or sys.stderr)
    console_handler.name = _MANAGED_HANDLER_NAME
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        resolved_log_file = Path(log_file).expanduser().resolve()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            resolved_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.name = _MANAGED_HANDLER_NAME
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        if level < 0:
            raise ValueError("Logging level cannot be negative")
        return level

    normalized = level.strip().upper()
    resolved = logging.getLevelNamesMapping().get(normalized)
    if resolved is None:
        raise ValueError(f"Unknown logging level: {level!r}")
    return resolved
