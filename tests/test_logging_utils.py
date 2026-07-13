"""Tests for UTC logging, secret redaction, and idempotent setup."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

from fod_yolo.logging_utils import UtcRedactingFormatter, configure_logging


def test_formatter_uses_utc_and_redacts_values_and_assignments() -> None:
    formatter = UtcRedactingFormatter(secret_values=("super-secret",))
    record = logging.LogRecord(
        name="fod_yolo.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=%s GH_TOKEN=visible-value",
        args=("super-secret",),
        exc_info=None,
    )

    rendered = formatter.format(record)

    assert "super-secret" not in rendered
    assert "visible-value" not in rendered
    assert rendered.endswith("token=[REDACTED] GH_TOKEN=[REDACTED]")
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", rendered)


def test_configure_logging_is_idempotent() -> None:
    first_stream = io.StringIO()
    second_stream = io.StringIO()
    logger = configure_logging(logger_name="fod_yolo.test.idempotent", console_stream=first_stream)
    logger = configure_logging(logger_name="fod_yolo.test.idempotent", console_stream=second_stream)

    logger.info("one message")

    assert first_stream.getvalue() == ""
    assert second_stream.getvalue().count("one message") == 1
    assert len(logger.handlers) == 1


def test_rotating_file_handler_writes_utf8_log(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "environment.log"
    logger = configure_logging(
        logger_name="fod_yolo.test.file",
        log_file=log_path,
        console_stream=io.StringIO(),
        max_bytes=1024,
        backup_count=1,
    )

    logger.info("Environment ready")
    for handler in logger.handlers:
        handler.flush()

    assert "Environment ready" in log_path.read_text(encoding="utf-8")
