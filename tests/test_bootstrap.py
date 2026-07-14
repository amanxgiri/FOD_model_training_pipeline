"""Tests for secret-safe optional project environment loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from fod_yolo.bootstrap import EnvironmentFileError, load_project_environment


def test_dotenv_loads_credentials_without_returning_values(tmp_path: Path) -> None:
    secret = "private-api-key"
    (tmp_path / ".env").write_text(
        f'KAGGLE_USERNAME="user"\nKAGGLE_KEY={secret}\n',
        encoding="utf-8",
    )
    environment: dict[str, str] = {}

    loaded = load_project_environment(tmp_path, environment=environment)

    assert loaded == ("KAGGLE_USERNAME", "KAGGLE_KEY")
    assert environment["KAGGLE_USERNAME"] == "user"
    assert environment["KAGGLE_KEY"] == secret
    assert secret not in repr(loaded)


def test_existing_operating_system_value_wins_by_default(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("KAGGLE_USERNAME=file-user\n", encoding="utf-8")
    environment = {"KAGGLE_USERNAME": "system-user"}

    assert load_project_environment(tmp_path, environment=environment) == ()
    assert environment["KAGGLE_USERNAME"] == "system-user"


def test_malformed_dotenv_fails_without_echoing_the_line(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("NOT AN ASSIGNMENT\n", encoding="utf-8")

    with pytest.raises(EnvironmentFileError, match="line 1"):
        load_project_environment(tmp_path, environment={})
