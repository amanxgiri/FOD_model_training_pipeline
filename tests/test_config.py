"""Tests for safe configuration loading and overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from fod_yolo.config import ConfigError, apply_overrides, load_config, parse_config_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


@pytest.mark.parametrize(
    "config_name",
    (
        "dataset.yaml",
        "train_yolo26n_1280.yaml",
        "evaluate.yaml",
        "promotion.yaml",
        "video_inference.yaml",
    ),
)
def test_specification_configs_load_as_root_mappings(config_name: str) -> None:
    loaded = load_config(CONFIG_ROOT / config_name)

    assert loaded.source == (CONFIG_ROOT / config_name).resolve()
    assert loaded.values


def test_training_overrides_preserve_yaml_types() -> None:
    loaded = load_config(
        CONFIG_ROOT / "train_yolo26n_1280.yaml",
        overrides=(
            "training.batch=4",
            "training.amp=false",
            "metadata.tags=[phase1, local]",
        ),
    )

    training = loaded.values["training"]
    metadata = loaded.values["metadata"]
    assert isinstance(training, dict)
    assert isinstance(metadata, dict)
    assert training["batch"] == 4
    assert training["amp"] is False
    assert metadata["tags"] == ["phase1", "local"]


def test_overrides_do_not_mutate_the_input() -> None:
    original = {"training": {"batch": -1}}

    resolved = apply_overrides(original, ("training.batch=2",))

    assert original == {"training": {"batch": -1}}
    assert resolved == {"training": {"batch": 2}}


@pytest.mark.parametrize(
    "override",
    (
        "training.missing=4",
        "training.batch",
        "training..batch=4",
        "training.batch=",
    ),
)
def test_invalid_overrides_fail_with_actionable_errors(override: str) -> None:
    with pytest.raises(ConfigError):
        apply_overrides({"training": {"batch": -1}}, (override,))


def test_config_root_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="mapping at the root"):
        parse_config_text("- one\n- two\n", source="test.yaml")


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(ConfigError, match="Non-finite"):
        parse_config_text("threshold: .nan\n", source="test.yaml")
