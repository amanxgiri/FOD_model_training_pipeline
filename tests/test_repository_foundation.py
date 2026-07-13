"""Tests for the repository and package foundation."""

from __future__ import annotations

import importlib
import tomllib
import unittest
from pathlib import Path

import fod_yolo

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RepositoryFoundationTests(unittest.TestCase):
    """Verify that the initial package can be discovered consistently."""

    def test_package_exposes_semantic_version(self) -> None:
        self.assertEqual(fod_yolo.__version__, "0.1.0")

    def test_declared_package_boundaries_are_importable(self) -> None:
        package_names = (
            "fod_yolo.dataset",
            "fod_yolo.training",
            "fod_yolo.evaluation",
            "fod_yolo.registry",
            "fod_yolo.inference",
            "fod_yolo.reporting",
        )

        for package_name in package_names:
            with self.subTest(package_name=package_name):
                self.assertIsNotNone(importlib.import_module(package_name))

    def test_pyproject_supports_local_python_version(self) -> None:
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with pyproject_path.open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)

        self.assertEqual(pyproject["project"]["requires-python"], ">=3.11")
        self.assertEqual(pyproject["tool"]["setuptools"]["package-dir"][""], "src")


if __name__ == "__main__":
    unittest.main()
