"""Fixture-backed evaluation orchestration tests with a fake Ultralytics model."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fod_yolo.dataset.pipeline import PreparationResult
from fod_yolo.evaluation import EvaluationConfigurationError
from fod_yolo.evaluation.config import load_evaluation_settings
from fod_yolo.evaluation.data import load_evaluation_images
from fod_yolo.evaluation.runner import run_evaluation
from fod_yolo.paths import ProjectPaths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEvaluationModel:
    def __init__(self, ground_truth: dict[str, list[list[float]]]) -> None:
        self.ground_truth = ground_truth
        self.val_calls: list[dict[str, object]] = []

    def val(self, **kwargs: object) -> object:
        self.val_calls.append(kwargs)
        return SimpleNamespace(
            box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.75, map75=0.55, map=0.50)
        )

    def predict(self, **kwargs: object) -> list[object]:
        source = kwargs["source"]
        sources = [source] if isinstance(source, str) else list(source)  # type: ignore[arg-type]
        results = []
        for raw_path in sources:
            path = Path(str(raw_path))
            coordinates = self.ground_truth[path.stem]
            results.append(
                SimpleNamespace(
                    path=str(path),
                    boxes=SimpleNamespace(
                        xyxyn=coordinates,
                        conf=[0.9] * len(coordinates),
                        cls=[0.0] * len(coordinates),
                    ),
                    speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0},
                )
            )
        return results


def test_validation_evaluation_writes_metrics_and_threshold_artifacts(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    images = load_evaluation_images(prepared_tiny_dataset.dataset_yaml, "val")
    ground_truth = {
        image.image_id: [[box.x1, box.y1, box.x2, box.y2] for box in image.ground_truth]
        for image in images
    }
    fake_model = FakeEvaluationModel(ground_truth)
    model_path = tmp_path / "runs" / "train" / "fixture-run" / "weights" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"fixture-model")
    paths = _project_paths(tmp_path)

    result = run_evaluation(
        model_path=model_path,
        dataset_yaml=prepared_tiny_dataset.dataset_yaml,
        split="val",
        settings=load_evaluation_settings(PROJECT_ROOT / "configs" / "evaluate.yaml"),
        project_paths=paths,
        model_factory=lambda checkpoint: fake_model,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["run_id"] == "fixture-run"
    assert metrics["split"] == "val"
    assert metrics["threshold_selection_allowed"] is True
    assert metrics["counts"]["false_negatives"] == 0
    assert metrics["metrics"]["map50_95"] == pytest.approx(0.5)
    assert metrics["latency"]["mean_end_to_end_ms"] == pytest.approx(4.0)
    assert result.threshold_csv_path.is_file()
    assert result.evaluation_directory.joinpath("ultralytics").is_dir()
    assert fake_model.val_calls[0]["split"] == "val"


def test_test_split_requires_locked_validation_threshold(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"model")

    with pytest.raises(EvaluationConfigurationError, match="locked-threshold"):
        run_evaluation(
            model_path=model,
            dataset_yaml=prepared_tiny_dataset.dataset_yaml,
            split="test",
            settings=load_evaluation_settings(PROJECT_ROOT / "configs" / "evaluate.yaml"),
            project_paths=_project_paths(tmp_path),
        )


def test_locked_test_evaluation_does_not_perform_threshold_selection(
    tmp_path: Path,
    prepared_tiny_dataset: PreparationResult,
) -> None:
    images = load_evaluation_images(prepared_tiny_dataset.dataset_yaml, "test")
    ground_truth = {
        image.image_id: [[box.x1, box.y1, box.x2, box.y2] for box in image.ground_truth]
        for image in images
    }
    fake_model = FakeEvaluationModel(ground_truth)
    model_path = tmp_path / "runs" / "train" / "fixture-run" / "weights" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"fixture-model")

    result = run_evaluation(
        model_path=model_path,
        dataset_yaml=prepared_tiny_dataset.dataset_yaml,
        split="test",
        settings=load_evaluation_settings(PROJECT_ROOT / "configs" / "evaluate.yaml"),
        project_paths=_project_paths(tmp_path),
        locked_threshold=0.42,
        model_factory=lambda checkpoint: fake_model,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    sweep = json.loads(
        result.evaluation_directory.joinpath("threshold_sweep.json").read_text(encoding="utf-8")
    )
    assert metrics["threshold_selection_allowed"] is False
    assert metrics["confidence_threshold"] == pytest.approx(0.42)
    assert len(sweep) == 1
    assert sweep[0]["threshold"] == pytest.approx(0.42)


def _project_paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=PROJECT_ROOT,
        data_root=tmp_path / "data",
        runs_root=tmp_path / "runs",
        artifacts_root=tmp_path / "artifacts",
        reports_root=tmp_path / "reports",
        models_root=tmp_path / "models",
        configs_root=PROJECT_ROOT / "configs",
    )
