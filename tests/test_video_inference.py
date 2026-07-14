"""Fixture-free video inference tests with fake OpenCV and detector adapters."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fod_yolo.inference.config import VideoInferenceSettings
from fod_yolo.inference.detector import Detection, FODDetector, FramePrediction
from fod_yolo.inference.video import run_video_inference


class FakeFrame:
    shape = (48, 64, 3)

    def __init__(self, index: int) -> None:
        self.index = index

    def copy(self) -> FakeFrame:
        return FakeFrame(self.index)


class FakeCapture:
    def __init__(self, frames: list[FakeFrame], properties: dict[int, float]) -> None:
        self.frames = frames
        self.properties = properties
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV.
        return True

    def get(self, key: int) -> float:
        return self.properties[key]

    def set(self, key: int, value: float) -> bool:
        del key
        self.position = int(value)
        return True

    def read(self) -> tuple[bool, FakeFrame | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.frames: list[FakeFrame] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV.
        return True

    def write(self, frame: FakeFrame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_FRAME_COUNT = 4
    CAP_PROP_FOURCC = 5
    CAP_PROP_POS_FRAMES = 6
    FONT_HERSHEY_SIMPLEX = 7
    LINE_AA = 8

    def __init__(self, frame_count: int = 3) -> None:
        codec = sum(ord(character) << (8 * index) for index, character in enumerate("mp4v"))
        self.capture = FakeCapture(
            [FakeFrame(index) for index in range(frame_count)],
            {
                self.CAP_PROP_FRAME_WIDTH: 64.0,
                self.CAP_PROP_FRAME_HEIGHT: 48.0,
                self.CAP_PROP_FPS: 30.0,
                self.CAP_PROP_FRAME_COUNT: float(frame_count),
                self.CAP_PROP_FOURCC: float(codec),
            },
        )
        self.writer: FakeWriter | None = None
        self.rectangles = 0

    def VideoCapture(self, source: str) -> FakeCapture:  # noqa: N802
        del source
        return self.capture

    def VideoWriter(  # noqa: N802
        self,
        output: str,
        codec: int,
        fps: float,
        size: tuple[int, int],
    ) -> FakeWriter:
        del output, codec, size
        self.writer = FakeWriter(fps)
        return self.writer

    @staticmethod
    def VideoWriter_fourcc(*codec: str) -> int:  # noqa: N802
        return sum(ord(character) << (8 * index) for index, character in enumerate(codec))

    def rectangle(self, *args: object) -> None:
        del args
        self.rectangles += 1

    @staticmethod
    def putText(*args: object) -> None:  # noqa: N802
        del args

    @staticmethod
    def imwrite(path: str, frame: FakeFrame) -> bool:
        del frame
        Path(path).write_bytes(b"frame")
        return True


class FakeDetector:
    def __init__(self, *, interrupt_after: int | None = None) -> None:
        self.calls = 0
        self.interrupt_after = interrupt_after

    def predict_frame(self, frame: Any, **kwargs: object) -> FramePrediction:
        del frame, kwargs
        if self.interrupt_after is not None and self.calls >= self.interrupt_after:
            raise KeyboardInterrupt
        self.calls += 1
        return FramePrediction(
            detections=(Detection(0, "FOD", 0.8, 10.0, 12.0, 30.0, 32.0),),
            preprocess_ms=1.0,
            inference_ms=2.0,
            postprocess_ms=0.5,
        )


def _settings(tmp_path: Path, *, frame_stride: int = 2) -> VideoInferenceSettings:
    model = tmp_path / "best.pt"
    source = tmp_path / "sample.mp4"
    model.write_bytes(b"model")
    source.write_bytes(b"video")
    return VideoInferenceSettings(
        config_source=tmp_path / "video_inference.yaml",
        model=model,
        source=source,
        imgsz=1280,
        confidence=0.25,
        device=0,
        frame_stride=frame_stride,
        max_detections=300,
        start_time_seconds=None,
        end_time_seconds=None,
        save_annotated_video=True,
        save_detection_csv=True,
        save_summary_json=True,
        save_detection_frames=True,
        stream=True,
        output_root=tmp_path / "runs" / "inference" / "video",
        output_codec="mp4v",
    )


def test_video_inference_writes_annotated_outputs_and_statistics(tmp_path: Path) -> None:
    cv2 = FakeCv2(frame_count=3)
    detector = FakeDetector()
    ticks = iter(index * 0.01 for index in range(20))

    result = run_video_inference(
        _settings(tmp_path),
        detector_factory=lambda model: detector,
        cv2_module=cv2,
        now=lambda: datetime(2026, 7, 14, tzinfo=UTC),
        monotonic=lambda: next(ticks),
    )

    assert result.summary["completion_status"] == "complete"
    assert result.summary["evaluation_type"] == "inference-only"
    assert result.summary["frames_processed"] == 2
    assert result.summary["frames_with_detections"] == 2
    assert result.summary["total_detections"] == 2
    assert result.summary["accuracy_metrics_available"] is False
    assert cv2.writer is not None
    assert cv2.writer.fps == pytest.approx(15.0)
    assert len(cv2.writer.frames) == 2
    assert cv2.rectangles == 2
    assert len(list((result.run_directory / "detection_frames").glob("*.jpg"))) == 2

    with (result.run_directory / "detections.csv").open(encoding="utf-8", newline="") as file:
        detections = list(csv.DictReader(file))
    assert [row["frame_index"] for row in detections] == ["0", "2"]
    assert detections[0]["class_name"] == "FOD"
    assert float(detections[0]["box_area_ratio"]) == pytest.approx(
        400 / (64 * 48),
        abs=1e-6,
    )
    assert json.loads(result.summary_path.read_text(encoding="utf-8")) == result.summary


def test_interrupted_video_retains_incomplete_summary(tmp_path: Path) -> None:
    cv2 = FakeCv2(frame_count=3)
    detector = FakeDetector(interrupt_after=1)

    result = run_video_inference(
        _settings(tmp_path, frame_stride=1),
        detector_factory=lambda model: detector,
        cv2_module=cv2,
        now=lambda: datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result.summary["completion_status"] == "incomplete"
    assert result.summary["frames_processed"] == 1
    assert result.summary["error"] == "Interrupted by user"
    assert result.summary_path.is_file()
    assert cv2.capture.released is True
    assert cv2.writer is not None and cv2.writer.released is True


def test_fod_detector_normalizes_ultralytics_result(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"model")

    class Boxes:
        xyxy = [[1.0, 2.0, 11.0, 22.0]]
        conf = [0.75]
        cls = [0]

    class Result:
        boxes = Boxes()
        names = {0: "FOD"}
        speed = {"preprocess": 1.0, "inference": 3.0, "postprocess": 0.5}

    class Model:
        def predict(self, **kwargs: object) -> list[Result]:
            del kwargs
            return [Result()]

    detector = FODDetector(checkpoint, model_factory=lambda path: Model())
    prediction = detector.predict_frame(
        FakeFrame(0),
        imgsz=1280,
        confidence=0.25,
        device=0,
        max_detections=300,
    )

    assert prediction.detections == (Detection(0, "FOD", 0.75, 1.0, 2.0, 11.0, 22.0),)
    assert prediction.inference_ms == 3.0
