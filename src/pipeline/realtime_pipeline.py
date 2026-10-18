"""Điều phối T1002; giữ API upstream qua adapter typed cục bộ."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
import logging
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal, Protocol, TypedDict, cast

import cv2
import numpy as np

from src.acquisition.camera import CameraStream
from src.acquisition.video_reader import VideoReader
from src.alert.alert_manager import AlertManager, AlertState
from src.detection.face_detector import FaceDetector
from src.detection.landmark_detector import LandmarkDetector
from src.drowsiness.decision_engine import DecisionEngine
from src.features.measurement_adapter import measure_eye, measure_mouth
from src.temporal.temporal_analyzer import TemporalAnalyzer
from src.utils.config import load_config, numeric_setting
from src.utils.contracts import (
    TemporalAnalysisResult,
    unknown_eye,
    unknown_mouth,
)
from src.visualization.overlay import HUDVisualizer

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
Mode = Literal["eye_only", "eye_yawn"]
State = Literal["NON_DROWSY", "DROWSY", "UNKNOWN"]
Points = list[tuple[float, float]]


class FaceResult(TypedDict):
    """
    Hợp đồng detector; Parameters: cờ/bbox; Returns: typed; Raises: không.
    """

    face_detected: bool
    bbox: tuple[int, int, int, int] | None


class LandmarkResult(TypedDict):
    """Mốc upstream; Parameters: điểm/cờ; Returns: typed; Raises: không."""

    landmark_valid: bool
    left_eye_points: Points | None
    right_eye_points: Points | None
    mouth_points: Points | None


class RuleInput(TypedDict):
    """Adapter luật; Parameters: hành vi; Returns: typed; Raises: không."""

    face_detected: bool
    prolonged_eye_closure: bool
    yawn_frequency: float


class DecisionResult(TypedDict):
    """
    Quyết định legacy; Parameters: state/reason; Returns: typed; Raises: không.
    """

    state: State
    reason: str


class Reader(Protocol):
    """Nguồn upstream; Parameters: không; Returns: frame; Raises: lỗi I/O."""

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Đọc nguồn; Parameters: không; Returns: iterator; Raises: lỗi I/O."""
        ...

    def release(self) -> None:
        """Đóng nguồn; Parameters: không; Returns: không; Raises: lỗi I/O."""
        ...


class FaceStage(Protocol):
    """Detector mặt; Parameters: frame; Returns: typed; Raises: lỗi backend."""

    def process(self, frame: np.ndarray) -> FaceResult:
        """Đo mặt; Parameters: BGR; Returns: typed; Raises: lỗi backend."""
        ...

    def release(self) -> None:
        """
        Đóng model; Parameters: không; Returns: không; Raises: lỗi backend.
        """
        ...


class LandmarkStage(Protocol):
    """Detector mốc; Parameters: frame; Returns: typed; Raises: lỗi backend."""

    def process(self, frame: np.ndarray) -> LandmarkResult:
        """Đo mốc; Parameters: BGR; Returns: typed; Raises: lỗi backend."""
        ...

    def release(self) -> None:
        """
        Đóng model; Parameters: không; Returns: không; Raises: lỗi backend.
        """
        ...


@dataclass(frozen=True)
class FrameResult:
    """
    Snapshot frame; Parameters: đo/quyết định; Returns: typed; Raises: không.
    """

    frame_id: int
    timestamp: float
    face_detected: bool
    temporal: TemporalAnalysisResult
    decision: DecisionResult
    alert: AlertState
    latency_ms: float
    fps: float


@dataclass(frozen=True)
class ExecutionSummary:
    """Tổng kết; Parameters: bộ đếm/timing; Returns: typed; Raises: không."""

    total_frames: int
    unknown_frames: int
    drowsiness_frames: int
    alert_count: int
    average_fps: float
    average_latency_ms: float
    elapsed_seconds: float
    interrupted: bool


class RealtimePipeline:
    """Chạy video/camera; Parameters: config và stage; Returns: runner.

    Raises: OSError, ValueError nếu cấu hình sai.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        mode: Mode | None = None,
        *,
        reader: Reader | None = None,
        face_detector: FaceStage | None = None,
        landmark_detector: LandmarkStage | None = None,
    ) -> None:
        """Nạp cấu hình chung, chưa mở thiết bị.

        Parameters: config_path, mode và dependency tùy chọn do runner sở hữu.
        Returns: Không có.
        Raises: OSError, ValueError nếu cấu hình không hợp lệ.
        """
        self.config_path = (
            ROOT / "configs/default.yaml"
            if config_path is None
            else Path(config_path)
        )
        if not self.config_path.is_absolute():
            self.config_path = ROOT / self.config_path
        self.config = load_config(self.config_path)
        if mode is not None:
            self.config.setdefault("drowsiness", {})["mode"] = mode
        self.temporal = TemporalAnalyzer(self.config_path)
        numeric_setting(self.config, "mouth", "yawn_frequency_threshold")
        self.confidence = numeric_setting(
            self.config, "face_detection", "min_detection_confidence"
        )
        if self.confidence > 1:
            raise ValueError("face_detection confidence must be <= 1")
        self.engine = DecisionEngine(config=self.config)
        self.alert_manager = AlertManager(config=self.config)
        self.visualizer = HUDVisualizer()
        self.reader = reader
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        self._used = False

    def _open(self, source: str | int | Path) -> None:
        """Mở lần lượt để finally có thể đóng tài nguyên đã tạo.

        Parameters: source là webcam, chỉ số hoặc đường dẫn video.
        Returns: Không có.
        Raises: OSError, ValueError, RuntimeError khi nguồn/model lỗi.
        """
        if self.reader is None:
            text = str(source)
            if text == "webcam" or text.isdecimal():
                self.reader = CameraStream(
                    0 if text == "webcam" else int(text)
                )
            else:
                self.reader = VideoReader(source)
        if self.face_detector is None:
            self.face_detector = FaceDetector(
                model_path=str(
                    ROOT / "models/pretrained/blaze_face_short_range.tflite"
                ),
                min_detection_confidence=self.confidence,
            )
        if self.landmark_detector is None:
            self.landmark_detector = LandmarkDetector(
                model_path=str(
                    ROOT / "models/pretrained/face_landmarker.task"
                ),
                min_detection_confidence=self.confidence,
            )

    def run(
        self,
        source: str | int | Path = "webcam",
        *,
        display: bool = True,
        max_frames: int | None = None,
        on_frame: Callable[[FrameResult], None] | None = None,
    ) -> ExecutionSummary:
        """Chạy một luồng, giữ bộ nhớ hữu hạn và luôn teardown.

        Parameters: source, display, giới hạn dương và callback snapshot.
        Returns: ExecutionSummary; alert_count đếm lần vào WARNING.
        Raises: ValueError khi giới hạn sai; lỗi stage không bị che giấu.

        HUD dùng latency suy luận hiện tại và FPS xử lý của tối đa 30 frame
        trước; snapshot/mean latency gồm cả HUD, không gồm đọc nguồn/GUI.
        average_fps là throughput toàn run, gồm mở model, I/O và teardown.
        """
        if max_frames is not None and (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer")
        if self._used:
            raise ValueError("Pipeline is single-use; create a new runner")
        self._used = True
        total = unknown = drowsy = alerts = 0
        latency_sum = 0.0
        interrupted = False
        recent: deque[float] = deque(maxlen=30)
        started = perf_counter()
        try:
            self._open(source)
            assert self.reader is not None
            assert self.face_detector is not None
            assert self.landmark_detector is not None
            frames = iter(self.reader)
            if max_frames is not None:
                frames = islice(frames, max_frames)
            for frame_id, timestamp, frame in frames:
                tick = perf_counter()
                face = self.face_detector.process(frame)
                detected = bool(face.get("face_detected", False))
                eye, mouth = unknown_eye(), unknown_mouth()
                landmarks = None
                if detected:
                    points = self.landmark_detector.process(frame)
                    if points.get("landmark_valid", False):
                        eye = measure_eye(
                            points.get("left_eye_points"),
                            points.get("right_eye_points"),
                            self.temporal.settings,
                        )
                        mouth = measure_mouth(
                            points.get("mouth_points"), self.temporal.settings
                        )
                        if eye.is_valid and mouth.is_valid:
                            landmarks = points
                temporal = self.temporal.update(
                    detected, timestamp, eye, mouth
                )
                rule_input = RuleInput(
                    face_detected=temporal.is_valid,
                    prolonged_eye_closure=(
                        temporal.behavior.is_prolonged_closure
                    ),
                    yawn_frequency=(
                        temporal.behavior.yawn_frequency_in_window
                    ),
                )
                decision = cast(
                    DecisionResult, self.engine.evaluate(rule_input)
                )
                previous_alert = self.alert_manager.state
                alert = self.alert_manager.update(decision, timestamp)
                alerts += int(
                    alert == AlertState.WARNING
                    and previous_alert != AlertState.WARNING
                )
                inference_ms = (perf_counter() - tick) * 1000
                hud_fps = len(recent) / sum(recent) if sum(recent) else 0.0
                rendered = self.visualizer.draw_hud(
                    frame,
                    temporal,
                    decision["state"],
                    fps=hud_fps,
                    latency_ms=inference_ms,
                    bbox=face.get("bbox"),
                    landmarks=landmarks,
                )
                cv2.putText(
                    rendered,
                    f"ALERT: {alert.value}",
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255) if alert != AlertState.NORMAL else (0, 255, 0),
                    2,
                )
                duration = perf_counter() - tick
                recent.append(duration)
                latency_ms = duration * 1000
                total += 1
                unknown += int(decision["state"] == "UNKNOWN")
                drowsy += int(decision["state"] == "DROWSY")
                latency_sum += latency_ms
                if on_frame is not None:
                    on_frame(
                        FrameResult(
                            frame_id,
                            timestamp,
                            detected,
                            temporal,
                            decision,
                            alert,
                            latency_ms,
                            len(recent) / sum(recent) if sum(recent) else 0.0,
                        )
                    )
                if display:
                    cv2.imshow("Driver Drowsiness Detection", rendered)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        interrupted = True
                        break
        except KeyboardInterrupt:
            interrupted = True
            LOGGER.info("Stream interrupted by user")
        finally:
            self._cleanup()
        elapsed = perf_counter() - started
        return ExecutionSummary(
            total,
            unknown,
            drowsy,
            alerts,
            total / elapsed if elapsed else 0.0,
            latency_sum / total if total else 0.0,
            elapsed,
            interrupted,
        )

    def _cleanup(self) -> None:
        """Thử tất cả thao tác đóng kể cả khi một thao tác thất bại.

        Parameters: Không có.
        Returns: Không có.
        Raises: RuntimeError khi teardown lỗi và không có lỗi gốc.
        """
        unwinding = sys.exc_info()[0] is not None
        actions = [
            self.temporal.reset,
            self.alert_manager.reset,
            cv2.destroyAllWindows,
        ]
        for component in (
            self.reader,
            self.face_detector,
            self.landmark_detector,
        ):
            if component is not None:
                actions.append(component.release)
        failures = []
        for action in actions:
            try:
                action()
            except Exception as exc:
                failures.append(exc)
                LOGGER.exception("Resource cleanup failed: %s", action)
        if failures and not unwinding:
            raise RuntimeError("Resource cleanup failed") from failures[0]
