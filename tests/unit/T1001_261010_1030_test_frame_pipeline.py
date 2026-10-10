"""T1001/WP10: tích hợp phân tích khuôn mặt, kiểm thử và demo M03.

Mô tả: VideoReader/CameraStream -> LandmarkDetector -> EAR/MAR -> tracker.
Giữ hợp đồng dictionary upstream bằng TypedDict, không thêm engine DROWSY.
Tham số (Parameters): CLI --visual/--no-visual, --source, --max-frames.
Giá trị trả về (Returns): kết quả pytest hoặc báo cáo SCri014 trên stdout.
Ngoại lệ (Raises): lỗi nguồn, cấu hình, mô hình không bị che giấu.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from itertools import islice
import math
from pathlib import Path
import sys
import time
from typing import Generator, Literal, TypedDict, cast
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.camera import CameraStream  # noqa: E402
from src.acquisition.video_reader import VideoReader  # noqa: E402
from src.detection.landmark_detector import LandmarkDetector  # noqa: E402
from src.features.eye import calculate_eye_state  # noqa: E402
from src.features.mouth import calculate_mouth_state  # noqa: E402
from src.temporal.eye_closure import EyeClosureTracker  # noqa: E402
from src.utils.config import load_config  # noqa: E402

Points = list[tuple[float, float]]
EyeState = Literal["OPEN", "CLOSED", "UNKNOWN"]
MouthState = Literal["MOUTH_OPEN", "MOUTH_CLOSED", "UNKNOWN"]
SAMPLE_VIDEO = REPO_ROOT / "data/raw/d3s/d3s_video_dataset.mp4"
MODEL_PATH = REPO_ROOT / "models/pretrained/face_landmarker.task"


class DetectionResult(TypedDict):
    """Mô tả: hợp đồng detector hiện có, không thay đổi khóa upstream."""

    landmark_valid: bool
    left_eye_points: Points | None
    right_eye_points: Points | None
    mouth_points: Points | None


class EyeResult(TypedDict):
    """Mô tả: số đo EAR hữu hạn, không âm và trạng thái tức thời."""

    ear: float
    eye_state: EyeState


class MouthResult(TypedDict):
    """Mô tả: số đo MAR hữu hạn, không âm và trạng thái tức thời."""

    mar: float
    mouth_state: MouthState


class TemporalResult(TypedDict):
    """Mô tả: hợp đồng EyeClosureTracker hiện có."""

    prolonged_eye_closure: bool
    blink_detected: bool
    closed_start_time: float | None
    consecutive_closed_frames: int
    closure_duration: float


class LandmarkResult(TypedDict):
    """Mô tả: điểm pixel đã kiểm tra, hoặc None khi không hợp lệ."""

    left_eye: Points | None
    right_eye: Points | None
    mouth: Points | None


class FrameResult(TypedDict):
    """Mô tả: kết quả tích hợp PoC, không phải chẩn đoán buồn ngủ."""

    timestamp: float
    latency_ms: float
    landmark_valid: bool
    landmarks: LandmarkResult
    eye: EyeResult
    mouth: MouthResult
    temporal: TemporalResult


class PerformanceReport(TypedDict):
    """Mô tả: FPS bao gồm đọc nguồn và GUI nếu bật, không chỉ detector."""

    total_frames: int
    unknown_frames: int
    elapsed_seconds: float
    avg_latency_ms: float
    p95_latency_ms: float
    effective_fps: float
    meets_scri014: bool


def normalize_points(points: object, count: int, end: int) -> Points | None:
    """Mô tả: kiểm tra cấu trúc, tọa độ hữu hạn và chiều ngang khác 0.

    Tham số (Parameters): points từ detector; count số điểm; end mép phải.
    Giá trị trả về (Returns): danh sách pixel float, hoặc None nếu hỏng.
    Ngoại lệ (Raises): không; dữ liệu hỏng được chuyển thành UNKNOWN.
    """
    try:
        if not isinstance(points, (list, tuple, np.ndarray)):
            return None
        if len(points) != count:
            return None
        normalized = []
        for point in points:
            if len(point) != 2:
                return None
            x, y = (float(value) for value in point)
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            normalized.append((x, y))
        if normalized[0] == normalized[end]:
            return None
        return normalized
    except (TypeError, ValueError, OverflowError):
        return None


class IntegratedFacialPipeline:
    """Mô tả: điều phối PoC với chốt UNKNOWN và teardown idempotent."""

    def __init__(self) -> None:
        """Mô tả: nạp default.yaml qua API thực tế load_config().

        Tham số (Parameters): không; config riêng chưa được upstream hỗ trợ.
        Giá trị trả về (Returns): pipeline sở hữu một detector và tracker.
        Ngoại lệ (Raises): ValueError cấu hình hỏng; FileNotFoundError mô hình.
        """
        self.config = load_config()
        for section, key in (
            ("eye", "ear_threshold"),
            ("eye", "closure_time_threshold"),
            ("mouth", "mar_threshold"),
        ):
            try:
                value = float(self.config[section][key])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Cấu hình thiếu/hỏng: {section}.{key}"
                ) from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Cấu hình không hợp lệ: {section}.{key}")
        try:
            min_conf = float(
                self.config["face_detection"]["min_detection_confidence"]
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Cấu hình face_detection không hợp lệ") from exc
        if not math.isfinite(min_conf) or not 0 <= min_conf <= 1:
            raise ValueError("min_detection_confidence phải thuộc [0, 1]")
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Không tìm thấy mô hình: {MODEL_PATH}")
        self.eye_tracker = EyeClosureTracker()
        self.detector = LandmarkDetector(
            model_path=str(MODEL_PATH), min_detection_confidence=min_conf
        )
        self._released = False
        self._last_timestamp: float | None = None

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> FrameResult:
        """Mô tả: đo EAR/MAR, cập nhật temporal và latency một frame.

        Tham số (Parameters): frame BGR; timestamp giây tăng không giảm.
        Giá trị trả về (Returns): FrameResult; vùng hỏng -> UNKNOWN cả hai.
        Ngoại lệ (Raises): RuntimeError đã đóng; ValueError frame/time hỏng.
        """
        start = time.perf_counter()
        if self._released:
            raise RuntimeError("Pipeline đã giải phóng")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp phải hữu hạn và không âm")
        if self._last_timestamp is not None:
            if timestamp < self._last_timestamp:
                raise ValueError("timestamp không được giảm trong một stream")
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
            or frame.dtype != np.uint8
        ):
            raise ValueError("frame phải là ảnh BGR uint8 không rỗng")
        detection = cast(DetectionResult, self.detector.process(frame))
        left = normalize_points(detection.get("left_eye_points"), 6, 3)
        right = normalize_points(detection.get("right_eye_points"), 6, 3)
        mouth = normalize_points(detection.get("mouth_points"), 8, 4)
        valid = detection.get("landmark_valid") is True and all(
            points is not None for points in (left, right, mouth)
        )
        eye: EyeResult = {"ear": 0.0, "eye_state": "UNKNOWN"}
        lips: MouthResult = {"mar": 0.0, "mouth_state": "UNKNOWN"}
        if valid:
            eye = cast(EyeResult, calculate_eye_state(left, right))
            lips = cast(MouthResult, calculate_mouth_state(mouth, True))
            valid = (
                eye["eye_state"] != "UNKNOWN"
                and lips["mouth_state"] != "UNKNOWN"
                and math.isfinite(eye["ear"])
                and math.isfinite(lips["mar"])
                and eye["ear"] >= 0
                and lips["mar"] >= 0
            )
        if not valid:
            left = right = mouth = None
            eye = {"ear": 0.0, "eye_state": "UNKNOWN"}
            lips = {"mar": 0.0, "mouth_state": "UNKNOWN"}
        previous_closed = self.eye_tracker.consecutive_closed_frames
        temporal = cast(
            TemporalResult,
            self.eye_tracker.update(eye["eye_state"], timestamp),
        )
        if not valid:
            # Upstream giữ duration cũ trên UNKNOWN; PoC xóa ngay giá trị đó.
            temporal["closure_duration"] = 0.0
        # Frame CLOSED riêng lẻ không cảnh báo, kể cả threshold=0 hoặc
        # OPEN đến sau khoảng timestamp dài. Không sửa hành vi upstream.
        observed_closed = (
            temporal["consecutive_closed_frames"]
            if eye["eye_state"] == "CLOSED"
            else previous_closed
        )
        if observed_closed < 2:
            temporal["prolonged_eye_closure"] = False
        self._last_timestamp = timestamp
        return {
            "timestamp": timestamp,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "landmark_valid": valid,
            "landmarks": {
                "left_eye": left,
                "right_eye": right,
                "mouth": mouth,
            },
            "eye": eye,
            "mouth": lips,
            "temporal": temporal,
        }

    def release(self) -> None:
        """Mô tả: đóng detector tối đa một lần, kể cả nhiều lần teardown.

        Tham số (Parameters): không.
        Giá trị trả về (Returns): None.
        Ngoại lệ (Raises): lỗi backend khi đóng được truyền cho caller.
        """
        if not self._released:
            self._released = True
            self.detector.release()


def render_hud(
    frame: np.ndarray, frame_id: int, result: FrameResult, fps: float
) -> np.ndarray:
    """Mô tả: vẽ điểm pixel, EAR/MAR, FPS, latency và trạng thái temporal.

    Tham số (Parameters): frame BGR; frame_id; result hợp lệ; fps đo thực.
    Giá trị trả về (Returns): bản sao có HUD, không sửa frame đầu vào.
    Ngoại lệ (Raises): cv2.error nếu backend vẽ không thực thi được.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    if result["landmark_valid"]:
        for name, color in (
            ("left_eye", (0, 255, 255)),
            ("right_eye", (255, 200, 0)),
            ("mouth", (0, 165, 255)),
        ):
            for x, y in result["landmarks"][name] or []:
                cv2.circle(canvas, (round(x), round(y)), 2, color, -1)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 125), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    temporal = result["temporal"]
    lines = [
        f"Frame {frame_id} | Time {result['timestamp']:.2f}s | "
        f"FPS {fps:.1f} | Latency {result['latency_ms']:.1f}ms",
        f"EAR {result['eye']['ear']:.3f} [{result['eye']['eye_state']}] | "
        f"MAR {result['mouth']['mar']:.3f} "
        f"[{result['mouth']['mouth_state']}]",
        f"Closure {temporal['closure_duration']:.2f}s | "
        f"Frames {temporal['consecutive_closed_frames']} | "
        f"PROLONGED {temporal['prolonged_eye_closure']} | "
        f"Blink {temporal['blink_detected']}",
        "Exit: q / Q / ESC / Ctrl+C",
    ]
    color = (0, 0, 255) if temporal["prolonged_eye_closure"] else (0, 255, 0)
    scale = min(0.55, width / 1500.0)
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (8, 25 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color if index == 2 else (255, 255, 255),
            1,
        )
    if temporal["prolonged_eye_closure"]:
        cv2.putText(
            canvas,
            "WARNING: PROLONGED EYE CLOSURE!",
            (8, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 255),
            2,
        )
    return canvas


def report_performance(
    latencies: list[float], elapsed: float, unknown: int
) -> PerformanceReport:
    """Mô tả: tổng kết SCri014, kể cả khi hết nguồn/ngắt/lỗi.

    Tham số (Parameters): latencies ms từng frame; elapsed giây; unknown số lỗi.
    Giá trị trả về (Returns): thống kê; không dữ liệu -> chưa đạt.
    Ngoại lệ (Raises): không với số đo hợp lệ từ runner.
    """
    mean = float(np.mean(latencies)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    fps = len(latencies) / elapsed if elapsed > 0 else 0.0
    report: PerformanceReport = {
        "total_frames": len(latencies),
        "unknown_frames": unknown,
        "elapsed_seconds": elapsed,
        "avg_latency_ms": mean,
        "p95_latency_ms": p95,
        "effective_fps": fps,
        "meets_scri014": bool(latencies) and mean <= 50.0 and fps >= 20.0,
    }
    print("\nSCri014 PROFILING (không bao gồm khởi tạo mô hình)")
    print(f"Frames: {len(latencies)} | UNKNOWN: {unknown}")
    print(f"Elapsed: {elapsed:.3f}s | Mean: {mean:.2f}ms | P95: {p95:.2f}ms")
    print(f"Effective FPS: {fps:.2f} | SCri014: {report['meets_scri014']}")
    return report


def run_interactive_pipeline(
    video_source: str, max_frames: int | None = None, visual: bool = True
) -> PerformanceReport:
    """Mô tả: chạy video hoặc camera, cleanup cả khi lỗi/ngắt/EOF.

    Tham số (Parameters): nguồn repo-relative/absolute hoặc ID camera; giới hạn;
        visual bật GUI. Headless không gọi bất kỳ hàm cửa sổ OpenCV nào.
    Giá trị trả về (Returns): số đo latency/FPS và số frame UNKNOWN.
    Ngoại lệ (Raises): lỗi nguồn, xử lý, GUI được truyền sau teardown/report.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames phải là số nguyên dương")
    latencies: list[float] = []
    unknown = 0
    start: float | None = None
    elapsed = 0.0
    try:
        with ExitStack() as resources:
            if visual:
                resources.callback(cv2.destroyAllWindows)
            pipeline = IntegratedFacialPipeline()
            resources.callback(pipeline.release)
            if video_source.isdecimal():
                reader = CameraStream(device_id=int(video_source))
            else:
                path = Path(video_source)
                path = path if path.is_absolute() else REPO_ROOT / path
                reader = VideoReader(path.resolve())
            resources.callback(reader.release)
            print(f"[INFO] Nguồn: {video_source}; visual={visual}")
            stream = iter(reader)
            resources.callback(stream.close)
            start = time.perf_counter()
            try:
                for frame_id, timestamp, frame in islice(stream, max_frames):
                    result = pipeline.process_frame(frame, timestamp)
                    latencies.append(result["latency_ms"])
                    unknown += int(not result["landmark_valid"])
                    fps = len(latencies) / (time.perf_counter() - start)
                    if visual:
                        cv2.imshow(
                            "Facial Analysis PoC - T1001",
                            render_hud(frame, frame_id, result, fps),
                        )
                        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                            break
            finally:
                elapsed = time.perf_counter() - start
    except KeyboardInterrupt:
        print("[INFO] Đã nhận Ctrl+C; kết thúc và giải phóng tài nguyên.")
    finally:
        report = report_performance(latencies, elapsed, unknown)
    return report


# ---------------------------------------------------------------------------
# PYTEST HEADLESS: downstream thật + detector giả lập để kiểm tra lỗi xác định.
# Kiểm thử mô hình thật/video thật được tách riêng, không mock số đo profiling.
# ---------------------------------------------------------------------------


def synthetic_detection(ear: float, mar: float) -> DetectionResult:
    """Mô tả: tạo hình học có EAR/MAR biết trước, không làm ground truth D3S.

    Tham số (Parameters): ear và mar mong đợi, không phải ngưỡng vận hành.
    Giá trị trả về (Returns): hợp đồng detector với 6/6/8 điểm pixel.
    Ngoại lệ (Raises): không với các tỷ lệ hữu hạn.
    """
    eye = [
        (0, 50),
        (25, 50 - 50 * ear),
        (75, 50 - 50 * ear),
        (100, 50),
        (75, 50 + 50 * ear),
        (25, 50 + 50 * ear),
    ]
    gap = 100 * mar / 3
    mouth = [
        (0, 150),
        (25, 150 - gap),
        (50, 150 - gap),
        (75, 150 - gap),
        (100, 150),
        (75, 150 + gap),
        (50, 150 + gap),
        (25, 150 + gap),
    ]
    return {
        "landmark_valid": True,
        "left_eye_points": eye,
        "right_eye_points": eye.copy(),
        "mouth_points": mouth,
    }


@pytest.fixture
def mock_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[IntegratedFacialPipeline, Mock], None, None]:
    """Mô tả: dựng downstream thật, thay đúng biên detector.

    Tham số (Parameters): monkeypatch tự phục hồi sau test.
    Giá trị trả về (Returns): pipeline và mock detector, teardown qua finally.
    Ngoại lệ (Raises): lỗi cấu hình thật không bị bỏ qua.
    """
    detector = Mock(spec=LandmarkDetector)
    detector.process.return_value = synthetic_detection(0.5, 0.2)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "MODEL_PATH", Path(__file__))
    monkeypatch.setattr(
        module, "LandmarkDetector", Mock(return_value=detector)
    )
    pipeline = IntegratedFacialPipeline()
    try:
        yield pipeline, detector
    finally:
        pipeline.release()


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Mô tả: ảnh BGR đen dùng cho test headless.

    Tham số (Parameters): không.
    Giá trị trả về (Returns): ndarray uint8 độc lập cho từng test.
    Ngoại lệ (Raises): MemoryError nếu không cấp phát được.
    """
    return np.zeros((240, 320, 3), dtype=np.uint8)


def assert_contract(result: FrameResult) -> None:
    """Mô tả: xác nhận đủ khóa, kiểu, numeric bounds và chốt UNKNOWN.

    Tham số (Parameters): result từ pipeline thật hoặc downstream thật.
    Giá trị trả về (Returns): None nếu toàn bộ hợp đồng đúng.
    Ngoại lệ (Raises): AssertionError khi có vi phạm.
    """
    assert set(result) == {
        "timestamp",
        "latency_ms",
        "landmark_valid",
        "landmarks",
        "eye",
        "mouth",
        "temporal",
    }
    assert type(result["landmark_valid"]) is bool
    for value in (
        result["timestamp"],
        result["latency_ms"],
        result["eye"]["ear"],
        result["mouth"]["mar"],
    ):
        assert isinstance(value, float) and math.isfinite(value) and value >= 0
    assert set(result["eye"]) == {"ear", "eye_state"}
    assert set(result["mouth"]) == {"mar", "mouth_state"}
    assert result["eye"]["eye_state"] in ("OPEN", "CLOSED", "UNKNOWN")
    assert result["mouth"]["mouth_state"] in (
        "MOUTH_OPEN",
        "MOUTH_CLOSED",
        "UNKNOWN",
    )
    temporal = result["temporal"]
    assert set(temporal) == {
        "prolonged_eye_closure",
        "blink_detected",
        "closed_start_time",
        "consecutive_closed_frames",
        "closure_duration",
    }
    assert type(temporal["prolonged_eye_closure"]) is bool
    assert type(temporal["blink_detected"]) is bool
    assert type(temporal["consecutive_closed_frames"]) is int
    assert temporal["consecutive_closed_frames"] >= 0
    assert math.isfinite(temporal["closure_duration"])
    assert temporal["closure_duration"] >= 0
    start = temporal["closed_start_time"]
    assert start is None or (isinstance(start, float) and math.isfinite(start))
    assert set(result["landmarks"]) == {"left_eye", "right_eye", "mouth"}
    if result["landmark_valid"]:
        for name, count in (("left_eye", 6), ("right_eye", 6), ("mouth", 8)):
            points = result["landmarks"][name]
            assert points is not None and len(points) == count
            assert all(
                len(p) == 2 and all(map(math.isfinite, p)) for p in points
            )
    else:
        assert all(p is None for p in result["landmarks"].values())
        assert result["eye"] == {"ear": 0.0, "eye_state": "UNKNOWN"}
        assert result["mouth"] == {"mar": 0.0, "mouth_state": "UNKNOWN"}
        assert temporal == {
            "prolonged_eye_closure": False,
            "blink_detected": False,
            "closed_start_time": None,
            "consecutive_closed_frames": 0,
            "closure_duration": 0.0,
        }


@pytest.mark.parametrize(
    "region", ["left_eye_points", "right_eye_points", "mouth_points"]
)
@pytest.mark.parametrize(
    "corruption",
    [
        None,
        [],
        [(float("nan"), 0)] * 8,
        [(float("inf"), 0)] * 6,
        [(0, 0)] * 6,
        "broken",
    ],
)
def test_corrupted_landmarks_reset_both_branches(
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    blank_frame: np.ndarray,
    region: str,
    corruption: object,
) -> None:
    """Mô tả: một vùng hỏng làm cả hai UNKNOWN và xóa closure đang tích lũy.

    Tham số (Parameters): pipeline, ảnh đen, vùng và lỗi tham số hóa.
    Giá trị trả về (Returns): None khi chốt FReq004/FReq030 đúng.
    Ngoại lệ (Raises): AssertionError nếu cảnh báo còn giữ sau mất dấu.
    """
    pipeline, detector = mock_pipeline
    detector.process.return_value = synthetic_detection(0.0, 0.2)
    pipeline.process_frame(blank_frame, 0.0)
    threshold = pipeline.eye_tracker.closure_time_threshold
    prolonged = pipeline.process_frame(blank_frame, threshold)
    assert prolonged["temporal"]["prolonged_eye_closure"] is True
    damaged = synthetic_detection(0.0, 0.2)
    damaged[region] = corruption
    detector.process.return_value = damaged
    result = pipeline.process_frame(blank_frame, threshold + 0.1)
    assert result["landmark_valid"] is False
    assert_contract(result)
    detector.process.return_value = synthetic_detection(0.0, 0.2)
    restarted = pipeline.process_frame(blank_frame, threshold + 0.2)
    assert restarted["temporal"]["consecutive_closed_frames"] == 1
    assert restarted["temporal"]["closure_duration"] == 0.0
    assert not restarted["temporal"]["prolonged_eye_closure"]


def test_missing_face_and_single_blink_safety(
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    blank_frame: np.ndarray,
) -> None:
    """Mô tả: cờ mất mặt thắng tọa độ còn sót; một CLOSED không cảnh báo.

    Tham số (Parameters): pipeline mock detector và ảnh BGR.
    Giá trị trả về (Returns): None khi red line được bảo vệ.
    Ngoại lệ (Raises): AssertionError nếu một blink gây prolonged closure.
    """
    pipeline, detector = mock_pipeline
    threshold = pipeline.eye_tracker.closure_time_threshold
    detector.process.return_value = synthetic_detection(0.0, 0.2)
    assert not pipeline.process_frame(blank_frame, 0.0)["temporal"][
        "prolonged_eye_closure"
    ]
    detector.process.return_value = synthetic_detection(0.5, 0.2)
    assert not pipeline.process_frame(blank_frame, threshold + 1)["temporal"][
        "prolonged_eye_closure"
    ]
    detector.process.return_value["landmark_valid"] = False
    result = pipeline.process_frame(blank_frame, threshold + 2)
    assert not result["landmark_valid"]
    assert_contract(result)


def test_short_blink_and_hud(
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    blank_frame: np.ndarray,
) -> None:
    """Mô tả: blink ngắn, tỷ lệ biết trước và HUD không sửa ảnh nguồn.

    Tham số (Parameters): pipeline, ảnh đen.
    Giá trị trả về (Returns): None khi numeric/state/overlay hợp lệ.
    Ngoại lệ (Raises): AssertionError khi tích hợp sai.
    """
    pipeline, detector = mock_pipeline
    detector.process.return_value = synthetic_detection(0.0, 0.2)
    first = pipeline.process_frame(blank_frame, 0.0)
    assert first["eye"]["eye_state"] == "CLOSED"
    detector.process.return_value = synthetic_detection(0.5, 0.2)
    result = pipeline.process_frame(
        blank_frame, pipeline.eye_tracker.closure_time_threshold / 2
    )
    assert_contract(result)
    assert result["temporal"]["blink_detected"]
    assert not result["temporal"]["prolonged_eye_closure"]
    assert result["eye"]["ear"] == pytest.approx(0.5)
    assert result["mouth"]["mar"] == pytest.approx(0.2)
    canvas = render_hud(blank_frame, 1, result, 30.0)
    assert canvas.shape == blank_frame.shape
    assert np.any(canvas != blank_frame)
    assert not np.any(blank_frame)


@pytest.mark.parametrize("timestamp", [-1.0, float("nan"), float("inf")])
def test_invalid_timestamp(
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    blank_frame: np.ndarray,
    timestamp: float,
) -> None:
    """Mô tả: từ chối timestamp sai trước khi detector chạy.

    Tham số (Parameters): pipeline, frame, timestamp sai.
    Giá trị trả về (Returns): None nếu có ValueError.
    Ngoại lệ (Raises): AssertionError nếu đầu vào sai được chấp nhận.
    """
    pipeline, detector = mock_pipeline
    with pytest.raises(ValueError, match="timestamp"):
        pipeline.process_frame(blank_frame, timestamp)
    detector.process.assert_not_called()


def test_release_idempotent_and_monotonic_time(
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    blank_frame: np.ndarray,
) -> None:
    """Mô tả: thời gian không giảm; detector đóng một lần; cấm dùng sau đóng.

    Tham số (Parameters): pipeline và ảnh BGR.
    Giá trị trả về (Returns): None nếu kiểm tra vòng đời đúng.
    Ngoại lệ (Raises): AssertionError khi teardown không an toàn.
    """
    pipeline, detector = mock_pipeline
    pipeline.process_frame(blank_frame, 1.0)
    with pytest.raises(ValueError, match="timestamp"):
        pipeline.process_frame(blank_frame, 0.0)
    pipeline.release()
    pipeline.release()
    detector.release.assert_called_once()
    with pytest.raises(RuntimeError, match="giải phóng"):
        pipeline.process_frame(blank_frame, 2.0)


@pytest.mark.parametrize("value", [None, -1, float("nan"), "bad"])
def test_invalid_config_stops_before_detector(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    """Mô tả: cấu hình hỏng phải dừng, không dùng fallback cứng của mouth.

    Tham số (Parameters): monkeypatch và giá trị MAR không hợp lệ.
    Giá trị trả về (Returns): None khi có lỗi cấu hình rõ ràng.
    Ngoại lệ (Raises): AssertionError nếu detector được mở với config hỏng.
    """
    config = load_config()
    config["mouth"]["mar_threshold"] = value
    module = sys.modules[__name__]
    factory = Mock()
    monkeypatch.setattr(module, "load_config", lambda: config)
    monkeypatch.setattr(module, "LandmarkDetector", factory)
    with pytest.raises(ValueError, match="mouth.mar_threshold"):
        IntegratedFacialPipeline()
    factory.assert_not_called()


def test_thresholds_loaded_dynamically(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    blank_frame: np.ndarray,
) -> None:
    """Mô tả: thay loader trong bộ nhớ, không sửa configs read-only.

    Tham số (Parameters): monkeypatch; pipeline; frame.
    Giá trị trả về (Returns): None nếu EAR/MAR/tracker dùng config mới.
    Ngoại lệ (Raises): AssertionError nếu ngưỡng bị hardcode.
    """
    config = load_config()
    config["eye"]["ear_threshold"] = 0.8
    config["mouth"]["mar_threshold"] = 0.1
    config["eye"]["closure_time_threshold"] = 0.0
    for name in (
        __name__,
        "src.features.eye",
        "src.features.mouth",
        "src.temporal.eye_closure",
    ):
        monkeypatch.setattr(sys.modules[name], "load_config", lambda: config)
    pipeline = IntegratedFacialPipeline()
    try:
        result = pipeline.process_frame(blank_frame, 0.0)
        assert result["eye"]["eye_state"] == "CLOSED"
        assert result["mouth"]["mouth_state"] == "MOUTH_OPEN"
        assert pipeline.eye_tracker.closure_time_threshold == 0.0
        assert not result["temporal"]["prolonged_eye_closure"]
        result = pipeline.process_frame(blank_frame, 0.1)
        assert result["temporal"]["prolonged_eye_closure"]
    finally:
        pipeline.release()


@pytest.fixture
def short_video(tmp_path: Path) -> Path:
    """Mô tả: tạo video MJPG ba frame, không ghi vào data/raw.

    Tham số (Parameters): thư mục tạm pytest.
    Giá trị trả về (Returns): đường dẫn video kiểm thử acquisition thật.
    Ngoại lệ (Raises): AssertionError nếu codec không mở được.
    """
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48)
    )
    try:
        assert writer.isOpened(), "Không mở được MJPG VideoWriter"
        for value in (0, 60, 120):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("limit, expected", [(None, 3), (2, 2)])
def test_headless_runner_eof_and_frame_limit(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    short_video: Path,
    limit: int | None,
    expected: int,
) -> None:
    """Mô tả: acquisition thật, EOF/giới hạn và tuyệt đối không gọi GUI.

    Tham số (Parameters): patch; pipeline; video tạm; số frame giới hạn/kỳ vọng.
    Giá trị trả về (Returns): None nếu capture/detector đóng đúng.
    Ngoại lệ (Raises): AssertionError nếu GUI bị gọi hoặc tài nguyên còn mở.
    """
    pipeline, detector = mock_pipeline
    reader = VideoReader(short_video)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "IntegratedFacialPipeline", lambda: pipeline)
    monkeypatch.setattr(module, "VideoReader", lambda path: reader)
    for name in ("imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, Mock(side_effect=AssertionError(name)))
    report = run_interactive_pipeline(str(short_video), limit, visual=False)
    assert report["total_frames"] == expected
    assert detector.process.call_count == expected
    assert report["unknown_frames"] == 0
    assert report["effective_fps"] > 0
    assert not reader._capture.isOpened()
    detector.release.assert_called_once()


@pytest.mark.parametrize("exit_mode", ["q", "Q", "ESC", "interrupt", "error"])
def test_visual_exit_cleanup_without_real_gui(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    short_video: Path,
    exit_mode: str,
) -> None:
    """Mô tả: kiểm tra các đường thoát visual bằng GUI mock, không mở cửa sổ.

    Tham số (Parameters): patch, pipeline, video thật và cách thoát.
    Giá trị trả về (Returns): None khi capture/detector/GUI teardown đủ.
    Ngoại lệ (Raises): AssertionError nếu thiếu cleanup hoặc nuốt lỗi xử lý.
    """
    pipeline, detector = mock_pipeline
    reader = VideoReader(short_video)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "IntegratedFacialPipeline", lambda: pipeline)
    monkeypatch.setattr(module, "VideoReader", lambda path: reader)
    monkeypatch.setattr(cv2, "imshow", Mock())
    wait = Mock(return_value=27 if exit_mode == "ESC" else ord(exit_mode[0]))
    monkeypatch.setattr(cv2, "waitKey", wait)
    destroy = Mock()
    monkeypatch.setattr(cv2, "destroyAllWindows", destroy)
    if exit_mode in ("interrupt", "error"):
        wait.side_effect = (
            KeyboardInterrupt()
            if exit_mode == "interrupt"
            else RuntimeError("injected GUI failure")
        )
    if exit_mode == "error":
        with pytest.raises(RuntimeError, match="injected"):
            run_interactive_pipeline(str(short_video), visual=True)
    else:
        report = run_interactive_pipeline(str(short_video), visual=True)
        assert report["total_frames"] == 1
    assert not reader._capture.isOpened()
    detector.release.assert_called_once()
    destroy.assert_called_once()


def test_source_failure_releases_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    tmp_path: Path,
) -> None:
    """Mô tả: mở nguồn thất bại vẫn đóng detector đã khởi tạo.

    Tham số (Parameters): patch, pipeline, đường dẫn chắc chắn thiếu.
    Giá trị trả về (Returns): None khi lỗi nguồn rõ ràng và không rò detector.
    Ngoại lệ (Raises): AssertionError nếu thiếu FileNotFoundError/cleanup.
    """
    pipeline, detector = mock_pipeline
    monkeypatch.setattr(
        sys.modules[__name__], "IntegratedFacialPipeline", lambda: pipeline
    )
    with pytest.raises(FileNotFoundError):
        run_interactive_pipeline(str(tmp_path / "missing.mp4"), visual=False)
    detector.release.assert_called_once()


def test_camera_id_uses_camera_stream(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline: tuple[IntegratedFacialPipeline, Mock],
    blank_frame: np.ndarray,
) -> None:
    """Mô tả: ID camera chuyển qua CameraStream, không kiểm tra như file.

    Tham số (Parameters): patch, pipeline, frame; không dùng camera vật lý.
    Giá trị trả về (Returns): None nếu chọn đúng nguồn và teardown.
    Ngoại lệ (Raises): AssertionError khi camera bị hiểu thành đường dẫn.
    """
    pipeline, detector = mock_pipeline
    detector.process.return_value["landmark_valid"] = False
    reader = Mock()
    reader.__iter__ = Mock(
        return_value=((i, i / 25.0, blank_frame) for i in range(2))
    )
    factory = Mock(return_value=reader)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "IntegratedFacialPipeline", lambda: pipeline)
    monkeypatch.setattr(module, "CameraStream", factory)
    report = run_interactive_pipeline("0", max_frames=1, visual=False)
    factory.assert_called_once_with(device_id=0)
    reader.release.assert_called_once()
    detector.release.assert_called_once()
    assert report["total_frames"] == report["unknown_frames"] == 1


def test_profiling_statistics() -> None:
    """Mô tả: xác nhận công thức mean, P95, FPS và benchmark rỗng.

    Tham số (Parameters): không; số đo tổng hợp biết trước.
    Giá trị trả về (Returns): None khi số đo/thông báo đạt chính xác.
    Ngoại lệ (Raises): AssertionError khi báo đạt cho nguồn rỗng/chậm.
    """
    report = report_performance([10.0, 20.0, 30.0, 40.0], 0.1, 1)
    assert report["avg_latency_ms"] == 25.0
    assert report["p95_latency_ms"] == pytest.approx(38.5)
    assert report["effective_fps"] == 40.0
    assert report["meets_scri014"]
    assert not report_performance([], 0.0, 0)["meets_scri014"]
    assert not report_performance([60.0], 1.0, 0)["meets_scri014"]


@pytest.fixture
def facial_pipeline() -> Generator[IntegratedFacialPipeline, None, None]:
    """Mô tả: detector MediaPipe thật; chỉ skip nếu thiếu model local.

    Tham số (Parameters): không; fixture riêng từng test để reset timestamp.
    Giá trị trả về (Returns): pipeline thật có finally teardown.
    Ngoại lệ (Raises): lỗi backend/model hỏng phải fail, không skip.
    """
    if not MODEL_PATH.is_file():
        pytest.skip(f"Thiếu model local: {MODEL_PATH}")
    pipeline = IntegratedFacialPipeline()
    try:
        yield pipeline
    finally:
        pipeline.release()


def test_pipeline_on_synthetic_blank_frame_safety(
    facial_pipeline: IntegratedFacialPipeline,
    blank_frame: np.ndarray,
) -> None:
    """Mô tả: kiểm chứng MediaPipe thật không báo nhắm mắt trên ảnh đen.

    Tham số (Parameters): pipeline thật, frame BGR đen.
    Giá trị trả về (Returns): None khi UNKNOWN cả hai và temporal rỗng.
    Ngoại lệ (Raises): AssertionError khi ảnh không mặt tạo cảnh báo.
    """
    result = facial_pipeline.process_frame(blank_frame, 0.0)
    assert result["landmark_valid"] is False
    assert_contract(result)
    facial_pipeline.release()
    facial_pipeline.release()


def test_pipeline_data_contracts_on_d3s_video(
    facial_pipeline: IntegratedFacialPipeline,
) -> None:
    """Mô tả: acquisition và mô hình thật trên 20 frame đầu video local.

    Tham số (Parameters): pipeline thật với tracker mới.
    Giá trị trả về (Returns): None nếu đủ 20 frame và có đo mặt hợp lệ.
    Ngoại lệ (Raises): AssertionError hợp đồng sai; skip nếu thiếu video.
    """
    if not SAMPLE_VIDEO.is_file():
        pytest.skip(f"Thiếu video D3S local: {SAMPLE_VIDEO}")
    tested = valid = 0
    with VideoReader(SAMPLE_VIDEO) as reader:
        for frame_id, timestamp, frame in islice(reader, 20):
            assert frame_id == tested
            result = facial_pipeline.process_frame(frame, timestamp)
            assert_contract(result)
            valid += int(result["landmark_valid"])
            tested += 1
    assert tested == 20
    assert valid > 0, "Không có số đo hợp lệ; không được pass do toàn UNKNOWN"


def main(argv: list[str] | None = None) -> int:
    """Mô tả: CLI demo hoặc headless benchmark, không chạy pytest ngầm.

    Tham số (Parameters): argv tùy chọn, mặc định đọc sys.argv.
    Giá trị trả về (Returns): 0 thành công; 1 lỗi vận hành rõ ràng.
    Ngoại lệ (Raises): argparse SystemExit nếu cú pháp CLI không hợp lệ.
    """
    parser = argparse.ArgumentParser(description="T1001 facial analysis PoC")
    parser.add_argument(
        "--source",
        default=str(SAMPLE_VIDEO),
        help="Video repo-relative/absolute hoặc ID camera (0)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--visual", action="store_true", dest="visual")
    modes.add_argument("--no-visual", action="store_false", dest="visual")
    parser.set_defaults(visual=False)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames phải là số nguyên dương")
    try:
        run_interactive_pipeline(args.source, args.max_frames, args.visual)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
