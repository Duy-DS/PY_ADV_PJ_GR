"""T1002: pipeline thật với nguồn/mốc tổng hợp, CLI và teardown."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import main as cli  # noqa: E402
from src.alert.alert_manager import AlertState  # noqa: E402
from src.pipeline import realtime_pipeline as module  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def eye_points(closed: bool = False) -> list[tuple[float, float]]:
    """Tạo hình mắt; Parameters: closed; Returns: sáu điểm; Raises: không."""
    height = 0.1 if closed else 2.0
    return [
        (0, 0),
        (1, height),
        (3, height),
        (4, 0),
        (3, -height),
        (1, -height),
    ]


def mouth_points(opened: bool = False) -> list[tuple[float, float]]:
    """Tạo miệng; Parameters: opened; Returns: tám điểm; Raises: không."""
    height = 2.0 if opened else 0.1
    return [
        (0, 0),
        (1, height),
        (2, height),
        (3, height),
        (4, 0),
        (3, -height),
        (2, -height),
        (1, -height),
    ]


class Stream:
    """
    Nguồn giả; Parameters: mốc thời gian/lỗi; Returns: nguồn; Raises: không.
    """

    def __init__(self, times: list[float], error: Exception | None = None):
        """Lưu mốc; Parameters: times/error; Returns: không; Raises: không."""
        self.times = times
        self.error = error
        self.release = Mock()
        self.read_count = 0

    def __iter__(self):
        """Đọc; Parameters: không; Returns: iterator; Raises: lỗi được gán."""
        for index, timestamp in enumerate(self.times):
            self.read_count += 1
            yield index, timestamp, np.zeros((240, 320, 3), dtype=np.uint8)
        if self.error is not None:
            raise self.error


@pytest.fixture
def windows(monkeypatch):
    """Chặn GUI; Parameters: monkeypatch; Returns: mock; Raises: không."""
    destroy = Mock()
    monkeypatch.setattr(module.cv2, "destroyAllWindows", destroy)
    monkeypatch.setattr(module.cv2, "imshow", Mock())
    monkeypatch.setattr(module.cv2, "waitKey", Mock(return_value=-1))
    return destroy


def build(
    times: list[float],
    mode: str = "eye_yawn",
    *,
    closed: list[bool] | None = None,
    opened: list[bool] | None = None,
    missing: bool = False,
    corrupt: bool = False,
    config: Path | None = None,
    error: Exception | None = None,
):
    """
    Lắp stage thật; Parameters: kịch bản; Returns: runner; Raises: config lỗi.
    """
    count = len(times)
    closed = closed or [False] * count
    opened = opened or [False] * count
    face = Mock()
    face.process.return_value = {
        "face_detected": not missing,
        "bbox": (10, 10, 80, 80),
    }
    mesh = Mock()
    mesh.process.side_effect = [
        {
            "landmark_valid": True,
            "left_eye_points": eye_points(c),
            "right_eye_points": eye_points(c),
            "mouth_points": None if corrupt else mouth_points(o),
        }
        for c, o in zip(closed, opened)
    ]
    return module.RealtimePipeline(
        config,
        mode,
        reader=Stream(times, error),
        face_detector=face,
        landmark_detector=mesh,
    )


@pytest.mark.parametrize("mode", ["eye_only", "eye_yawn"])
def test_prolonged_closure_and_recovery(mode, windows):
    """Vòng cảnh báo; Parameters: mode; Returns: không; Raises: assertion."""
    runner = build(
        [0, 0.5, 1.5, 2, 2.5, 5.5],
        mode,
        closed=[True, True, True, False, False, False],
    )
    snapshots = []
    hud = Mock(wraps=runner.visualizer.draw_hud)
    runner.visualizer.draw_hud = hud
    summary = runner.run(display=False, on_frame=snapshots.append)
    assert snapshots[0].decision["state"] == "NON_DROWSY"
    assert snapshots[2].decision["state"] == "DROWSY"
    assert snapshots[2].alert == AlertState.WARNING
    assert snapshots[4].alert == AlertState.COOLDOWN
    assert snapshots[5].alert == AlertState.NORMAL
    assert summary.total_frames == 6
    assert summary.drowsiness_frames >= 1
    assert summary.alert_count == 1
    assert summary.average_fps > 0
    assert summary.average_latency_ms > 0
    assert hud.call_count == 6
    assert hud.call_args.kwargs["fps"] > 0
    assert hud.call_args.kwargs["latency_ms"] > 0
    assert runner.alert_manager.state == AlertState.NORMAL
    assert runner.temporal.eye_tracker.consecutive_closed_frames == 0
    runner.reader.release.assert_called_once()
    runner.face_detector.release.assert_called_once()
    runner.landmark_detector.release.assert_called_once()
    windows.assert_called_once()
    module.cv2.imshow.assert_not_called()
    module.cv2.waitKey.assert_not_called()


@pytest.mark.parametrize("mode", ["eye_only", "eye_yawn"])
@pytest.mark.parametrize("invalid", ["missing", "corrupt"])
def test_invalid_is_unknown(mode, invalid, windows):
    """
    Không suy diễn; Parameters: mode/invalid; Returns: không; Raises:
    assertion.
    """
    runner = build(
        [0, 2, 4],
        mode,
        closed=[True] * 3,
        missing=invalid == "missing",
        corrupt=invalid == "corrupt",
    )
    frames = []
    summary = runner.run(display=False, on_frame=frames.append)
    assert summary.unknown_frames == 3
    assert summary.drowsiness_frames == summary.alert_count == 0
    for frame in frames:
        assert frame.decision["state"] == "UNKNOWN"
        assert frame.temporal.eye.eye_state == "UNKNOWN"
        assert frame.temporal.mouth.mouth_state == "UNKNOWN"
    if invalid == "missing":
        runner.landmark_detector.process.assert_not_called()


def test_single_blink_is_not_drowsy(windows):
    """
    Frame đơn an toàn; Parameters: mock; Returns: không; Raises: assertion.
    """
    runner = build([0, 0.1, 2.0], closed=[False, True, False])
    summary = runner.run(display=False)
    assert summary.drowsiness_frames == summary.alert_count == 0


def test_missing_face_after_warning_is_unknown(windows):
    """Giữ an toàn sau cảnh báo.

    Parameters: windows chặn GUI.
    Returns: Không có.
    Raises: AssertionError nếu gap bị diễn giải thành DROWSY.
    """
    runner = build([0, 1.5, 1.6], closed=[True] * 3)
    runner.face_detector.process.side_effect = [
        {"face_detected": True, "bbox": None},
        {"face_detected": True, "bbox": None},
        {"face_detected": False, "bbox": None},
    ]
    frames = []
    result = runner.run(display=False, on_frame=frames.append)
    assert frames[1].decision["state"] == "DROWSY"
    assert frames[2].decision["state"] == "UNKNOWN"
    assert frames[2].temporal.eye.eye_state == "UNKNOWN"
    assert result.unknown_frames == 1


def test_display_keyboard_interrupt(windows):
    """Ngắt GUI vẫn teardown.

    Parameters: windows chặn GUI.
    Returns: Không có.
    Raises: AssertionError nếu ngắt không được xử lý.
    """
    module.cv2.waitKey.side_effect = KeyboardInterrupt()
    runner = build([0, 1])
    assert runner.run(display=True).interrupted
    runner.reader.release.assert_called_once()
    windows.assert_called_once()


@pytest.mark.parametrize("content", ["[]", "eye: [", "eye: {}"])
def test_invalid_yaml_stops_before_acquisition(content, tmp_path, monkeypatch):
    """Cấu hình lỗi không mở camera.

    Parameters: content YAML lỗi, tmp_path và monkeypatch.
    Returns: Không có.
    Raises: AssertionError nếu nguồn bị mở.
    """
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    camera = Mock()
    monkeypatch.setattr(module, "CameraStream", camera)
    assert cli.main(["--source", "0", "--config", str(path)]) == 1
    camera.assert_not_called()


def test_empty_stream_and_single_use(windows):
    """Nguồn rỗng và lifecycle rõ ràng.

    Parameters: windows chặn GUI.
    Returns: Không có.
    Raises: AssertionError nếu tổng kết hoặc lifecycle sai.
    """
    runner = build([])
    summary = runner.run(display=False)
    assert summary.total_frames == summary.average_latency_ms == 0
    with pytest.raises(ValueError, match="single-use"):
        runner.run(display=False)
    windows.assert_called_once()


@pytest.mark.parametrize("mode,expected", [("eye_only", 0), ("eye_yawn", 1)])
def test_repeated_yawns(mode, expected, windows):
    """
    So sánh A/B; Parameters: mode/expected; Returns: không; Raises: assertion.
    """
    runner = build(
        [0, 2, 3, 4, 6, 7, 8, 10],
        mode,
        opened=[True, True, False, True, True, False, True, True],
    )
    frames = []
    summary = runner.run(display=False, on_frame=frames.append)
    assert frames[-1].temporal.behavior.yawn_count == 3
    assert summary.drowsiness_frames == expected


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_display_exit(key, windows):
    """Thoát GUI; Parameters: key; Returns: không; Raises: assertion."""
    module.cv2.waitKey.return_value = key
    runner = build([0, 1, 2])
    result = runner.run(display=True)
    assert result.total_frames == 1 and result.interrupted
    runner.reader.release.assert_called_once()
    windows.assert_called_once()


@pytest.mark.parametrize(
    "error", [KeyboardInterrupt(), RuntimeError("decoder")]
)
def test_interrupt_and_error_cleanup(error, windows):
    """Đóng khi lỗi; Parameters: error; Returns: không; Raises: assertion."""
    runner = build([0], error=error)
    if isinstance(error, KeyboardInterrupt):
        assert runner.run(display=False).interrupted
    else:
        with pytest.raises(RuntimeError, match="decoder"):
            runner.run(display=False)
    runner.reader.release.assert_called_once()
    windows.assert_called_once()
    assert runner.alert_manager.state == AlertState.NORMAL


def test_partial_initialization_cleanup(monkeypatch, windows):
    """Lỗi mở mesh; Parameters: mock; Returns: không; Raises: assertion."""
    reader, face = Stream([]), Mock()
    monkeypatch.setattr(module, "VideoReader", Mock(return_value=reader))
    monkeypatch.setattr(module, "FaceDetector", Mock(return_value=face))
    monkeypatch.setattr(
        module,
        "LandmarkDetector",
        Mock(side_effect=RuntimeError("model missing")),
    )
    runner = module.RealtimePipeline()
    with pytest.raises(RuntimeError, match="model missing"):
        runner.run("test.mp4", display=False)
    reader.release.assert_called_once()
    face.release.assert_called_once()
    windows.assert_called_once()


def test_cleanup_failure_does_not_skip_other_resources(windows):
    """
    Teardown độc lập; Parameters: mock; Returns: không; Raises: assertion.
    """
    runner = build([0])
    runner.reader.release.side_effect = RuntimeError("release")
    with pytest.raises(RuntimeError, match="cleanup"):
        runner.run(display=False)
    runner.face_detector.release.assert_called_once()
    runner.landmark_detector.release.assert_called_once()
    windows.assert_called_once()


def test_frame_limit_and_custom_config(tmp_path, windows):
    """YAML chung; Parameters: tmp_path; Returns: không; Raises: assertion."""
    config = load_config()
    config["eye"]["closure_time_threshold"] = 0.2
    config["mouth"]["yawn_frequency_threshold"] = 1
    config["alert"]["cooldown"] = 0.1
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runner = build([0, 0.3, 1], closed=[True] * 3, config=path)
    result = runner.run(display=False, max_frames=2)
    assert result.total_frames == runner.reader.read_count == 2
    assert result.drowsiness_frames == 1
    assert runner.engine.yawn_frequency_threshold == 1
    assert runner.alert_manager.cooldown == 0.1


@pytest.mark.parametrize("source,device", [("webcam", 0), ("0", 0), ("2", 2)])
def test_camera_source_selection(source, device, monkeypatch, windows):
    """
    Chọn camera; Parameters: source/device; Returns: không; Raises: assertion.
    """
    reader = Stream([])
    camera = Mock(return_value=reader)
    monkeypatch.setattr(module, "CameraStream", camera)
    runner = build([])
    runner.reader = None
    assert runner.run(source, display=False).total_frames == 0
    camera.assert_called_once_with(device)
    reader.release.assert_called_once()


@pytest.mark.parametrize("flag", ["--headless", "--no-visual"])
def test_cli_mock_execution(flag, monkeypatch, capsys, windows):
    """
    CLI xuyên runner; Parameters: flag; Returns: không; Raises: assertion.
    """
    monkeypatch.setattr(module, "VideoReader", lambda source: Stream([0]))
    face, mesh = Mock(), Mock()
    face.process.return_value = {"face_detected": False, "bbox": None}
    monkeypatch.setattr(module, "FaceDetector", Mock(return_value=face))
    monkeypatch.setattr(module, "LandmarkDetector", Mock(return_value=mesh))
    assert (
        cli.main(
            [
                "--source",
                "mock.mp4",
                flag,
                "--mode",
                "eye_only",
                "--max-frames",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["unknown_frames"] == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--source", "missing.mp4", "--headless"],
        ["--source", "0", "--config", "missing.yaml", "--headless"],
    ],
)
def test_cli_actionable_errors(arguments, caplog, windows):
    """
    Lỗi rõ ràng; Parameters: arguments; Returns: không; Raises: assertion.
    """
    assert cli.main(arguments) == 1
    assert "Check --source, --config and model assets" in caplog.text


@pytest.mark.parametrize(
    "arguments",
    [
        ["--source", "0", "--max-frames", "0"],
        ["--source", "0", "--mode", "invalid"],
    ],
)
def test_cli_invalid_arguments(arguments):
    """
    Từ chối CLI sai; Parameters: arguments; Returns: không; Raises: assertion.
    """
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2


def test_cli_help_outside_repository(tmp_path):
    """
    Đường dẫn độc lập; Parameters: tmp_path; Returns: không; Raises: assertion.
    """
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/main.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--headless" in result.stdout


@pytest.mark.parametrize("mode", ["eye_only", "eye_yawn"])
def test_real_video_cli(mode):
    """Smoke D3S thật; Parameters: mode; Returns: không; Raises: assertion."""
    video = ROOT / "data/raw/d3s/d3s_video_dataset.mp4"
    assets = [
        video,
        ROOT / "models/pretrained/face_landmarker.task",
        ROOT / "models/pretrained/blaze_face_short_range.tflite",
    ]
    if not all(path.is_file() for path in assets):
        pytest.skip("Local D3S video/model assets unavailable")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/main.py"),
            "--source",
            str(video),
            "--mode",
            mode,
            "--headless",
            "--max-frames",
            "15",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["total_frames"] == 15
    assert summary["average_latency_ms"] > 0
