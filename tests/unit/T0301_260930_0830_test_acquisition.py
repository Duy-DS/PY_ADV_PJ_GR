"""Unit tests for T0301 input acquisition modules."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.camera import CameraStream  # noqa: E402
from src.acquisition.video_reader import VideoReader  # noqa: E402


def test_video_reader_reads_synthetic_video_with_metadata(
    tmp_path: Path,
) -> None:
    """VideoReader exposes properties and ordered raw-frame tuples."""

    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path, fps=5.0, frame_count=4)

    with VideoReader(video_path) as reader:
        assert reader.fps == pytest.approx(5.0)
        assert reader.width == 32
        assert reader.height == 24
        assert reader.total_frames == 4
        assert reader.duration == pytest.approx(0.8)

        frames = list(reader)

    assert [frame_id for frame_id, _, _ in frames] == [0, 1, 2, 3]
    assert [timestamp for _, timestamp, _ in frames] == pytest.approx(
        [0.0, 0.2, 0.4, 0.6]
    )
    assert all(isinstance(frame_id, int) for frame_id, _, _ in frames)
    assert all(isinstance(timestamp, float) for _, timestamp, _ in frames)
    assert all(isinstance(frame, np.ndarray) for _, _, frame in frames)
    assert all(frame.shape == (24, 32, 3) for _, _, frame in frames)


def test_video_reader_rejects_missing_video(tmp_path: Path) -> None:
    """A missing video path fails with an actionable file error."""

    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        VideoReader(tmp_path / "missing.mp4")


def test_video_reader_stops_at_eof_and_releases_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EOF must end iteration cleanly and release OpenCV resources."""

    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"placeholder")
    fake_capture = _FakeVideoCapture(
        frames=[
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.ones((2, 3, 3), dtype=np.uint8),
        ],
        fps=10.0,
        width=3,
        height=2,
    )
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda path: fake_capture,
    )

    reader = VideoReader(video_path)
    frames = list(reader)

    assert len(frames) == 2
    assert fake_capture.read_count == 3
    assert fake_capture.released is True


def test_video_reader_treats_empty_frame_as_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty frames are handled as EOF and never yielded downstream."""

    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"placeholder")
    fake_capture = _FakeVideoCapture(
        frames=[np.empty((0,), dtype=np.uint8)],
        fps=10.0,
        width=3,
        height=2,
    )
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda path: fake_capture,
    )

    reader = VideoReader(video_path)
    frames = list(reader)

    assert frames == []
    assert fake_capture.read_count == 1
    assert fake_capture.released is True


def test_video_reader_context_manager_releases_without_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The VideoReader context manager releases captures on exit."""

    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"placeholder")
    fake_capture = _FakeVideoCapture(
        frames=[np.zeros((2, 3, 3), dtype=np.uint8)],
        fps=15.0,
        width=3,
        height=2,
    )
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda path: fake_capture,
    )

    with VideoReader(video_path):
        assert fake_capture.released is False

    assert fake_capture.released is True


def test_camera_stream_yields_monotonic_elapsed_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CameraStream timestamps are elapsed seconds from perf_counter."""

    fake_capture = _FakeVideoCapture(
        frames=[
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.ones((4, 5, 3), dtype=np.uint8),
        ],
        fps=0.0,
        width=5,
        height=4,
    )
    perf_counter_values = iter([100.0, 100.0, 100.25])

    monkeypatch.setattr(cv2, "VideoCapture", lambda device_id: fake_capture)
    monkeypatch.setattr(
        "src.acquisition.camera.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    stream = CameraStream(device_id=1, width=640, height=480)
    frames = list(stream)

    assert [frame_id for frame_id, _, _ in frames] == [0, 1]
    assert [timestamp for _, timestamp, _ in frames] == pytest.approx(
        [0.0, 0.25]
    )
    assert fake_capture.set_calls == [
        (cv2.CAP_PROP_FRAME_WIDTH, 640.0),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480.0),
    ]
    assert fake_capture.released is True


def test_camera_stream_rejects_unavailable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable camera devices fail before streaming begins."""

    fake_capture = _FakeVideoCapture(
        frames=[],
        fps=0.0,
        width=0,
        height=0,
        opened=False,
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda device_id: fake_capture)

    with pytest.raises(ValueError, match="Unable to open camera device"):
        CameraStream(device_id=99)

    assert fake_capture.released is False


def test_camera_stream_context_manager_releases_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CameraStream context manager releases captures on exit."""

    fake_capture = _FakeVideoCapture(
        frames=[np.zeros((4, 5, 3), dtype=np.uint8)],
        fps=0.0,
        width=5,
        height=4,
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda device_id: fake_capture)

    with CameraStream(device_id=0):
        assert fake_capture.released is False

    assert fake_capture.released is True


class _FakeVideoCapture:
    """Small deterministic stand-in for cv2.VideoCapture."""

    def __init__(
        self,
        frames: list[np.ndarray],
        fps: float,
        width: int,
        height: int,
        opened: bool = True,
    ) -> None:
        self._frames = frames
        self._fps = fps
        self._width = width
        self._height = height
        self._opened = opened
        self._position = 0
        self.read_count = 0
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        """Return whether this fake capture is open."""

        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next frame or an EOF marker."""

        self.read_count += 1
        if self._position >= len(self._frames):
            return False, None

        frame = self._frames[self._position]
        self._position += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        """Return fake OpenCV capture properties."""

        values = {
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_WIDTH: float(self._width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self._height),
            cv2.CAP_PROP_FRAME_COUNT: float(len(self._frames)),
        }
        return values.get(prop_id, 0.0)

    def set(self, prop_id: int, value: float) -> bool:
        """Record property update requests."""

        self.set_calls.append((prop_id, value))
        return True

    def release(self) -> None:
        """Mark this fake capture as released."""

        self.released = True
        self._opened = False


def _write_test_video(
    path: Path,
    fps: float,
    frame_count: int,
) -> None:
    """Create a small AVI fixture readable by OpenCV."""

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot create AVI fixtures in this environment.")

    try:
        for index in range(frame_count):
            frame = np.full(
                (24, 32, 3),
                fill_value=index * 40,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()
