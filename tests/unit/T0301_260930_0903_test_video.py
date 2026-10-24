"""Focused unit tests for the T0301 video-file reader."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.video_reader import VideoReader  # noqa: E402


def test_video_reader_emits_metadata_timestamps_and_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VideoReader exposes metadata and the required acquisition tuple."""

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"test video placeholder")
    capture = _FakeVideoCapture(
        frames=[
            np.zeros((24, 32, 3), dtype=np.uint8),
            np.ones((24, 32, 3), dtype=np.uint8),
        ],
        fps=5.0,
        width=32,
        height=24,
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)

    with VideoReader(video_path) as reader:
        frames = list(reader)

        assert reader.fps == pytest.approx(5.0)
        assert reader.width == 32
        assert reader.height == 24
        assert reader.total_frames == 2
        assert reader.duration == pytest.approx(0.4)

    assert [frame_id for frame_id, _, _ in frames] == [0, 1]
    assert [timestamp for _, timestamp, _ in frames] == pytest.approx(
        [0.0, 0.2]
    )
    assert all(isinstance(frame_id, int) for frame_id, _, _ in frames)
    assert all(isinstance(timestamp, float) for _, timestamp, _ in frames)
    assert all(isinstance(frame, np.ndarray) for _, _, frame in frames)
    assert capture.released is True


def test_video_reader_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing input path raises an actionable exception."""

    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        VideoReader(missing_path)


class _FakeVideoCapture:
    """Deterministic stand-in for a file-backed OpenCV capture."""

    def __init__(
        self,
        frames: list[np.ndarray],
        fps: float,
        width: int,
        height: int,
    ) -> None:
        self._frames = frames
        self._fps = fps
        self._width = width
        self._height = height
        self._position = 0
        self._opened = True
        self.released = False

    def isOpened(self) -> bool:
        """Return whether the fake capture remains open."""

        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next frame or an EOF marker."""

        if self._position >= len(self._frames):
            return False, None

        frame = self._frames[self._position]
        self._position += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        """Return configured OpenCV video metadata."""

        properties = {
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_WIDTH: float(self._width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self._height),
            cv2.CAP_PROP_FRAME_COUNT: float(len(self._frames)),
        }
        return properties.get(prop_id, 0.0)

    def release(self) -> None:
        """Record capture release."""

        self.released = True
        self._opened = False
