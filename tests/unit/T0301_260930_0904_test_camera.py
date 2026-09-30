"""Focused unit tests for the T0301 camera stream."""

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


def test_camera_stream_emits_monotonic_timestamps_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CameraStream emits typed tuples and releases after disconnection."""

    capture = _FakeCameraCapture(
        frames=[
            np.zeros((24, 32, 3), dtype=np.uint8),
            np.ones((24, 32, 3), dtype=np.uint8),
        ]
    )
    counter_values = iter([10.0, 10.0, 10.25])
    monkeypatch.setattr(cv2, "VideoCapture", lambda device_id: capture)
    monkeypatch.setattr(
        "src.acquisition.camera.time.perf_counter",
        lambda: next(counter_values),
    )

    stream = CameraStream(device_id=0, width=640, height=480)
    frames = list(stream)

    assert [frame_id for frame_id, _, _ in frames] == [0, 1]
    assert [timestamp for _, timestamp, _ in frames] == pytest.approx(
        [0.0, 0.25]
    )
    assert all(isinstance(frame_id, int) for frame_id, _, _ in frames)
    assert all(isinstance(timestamp, float) for _, timestamp, _ in frames)
    assert all(isinstance(frame, np.ndarray) for _, _, frame in frames)
    assert capture.set_calls == [
        (cv2.CAP_PROP_FRAME_WIDTH, 640.0),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480.0),
    ]
    assert capture.released is True


def test_camera_stream_rejects_an_unavailable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable camera raises an actionable exception."""

    capture = _FakeCameraCapture(frames=[], opened=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda device_id: capture)

    with pytest.raises(ValueError, match="Unable to open camera device: 99"):
        CameraStream(device_id=99)


class _FakeCameraCapture:
    """Deterministic stand-in for a live OpenCV capture."""

    def __init__(
        self,
        frames: list[np.ndarray],
        opened: bool = True,
    ) -> None:
        self._frames = frames
        self._opened = opened
        self._position = 0
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        """Return whether the fake camera remains open."""

        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next frame or a disconnection marker."""

        if self._position >= len(self._frames):
            return False, None

        frame = self._frames[self._position]
        self._position += 1
        return True, frame

    def set(self, prop_id: int, value: float) -> bool:
        """Record requested capture dimensions."""

        self.set_calls.append((prop_id, value))
        return True

    def release(self) -> None:
        """Record camera release."""

        self.released = True
        self._opened = False
