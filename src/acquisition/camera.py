"""Live-camera acquisition utilities for the drowsiness pipeline."""

from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
import numpy as np


class CameraStream:
    """Stream raw frames from an OpenCV camera device.

    The stream yields only acquisition data: ``(frame_id, timestamp, frame)``.
    Timestamps are elapsed seconds from stream start, derived from
    ``time.perf_counter()`` so they remain monotonic.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Open a camera device and optionally request a frame size.

        Args:
            device_id: OpenCV camera device index.
            width: Optional requested frame width in pixels.
            height: Optional requested frame height in pixels.

        Raises:
            ValueError: If OpenCV cannot open the requested camera device.
        """

        self.device_id = device_id
        self.width = width
        self.height = height
        self._capture = cv2.VideoCapture(device_id)

        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

        if not self._capture.isOpened():
            self.release()
            raise ValueError(f"Unable to open camera device: {device_id}")

    def __enter__(self) -> "CameraStream":
        """Return this stream while participating in a context manager."""

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Release the underlying camera capture when leaving a context."""

        del exc_type, exc_value, traceback
        self.release()

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield camera frames until interruption or disconnection."""

        if not self._capture.isOpened():
            raise ValueError("Cannot iterate over a released camera capture.")

        frame_id = 0
        start_time = time.perf_counter()
        try:
            while True:
                ret, frame = self._capture.read()
                if not ret or frame is None or frame.size == 0:
                    break

                timestamp = time.perf_counter() - start_time
                yield frame_id, float(timestamp), frame
                frame_id += 1
        finally:
            self.release()

    def release(self) -> None:
        """Release the underlying OpenCV capture if it is still open."""

        if self._capture.isOpened():
            self._capture.release()
