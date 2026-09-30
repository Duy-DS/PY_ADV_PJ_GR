"""Video-file acquisition utilities for the drowsiness pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

import cv2
import numpy as np


class VideoReader:
    """Read ordered frames and timestamps from a video file.

    The reader is intentionally limited to acquisition concerns. It validates
    that the file can be opened, exposes core technical properties, and yields
    raw BGR frames as ``(frame_id, timestamp, frame)`` tuples.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, video_path: Union[str, Path]) -> None:
        """Open ``video_path`` and extract OpenCV video properties.

        Args:
            video_path: Absolute path or repository-relative path to a video.

        Raises:
            FileNotFoundError: If the requested video path does not exist.
            ValueError: If OpenCV cannot open the file or reports invalid
                properties needed for timestamp generation.
        """

        self.video_path = self._resolve_video_path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {self.video_path}"
            )

        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps <= 0.0:
            self.release()
            raise ValueError(
                f"Video has invalid FPS ({self.fps}): {self.video_path}"
            )

        self.duration = (
            self.total_frames / self.fps if self.total_frames > 0 else 0.0
        )

    def __enter__(self) -> "VideoReader":
        """Return this reader while participating in a context manager."""

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Release the underlying video capture when leaving a context."""

        del exc_type, exc_value, traceback
        self.release()

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield ``(frame_id, timestamp, raw_frame)`` tuples until EOF."""

        if not self._capture.isOpened():
            raise ValueError("Cannot iterate over a released video capture.")

        frame_id = 0
        try:
            while True:
                ret, frame = self._capture.read()
                if not ret or frame is None or frame.size == 0:
                    break

                timestamp = frame_id / self.fps
                yield frame_id, float(timestamp), frame
                frame_id += 1
        finally:
            self.release()

    def release(self) -> None:
        """Release the underlying OpenCV capture if it is still open."""

        if self._capture.isOpened():
            self._capture.release()

    @classmethod
    def _resolve_video_path(cls, video_path: Union[str, Path]) -> Path:
        """Resolve absolute or repository-relative video paths."""

        path = Path(video_path)
        if not path.is_absolute():
            path = cls._REPO_ROOT / path
        return path.resolve()
