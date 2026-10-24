"""Tiện ích đọc file video cho luồng phát hiện buồn ngủ."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

import cv2
import numpy as np


class VideoReader:
    """Đọc các khung hình và mốc thời gian theo thứ tự từ một file video.

    Trình đọc này được cố tình giới hạn ở các chức năng thu thập dữ liệu. Nó xác thực
    rằng file có thể mở được, hiển thị các thuộc tính kỹ thuật cốt lõi và trả về
    các khung hình BGR thô dưới dạng tuple ``(frame_id, timestamp, frame)``.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, video_path: Union[str, Path]) -> None:
        """Mở ``video_path`` và trích xuất các thuộc tính video của OpenCV.

        Tham số:
            video_path: Đường dẫn tuyệt đối hoặc đường dẫn tương đối từ kho lưu trữ đến một video.

        Ngoại lệ:
            FileNotFoundError: Nếu đường dẫn video yêu cầu không tồn tại.
            ValueError: Nếu OpenCV không thể mở file hoặc báo cáo các thuộc tính
                không hợp lệ cần thiết để tạo mốc thời gian.
        """

        self.video_path = self._resolve_video_path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"File video không tồn tại: {self.video_path}")

        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"Không thể mở file video: {self.video_path}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps <= 0.0:
            self.release()
            raise ValueError(
                f"Video có FPS không hợp lệ ({self.fps}): {self.video_path}"
            )

        self.duration = self.total_frames / self.fps if self.total_frames > 0 else 0.0

    def __enter__(self) -> "VideoReader":
        """Trả về trình đọc này khi tham gia vào một context manager."""

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Giải phóng bộ thu hình video nền tảng khi thoát khỏi context."""

        del exc_type, exc_value, traceback
        self.release()

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Trả về (yield) các tuple ``(frame_id, timestamp, raw_frame)`` cho đến khi kết thúc file (EOF)."""

        if not self._capture.isOpened():
            raise ValueError(
                "Không thể lặp (iterate) qua một bộ thu hình video đã bị giải phóng."
            )

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
        """Giải phóng bộ thu hình OpenCV nền tảng nếu nó vẫn đang mở."""

        if self._capture.isOpened():
            self._capture.release()

    @classmethod
    def _resolve_video_path(cls, video_path: Union[str, Path]) -> Path:
        """Phân giải các đường dẫn video tuyệt đối hoặc tương đối từ thư mục dự án."""

        path = Path(video_path)
        if not path.is_absolute():
            path = cls._REPO_ROOT / path
        return path.resolve()
