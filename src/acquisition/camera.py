"""Tiện ích thu thập dữ liệu từ camera trực tiếp cho luồng phát hiện buồn ngủ."""

from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
import numpy as np


class CameraStream:
    """Phát (stream) các khung hình thô từ một thiết bị camera OpenCV.

    Luồng này chỉ trả về dữ liệu thu thập: ``(frame_id, timestamp, frame)``.
    Các mốc thời gian là số giây đã trôi qua kể từ khi luồng bắt đầu, được lấy từ
    ``time.perf_counter()`` để đảm bảo chúng luôn tăng dần (monotonic).
    """

    def __init__(
        self,
        device_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Mở một thiết bị camera và tùy ý yêu cầu một kích thước khung hình.

        Tham số:
            device_id: Chỉ số thiết bị camera OpenCV.
            width: Chiều rộng khung hình yêu cầu tùy chọn (tính bằng pixel).
            height: Chiều cao khung hình yêu cầu tùy chọn (tính bằng pixel).

        Ngoại lệ:
            ValueError: Nếu OpenCV không thể mở thiết bị camera được yêu cầu.
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
            raise ValueError(f"Không thể mở thiết bị camera: {device_id}")

    def __enter__(self) -> "CameraStream":
        """Trả về luồng này khi tham gia vào một context manager (quản lý ngữ cảnh)."""

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Giải phóng bộ thu hình (capture) camera nền tảng khi thoát khỏi context."""

        del exc_type, exc_value, traceback
        self.release()

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Trả về (yield) các khung hình camera cho đến khi bị gián đoạn hoặc ngắt kết nối."""

        if not self._capture.isOpened():
            raise ValueError(
                "Không thể lặp (iterate) qua một bộ thu hình camera đã bị giải phóng."
            )

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
        """Giải phóng bộ thu hình OpenCV nền tảng nếu nó vẫn đang mở."""

        if self._capture.isOpened():
            self._capture.release()
