"""
Module hiển thị giao diện HUD và lớp phủ trực quan hóa thời gian thực (Task
T0902).
File: src/visualization/overlay.py
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
from src.utils.contracts import TemporalAnalysisResult


class HUDVisualizer:
    """
    Lớp tiện ích để vẽ HUD và các lớp phủ trực quan lên khung hình camera.
    """

    def __init__(self):
        # Định nghĩa bảng màu (BGR)
        self.COLOR_GREEN = (0, 255, 0)  # NON_DROWSY
        self.COLOR_RED = (0, 0, 255)  # DROWSY
        self.COLOR_YELLOW = (0, 255, 255)  # UNKNOWN
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_BLUE = (255, 0, 0)

    def _get_status_color(self, state: str) -> Tuple[int, int, int]:
        """Ánh xạ trạng thái sang màu sắc tương ứng theo cấu hình DoD."""
        state_upper = state.upper() if state else "UNKNOWN"
        if "NON_DROWSY" in state_upper or "NORMAL" in state_upper:
            return self.COLOR_GREEN
        elif "DROWSY" in state_upper or "WARNING" in state_upper:
            return self.COLOR_RED
        else:
            return self.COLOR_YELLOW

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        color: Tuple[int, int, int],
    ):
        """Vẽ bounding box bao quanh khuôn mặt."""
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Optional[Dict[str, List[Tuple[int, int]]]],
    ):
        """Vẽ các điểm mốc (landmarks) của mắt trái, mắt phải và miệng."""
        if not landmarks:
            return

        # Vẽ mắt bằng màu Xanh lá
        for pt in landmarks.get("left_eye_points", []):
            cv2.circle(
                frame, (int(pt[0]), int(pt[1])), 2, self.COLOR_GREEN, -1
            )
        for pt in landmarks.get("right_eye_points", []):
            cv2.circle(
                frame, (int(pt[0]), int(pt[1])), 2, self.COLOR_GREEN, -1
            )

        # Vẽ miệng bằng màu Vàng hoặc Xanh dương để dễ phân biệt
        for pt in landmarks.get("mouth_points", []):
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, self.COLOR_BLUE, -1)

    def draw_hud(
        self,
        frame: np.ndarray,
        temporal_result: Optional[TemporalAnalysisResult],
        decision_state: str,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ) -> np.ndarray:
        """
        Vẽ toàn bộ HUD lên khung hình bao gồm thông số kỹ thuật, số liệu
        EAR/MAR và Banner trạng thái.
        Không làm suy giảm FPS nhờ sử dụng các hàm vẽ cơ sở được tối ưu của
        OpenCV.
        """
        # 1. Xác định màu chủ đạo dựa trên quyết định/trạng thái hiện tại
        status_color = self._get_status_color(decision_state)

        # 2. Vẽ Bounding Box & Landmarks
        self.draw_bounding_box(frame, bbox, status_color)
        self.draw_landmarks(frame, landmarks)

        # 3. Góc trên bên trái: Chỉ số kỹ thuật hệ thống (FPS, Độ trễ)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLOR_YELLOW,
            2,
        )

        # Trích xuất dữ liệu từ TemporalAnalysisResult
        ear_val = 0.0
        mar_val = 0.0
        closed_duration = 0.0

        if temporal_result and temporal_result.is_valid:
            ear_val = temporal_result.eye.avg_ear
            mar_val = temporal_result.mouth.mar
            closed_duration = temporal_result.behavior.eye_closed_duration
        # Chuyển xuống phía dưới một chút để chừa không gian cho khuôn mặt

        # 4. Góc dưới bên trái: Các chỉ số toán học
        y_offset = 70
        cv2.putText(
            frame,
            f"EAR: {ear_val:.3f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLOR_WHITE,
            2,
        )
        cv2.putText(
            frame,
            f"MAR: {mar_val:.3f}",
            (10, y_offset + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLOR_WHITE,
            2,
        )
        cv2.putText(
            frame,
            f"Eye Closed: {closed_duration:.2f}s",
            (10, y_offset + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLOR_WHITE,
            2,
        )

        # 5. Đáy màn hình: Banner trạng thái lớn
        # Tạo lớp overlay đen mờ dưới đáy màn hình để text nổi bật
        overlay = frame.copy()
        banner_height = 50
        cv2.rectangle(
            overlay,
            (0, frame.shape[0] - banner_height),
            (frame.shape[1], frame.shape[0]),
            self.COLOR_BLACK,
            -1,
        )
        cv2.addWeighted(
            overlay, 0.6, frame, 0.4, 0, frame
        )  # Alpha blending nhẹ tránh giật lag

        # Căn giữa chữ cảnh báo
        banner_text = f"STATE: {decision_state}"
        text_size = cv2.getTextSize(
            banner_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2
        )[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 15

        cv2.putText(
            frame,
            banner_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            status_color,
            2,
        )

        return frame
