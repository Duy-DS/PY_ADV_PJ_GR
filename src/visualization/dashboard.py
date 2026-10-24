"""
Module vẽ biểu đồ thời gian thực (Real-time Dashboard) - Sửa lỗi vẽ đường (Line Chart).
File: src/visualization/dashboard.py
"""

import cv2
import numpy as np
from collections import deque


class DashboardVisualizer:
    def __init__(
        self,
        frame_height: int,
        history_size: int = 100,
        ear_thresh: float = 0.2,
        mar_thresh: float = 0.6,
    ):
        self.width = 450
        self.height = frame_height
        self.history_size = history_size
        self.ear_thresh = ear_thresh
        self.mar_thresh = mar_thresh

        # Khởi tạo mảng rỗng để biểu đồ mọc dần từ lề phải
        self.ear_data = deque(maxlen=history_size)
        self.mar_data = deque(maxlen=history_size)

    def update(self, ear_val: float, mar_val: float, frame_id: int = 0):
        """Cập nhật dữ liệu mới."""
        self.ear_data.append(ear_val if ear_val is not None else 0.0)
        self.mar_data.append(mar_val if mar_val is not None else 0.0)

    def render(self) -> np.ndarray:
        """Vẽ biểu đồ và trả về hình ảnh mảng Numpy."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (20, 25, 30)

        mid_y = self.height // 2

        # Vẽ biểu đồ Mắt (Nửa trên)
        self._draw_graph(
            canvas,
            self.ear_data,
            "EAR (MAT)",
            self.ear_thresh,
            0,
            mid_y,
            (0, 255, 255),
            y_max=0.5,
            is_ear=True,
        )

        # Vẽ biểu đồ Miệng (Nửa dưới)
        self._draw_graph(
            canvas,
            self.mar_data,
            "MAR (MIENG)",
            self.mar_thresh,
            mid_y,
            self.height,
            (0, 165, 255),
            y_max=1.0,
            is_ear=False,
        )

        # Viền phân cách
        cv2.line(canvas, (0, mid_y), (self.width, mid_y), (50, 60, 70), 2)

        return canvas

    def _draw_graph(
        self, canvas, data, title, thresh, y_start, y_end, color, y_max, is_ear
    ):
        margin_bottom = 25
        plot_h = (y_end - y_start) - margin_bottom

        # 1. Vẽ Lưới ngang
        for i in range(1, 4):
            gy = y_start + int(i * plot_h / 4)
            cv2.line(canvas, (0, gy), (self.width, gy), (40, 45, 50), 1)

        # 2. Vẽ Lưới dọc (dịch chuyển dựa trên số điểm)
        x_step = self.width / max(1, self.history_size - 1)
        for i in range(0, self.history_size, 20):
            x = int(self.width - i * x_step)
            if x >= 0:
                cv2.line(
                    canvas, (x, y_start), (x, y_end - margin_bottom), (40, 45, 50), 1
                )

        # 3. Vẽ Đường Ngưỡng (Threshold - Đứt nét đỏ)
        thresh_y = y_end - margin_bottom - int((thresh / y_max) * plot_h)
        for x_line in range(0, self.width, 10):
            cv2.line(canvas, (x_line, thresh_y), (x_line + 5, thresh_y), (0, 0, 255), 1)

        num_points = len(data)
        if num_points == 0:
            return

        # 4. Tính toán tọa độ và VẼ ĐƯỜNG KẺ NỐI LIỀN (Line Chart)
        pts = []
        for i, val in enumerate(data):
            x_offset = (num_points - 1 - i) * x_step
            x = int(self.width - x_offset)

            val_clamped = min(max(val, 0.0), y_max)
            y = y_end - margin_bottom - int((val_clamped / y_max) * plot_h)
            pts.append((x, y))

        if num_points > 1:
            # FIX TẠI ĐÂY: Dùng vòng lặp nối từng điểm trước (i-1) và điểm sau (i)
            # Cách này đảm bảo đường thẳng luôn được vẽ mượt mà mà không phụ thuộc numpy/polylines
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 2, lineType=cv2.LINE_AA)
        else:
            # Vẽ một chấm tròn nếu mới chỉ có 1 khung hình
            cv2.circle(canvas, pts[0], 2, color, -1)

        # 5. Tiêu đề và Giá trị thời gian thực
        cv2.putText(
            canvas,
            title,
            (10, y_start + 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        current_val = data[-1]
        is_warning = (is_ear and current_val < thresh) or (
            not is_ear and current_val >= thresh
        )
        val_color = (0, 0, 255) if is_warning else (0, 255, 0)

        cv2.putText(
            canvas,
            f"{current_val:.3f}",
            (self.width - 60, y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            val_color,
            2,
        )
