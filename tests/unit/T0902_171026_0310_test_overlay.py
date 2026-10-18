"""Script kiểm thử độc lập giao diện HUD (Task T0902)."""

import sys
from pathlib import Path
import cv2
import numpy as np

# Đảm bảo import được các module trong src
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.overlay import HUDVisualizer  # noqa: E402
from src.utils.contracts import (  # noqa: E402
    EyeMeasurement,
    MouthMeasurement,
    TemporalBehaviorState,
    TemporalAnalysisResult,
)


def test_hud_visualization(visual=False):
    # 1. Tạo một khung hình giả (ảnh đen 640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Tạo nền xám nhẹ để dễ nhìn chữ màu trắng/vàng
    frame[:] = (50, 50, 50)

    # 2. Khởi tạo các dữ liệu giả (Mock Data)
    visualizer = HUDVisualizer()

    bbox = (200, 100, 150, 200)  # (x, y, w, h)
    landmarks = {
        "left_eye_points": [(230, 150), (250, 150)],
        "right_eye_points": [(300, 150), (320, 150)],
        "mouth_points": [(250, 220), (300, 220), (275, 250)],
    }

    # Tạo dữ liệu TemporalAnalysisResult theo đúng hợp đồng (contracts.py)
    eye = EyeMeasurement(
        left_ear=0.25,
        right_ear=0.25,
        avg_ear=0.25,
        eye_state="OPEN",
        is_valid=True,
    )
    mouth = MouthMeasurement(
        mar=0.15, mouth_state="MOUTH_CLOSED", is_valid=True
    )
    behavior = TemporalBehaviorState(
        eye_closed_duration=0.5, is_prolonged_closure=False
    )

    temporal_result = TemporalAnalysisResult(
        eye=eye,
        mouth=mouth,
        behavior=behavior,
        action="CONTINUE",
        is_valid=True,
        consecutive_missing_frames=0,
    )

    # 3. Chạy thử 3 trạng thái cảnh báo khác nhau
    test_states = [
        ("NON_DROWSY", "Test 1: NORMAL (Green) - An phim bat ky de tiep tuc"),
        ("WARNING", "Test 2: WARNING (Red) - An phim bat ky de tiep tuc"),
        ("UNKNOWN", "Test 3: UNKNOWN (Yellow) - An phim bat ky de thoat"),
    ]

    for state, title in test_states:
        test_frame = frame.copy()

        # Vẽ HUD
        test_frame = visualizer.draw_hud(
            frame=test_frame,
            temporal_result=temporal_result,
            decision_state=state,
            fps=29.9,
            latency_ms=15.4,
            bbox=bbox,
            landmarks=landmarks,
        )

        assert test_frame.shape == frame.shape
        assert np.any(test_frame != frame)
        if visual:
            cv2.imshow("HUD Test", test_frame)
            print(title)
            cv2.waitKey(0)

    if visual:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_hud_visualization(visual=True)
