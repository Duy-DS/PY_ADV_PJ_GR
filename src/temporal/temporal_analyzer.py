"""
Module phân tích và xử lý chuỗi thời gian - Dung sai mất khung hình (Task T0703).
File: src/temporal/temporal_analyzer.py
"""

from typing import Dict, Any, Optional


class TemporalAnalyzer:
    def __init__(self, missing_frame_timeout: int = 5):
        """
        Khởi tạo bộ phân tích chuỗi thời gian với cơ chế dung sai (Tolerance).

        Args:
            missing_frame_timeout (int): Ngưỡng số frame tối đa cho phép mất dấu khuôn mặt
                                         trước khi reset toàn bộ trạng thái (Hold state limit).
        """
        self.missing_frame_timeout = missing_frame_timeout
        self.consecutive_missing_frames = 0

        # Trạng thái hợp lệ gần nhất để giữ lại (Hold state)
        self.last_valid_state = {
            "eye_state": "UNKNOWN",
            "mouth_state": "UNKNOWN",
            "eye_closure_duration": 0.0,
            "yawn_duration": 0.0,
            "drowsiness_status": "UNKNOWN",
        }

    def process_tolerance(
        self, face_detected: bool, current_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Cơ chế xử lý dung sai khi mất khung hình (Task T0703).
        Hàm này quyết định hệ thống sẽ TIẾP TỤC, HOLD hay RESET.
        """
        if face_detected and current_features is not None:
            # 1. Khuôn mặt hợp lệ: Reset bộ đếm mất dấu, lưu lại trạng thái mới nhất
            self.consecutive_missing_frames = 0
            self.last_valid_state.update(current_features)

            return {
                "action": "CONTINUE",
                "state": self.last_valid_state,
                "message": "Valid frame processed.",
            }

        else:
            # 2. Mất dấu khuôn mặt: Tăng bộ đếm
            self.consecutive_missing_frames += 1

            if self.consecutive_missing_frames <= self.missing_frame_timeout:
                # 2a. Mất dấu trong thời gian ngắn (<= Timeout): HOLD STATE
                return {
                    "action": "HOLD",
                    "state": self.last_valid_state,
                    "message": f"Face lost for {self.consecutive_missing_frames} frames. Holding state.",
                }
            else:
                # 2b. Mất dấu quá lâu (> Timeout): RESET TỚI UNKNOWN
                self.reset_states()
                return {
                    "action": "RESET",
                    "state": self.last_valid_state,
                    "message": f"Timeout reached ({self.consecutive_missing_frames} frames). State reset to UNKNOWN.",
                }

    def reset_states(self):
        """Reset toàn bộ bộ đếm thời gian và tính toán về trạng thái mặc định (UNKNOWN)."""
        self.last_valid_state = {
            "eye_state": "UNKNOWN",
            "mouth_state": "UNKNOWN",
            "eye_closure_duration": 0.0,
            "yawn_duration": 0.0,
            "drowsiness_status": "UNKNOWN",
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Trả về trạng thái hiện tại của hệ thống."""
        return self.last_valid_state
