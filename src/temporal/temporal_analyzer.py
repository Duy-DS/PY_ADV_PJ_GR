"""Coordinator T0704: hành vi typed, không thực hiện quyết định buồn ngủ.

Gap tường minh được loại khỏi timer, còn cửa sổ ngáp dùng thời gian video.
Không thể nhận biết decoder bỏ frame nếu caller không gửi quan sát UNKNOWN.
"""

import math

from src.temporal.eye_closure import EyeClosureTracker
from src.temporal.yawning_tracker import YawningTracker
from src.utils.config import load_temporal_settings, validate_timestamp
from src.utils.contracts import (
    EyeMeasurement,
    MouthMeasurement,
    TemporalAnalysisResult,
    TemporalBehaviorState,
    unknown_eye,
    unknown_mouth,
)


class TemporalAnalyzer:
    """Kết hợp hai tracker với chính sách invalidation toàn frame.

    Parameters: config_path là YAML hoặc None cho mặc định repository.
    Returns: Coordinator không sở hữu capture, window hay log.
    Raises: OSError, ValueError nếu cấu hình không hợp lệ.
    """

    def __init__(self, config_path=None):
        """Nạp cùng một cấu hình cho cả hai nhánh.

        Parameters: config_path là YAML hoặc None.
        Returns: Không có.
        Raises: OSError, ValueError nếu cấu hình sai.
        """
        self.settings = load_temporal_settings(config_path)
        self.eye_tracker = EyeClosureTracker(self.settings)
        self.yawn_tracker = YawningTracker(self.settings)
        self.reset()

    def reset(self):
        """Xóa toàn bộ trạng thái; video kế tiếp có thể bắt đầu tại zero.

        Parameters: Không có.
        Returns: Không có.
        Raises: Không có.
        """
        self.eye_tracker.reset()
        self.yawn_tracker.reset()
        self.consecutive_missing_frames = 0
        self._last_timestamp = None
        self._last_valid_timestamp = None
        self._gap_expired = False
        self._behavior = TemporalBehaviorState()

    def _clear_behavior(self):
        """Reset buffers khi gap hết hạn, giữ metadata của luồng hiện tại.

        Parameters: Không có.
        Returns: Không có.
        Raises: Không có.
        """
        self.eye_tracker.reset()
        self.yawn_tracker.reset()
        self._behavior = TemporalBehaviorState()
        self._gap_expired = True

    def _timed_out(self, timestamp):
        """Kiểm tra biên timeout tính từ quan sát hợp lệ cuối.

        Parameters: timestamp đã kiểm tra.
        Returns: bool; tại đúng timeout phải reset.
        Raises: Không có.
        """
        return self._last_valid_timestamp is None or (
            timestamp - self._last_valid_timestamp
            >= self.settings.missing_face_timeout_sec
        )

    def update(
        self,
        face_detected: bool,
        timestamp: float,
        eye: EyeMeasurement,
        mouth: MouthMeasurement,
    ) -> TemporalAnalysisResult:
        """Xử lý một frame; trạng thái hiện tại UNKNOWN ngay khi invalid.

        Parameters: face_detected, timestamp, eye và mouth typed.
        Returns: Envelope typed; HOLD giữ counter nhưng tắt cờ sự kiện.
        Raises: TypeError nếu sai hợp đồng; ValueError nếu thời gian sai.
        """
        timestamp = validate_timestamp(timestamp, self._last_timestamp)
        if not isinstance(face_detected, bool):
            raise TypeError("face_detected must be bool")
        if not isinstance(eye, EyeMeasurement) or not isinstance(
            mouth, MouthMeasurement
        ):
            raise TypeError("Expected EyeMeasurement and MouthMeasurement")
        valid = (
            face_detected
            and eye.is_valid
            and mouth.is_valid
            and eye.eye_state in ("OPEN", "CLOSED")
            and mouth.mouth_state in ("MOUTH_OPEN", "MOUTH_CLOSED")
            and all(
                math.isfinite(value) and value >= 0
                for value in (
                    eye.left_ear,
                    eye.right_ear,
                    eye.avg_ear,
                    mouth.mar,
                )
            )
        )
        self._last_timestamp = timestamp
        if not valid:
            if self.consecutive_missing_frames == 0 and (
                self._last_valid_timestamp is not None
            ):
                self.eye_tracker.suspend(self._last_valid_timestamp)
                self.yawn_tracker.suspend(self._last_valid_timestamp)
            self.consecutive_missing_frames += 1
            expired = (
                self._gap_expired
                or self._timed_out(timestamp)
                or self.consecutive_missing_frames
                > self.settings.missing_face_tolerance_frames
            )
            if expired:
                self._clear_behavior()
            else:
                self._behavior = self._behavior.model_copy(
                    update={
                        "is_prolonged_closure": False,
                        "is_yawning_event": False,
                        "yawn_frequency_in_window": float(
                            self.yawn_tracker.window_count(timestamp)
                        ),
                    }
                )
            return TemporalAnalysisResult(
                eye=unknown_eye(),
                mouth=unknown_mouth(),
                behavior=self._behavior.model_copy(deep=True),
                action="RESET" if expired else "HOLD",
                is_valid=False,
                consecutive_missing_frames=self.consecutive_missing_frames,
            )
        recovered_reset = False
        if self.consecutive_missing_frames:
            recovered_reset = self._gap_expired or self._timed_out(timestamp)
            if recovered_reset:
                self._clear_behavior()
            else:
                self.eye_tracker.resume(timestamp)
                self.yawn_tracker.resume(timestamp)
        self.consecutive_missing_frames = 0
        self._gap_expired = False
        eye_result = self.eye_tracker.update_typed(eye.eye_state, timestamp)
        mouth_result = self.yawn_tracker.update_typed(
            mouth.mouth_state, timestamp
        )
        self._behavior = TemporalBehaviorState(
            eye_closed_duration=eye_result.closure_duration,
            consecutive_closed_frames=eye_result.consecutive_closed_frames,
            is_prolonged_closure=eye_result.prolonged_eye_closure,
            blink_count=self.eye_tracker.blink_count,
            mouth_open_duration=mouth_result.yawn_duration,
            yawn_count=mouth_result.yawn_count,
            yawn_frequency_in_window=mouth_result.yawn_frequency,
            is_yawning_event=mouth_result.yawn_event,
        )
        self._last_valid_timestamp = timestamp
        return TemporalAnalysisResult(
            eye=eye.model_copy(deep=True),
            mouth=mouth.model_copy(deep=True),
            behavior=self._behavior.model_copy(deep=True),
            action="RESET" if recovered_reset else "CONTINUE",
            is_valid=True,
            consecutive_missing_frames=0,
        )
