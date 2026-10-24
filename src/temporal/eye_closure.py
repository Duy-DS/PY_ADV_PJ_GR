"""Theo dõi nhắm mắt với API typed và wrapper tương thích T0701."""

import time
from dataclasses import asdict, dataclass
from typing import Optional

from src.utils.config import (
    TemporalSettings,
    load_config,
    numeric_setting,
    validate_timestamp,
)


@dataclass(frozen=True)
class EyeClosureResult:
    """Kết quả tracker bất biến.

    Parameters: Cờ sự kiện, mốc bắt đầu, số frame và thời lượng.
    Returns: Kết quả typed.
    Raises: Không có.
    """

    prolonged_eye_closure: bool
    blink_detected: bool
    closed_start_time: Optional[float]
    consecutive_closed_frames: int
    closure_duration: float


class EyeClosureTracker:
    """Đếm nhắm mắt; coordinator sử dụng update_typed an toàn.

    Parameters: settings là TemporalSettings hoặc None cho YAML mặc định.
    Returns: Tracker có thể reset giữa video.
    Raises: ValueError nếu cấu hình không hợp lệ.
    """

    def __init__(self, settings: Optional[TemporalSettings] = None):
        """Khởi tạo từ cấu hình, không dùng ngưỡng số dự phòng.

        Parameters: settings dùng chung hoặc None.
        Returns: Không có.
        Raises: ValueError, OSError nếu cấu hình sai.
        """
        self.closure_time_threshold = (
            settings.closure_time_threshold
            if settings is not None
            else numeric_setting(
                load_config(), "eye", "closure_time_threshold"
            )
        )
        self.reset()

    def reset(self):
        """Xóa toàn bộ timer và bộ đếm mắt.

        Parameters: Không có.
        Returns: Không có.
        Raises: Không có.
        """
        self.closed_start_time = None
        self.consecutive_closed_frames = 0
        self.blink_count = 0
        self._last_timestamp = None
        self._suspended_at = None

    def suspend(self, timestamp):
        """Đóng băng timer từ mốc hợp lệ cuối, không sửa số frame.

        Parameters: timestamp là mốc bắt đầu khoảng không quan sát.
        Returns: Không có.
        Raises: ValueError nếu mốc không hợp lệ.
        """
        timestamp = validate_timestamp(timestamp, self._last_timestamp)
        if self._suspended_at is None:
            self._suspended_at = timestamp

    def resume(self, timestamp):
        """Loại khoảng mất dữ liệu khỏi thời lượng nhắm mắt.

        Parameters: timestamp là mốc quan sát trở lại.
        Returns: Không có.
        Raises: ValueError nếu thời gian lùi.
        """
        timestamp = validate_timestamp(timestamp, self._suspended_at)
        if self._suspended_at is not None:
            if self.closed_start_time is not None:
                self.closed_start_time += timestamp - self._suspended_at
            self._suspended_at = None

    def update_typed(self, eye_state, current_time, *, confirm=True):
        """Cập nhật tracker; yêu cầu nhiều quan sát cho nhắm mắt kéo dài.

        Parameters: eye_state, current_time; confirm bật bảo vệ frame đơn.
        Returns: EyeClosureResult.
        Raises: ValueError nếu trạng thái hoặc thời gian sai.
        """
        if eye_state not in ("OPEN", "CLOSED", "UNKNOWN"):
            raise ValueError("eye_state must be OPEN, CLOSED or UNKNOWN")
        current_time = validate_timestamp(current_time, self._last_timestamp)
        if self._suspended_at is not None:
            raise ValueError("Resume tracker before updating")
        duration = 0.0
        prolonged = False
        blink = False
        if self.closed_start_time is not None:
            duration = current_time - self.closed_start_time
        if eye_state == "CLOSED":
            if self.closed_start_time is None:
                self.closed_start_time = current_time
            self.consecutive_closed_frames += 1
            prolonged = duration >= self.closure_time_threshold
            if confirm:
                prolonged = (
                    prolonged
                    and self.consecutive_closed_frames > 1
                    and duration > 0
                )
        elif eye_state == "OPEN":
            if self.closed_start_time is not None:
                prolonged = duration >= self.closure_time_threshold
                if confirm:
                    prolonged = (
                        prolonged
                        and self.consecutive_closed_frames > 1
                        and duration > 0
                    )
                blink = not prolonged
                self.blink_count += int(blink)
            self.closed_start_time = None
            self.consecutive_closed_frames = 0
        else:
            self.closed_start_time = None
            self.consecutive_closed_frames = 0
        self._last_timestamp = current_time
        return EyeClosureResult(
            prolonged,
            blink,
            self.closed_start_time,
            self.consecutive_closed_frames,
            duration,
        )

    def update(self, eye_state, current_time=None):
        """Wrapper dictionary giữ nguyên semantics T0701 cho caller cũ.

        Parameters: eye_state, current_time hoặc None cho đồng hồ hệ thống.
        Returns: Dictionary legacy; coordinator không dùng API này.
        Raises: ValueError nếu trạng thái/thời gian sai.
        """
        if current_time is None:
            current_time = time.time()
        return asdict(
            self.update_typed(eye_state, current_time, confirm=False)
        )
