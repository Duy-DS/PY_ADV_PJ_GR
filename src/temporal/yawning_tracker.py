"""Theo dõi ngáp và cửa sổ trượt theo thời gian video gốc."""

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

from src.utils.config import (
    TemporalSettings,
    load_config,
    numeric_setting,
    validate_timestamp,
)


@dataclass(frozen=True)
class YawningResult:
    """Kết quả ngáp typed.

    Parameters: Sự kiện, thời lượng, tổng số và số lần trong cửa sổ.
    Returns: Snapshot bất biến.
    Raises: Không có.
    """

    yawn_event: bool
    yawn_duration: float
    yawn_count: int
    yawn_frequency: int


class YawningTracker:
    """Theo dõi một sự kiện mỗi chu kỳ miệng mở.

    Parameters: settings dùng chung hoặc None cho YAML mặc định.
    Returns: Tracker có reset/suspend/resume.
    Raises: ValueError nếu cấu hình sai.
    """

    def __init__(self, settings: Optional[TemporalSettings] = None):
        """Đọc thời lượng và cửa sổ bắt buộc từ cấu hình.

        Parameters: settings hoặc None.
        Returns: Không có.
        Raises: OSError, ValueError nếu cấu hình sai.
        """
        if settings is None:
            config = load_config()
            self.yawn_duration_threshold = numeric_setting(
                config, "mouth", "yawn_duration_threshold"
            )
            self.time_window = numeric_setting(
                config, "mouth", "yawn_frequency_window", positive=True
            )
        else:
            self.yawn_duration_threshold = settings.yawn_duration_threshold
            self.time_window = settings.yawn_frequency_window
        self.reset()

    def reset(self):
        """Xóa chu kỳ, bộ đếm và lịch sử ngáp giữa video.

        Parameters: Không có.
        Returns: Không có.
        Raises: Không có.
        """
        self.open_start_time = None
        self.yawn_count = 0
        self.counted_current_yawn = False
        self.yawn_timestamps = deque()
        self._open_frames = 0
        self._last_timestamp = None
        self._suspended_at = None

    def suspend(self, timestamp):
        """Đóng băng chu kỳ miệng ở mốc hợp lệ cuối.

        Parameters: timestamp bắt đầu khoảng không quan sát.
        Returns: Không có.
        Raises: ValueError nếu thời gian sai.
        """
        timestamp = validate_timestamp(timestamp, self._last_timestamp)
        if self._suspended_at is None:
            self._suspended_at = timestamp

    def resume(self, timestamp):
        """Bỏ thời gian gap khỏi timer nhưng giữ mốc ngáp gốc.

        Parameters: timestamp quan sát trở lại.
        Returns: Không có.
        Raises: ValueError nếu thời gian lùi.
        """
        timestamp = validate_timestamp(timestamp, self._suspended_at)
        if self._suspended_at is not None:
            if self.open_start_time is not None:
                self.open_start_time += timestamp - self._suspended_at
            self._suspended_at = None

    def window_count(self, timestamp):
        """Loại sự kiện cũ; giữ biên timestamp - event == time_window.

        Parameters: timestamp của video gốc.
        Returns: int, số lần ngáp trong cửa sổ, không phải rate mỗi giây.
        Raises: ValueError nếu thời gian sai.
        """
        timestamp = validate_timestamp(timestamp, self._last_timestamp)
        while self.yawn_timestamps and (
            timestamp - self.yawn_timestamps[0] > self.time_window
        ):
            self.yawn_timestamps.popleft()
        return len(self.yawn_timestamps)

    def update_typed(self, mouth_state, current_time, *, confirm=True):
        """Cập nhật ngáp; không phát lại sự kiện trong cùng chu kỳ.

        Parameters: mouth_state, current_time; confirm bảo vệ frame đơn.
        Returns: YawningResult.
        Raises: ValueError nếu trạng thái/thời gian sai hoặc chưa resume.
        """
        if mouth_state not in ("MOUTH_OPEN", "MOUTH_CLOSED", "UNKNOWN"):
            raise ValueError("Invalid mouth_state")
        current_time = validate_timestamp(current_time, self._last_timestamp)
        if self._suspended_at is not None:
            raise ValueError("Resume tracker before updating")
        event = False
        duration = 0.0
        if mouth_state == "MOUTH_OPEN":
            if self.open_start_time is None:
                self.open_start_time = current_time
                self.counted_current_yawn = False
            self._open_frames += 1
            duration = current_time - self.open_start_time
            corroborated = not confirm or (
                self._open_frames > 1 and duration > 0
            )
            if (
                duration >= self.yawn_duration_threshold
                and corroborated
                and not self.counted_current_yawn
            ):
                event = True
                self.yawn_count += 1
                self.yawn_timestamps.append(current_time)
                self.counted_current_yawn = True
        else:
            self.open_start_time = None
            self.counted_current_yawn = False
            self._open_frames = 0
        frequency = self.window_count(current_time)
        self._last_timestamp = current_time
        return YawningResult(event, duration, self.yawn_count, frequency)

    def update(self, mouth_state, current_time=None):
        """Wrapper dictionary tương thích caller cũ.

        Parameters: mouth_state và current_time hoặc đồng hồ hệ thống.
        Returns: Dictionary legacy; coordinator dùng update_typed.
        Raises: ValueError nếu trạng thái/thời gian sai.
        """
        if current_time is None:
            current_time = time.time()
        return asdict(
            self.update_typed(mouth_state, current_time, confirm=False)
        )
