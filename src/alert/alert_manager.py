import time
from collections.abc import Mapping
from enum import Enum

from src.alert.alarm import Alarm
from src.utils.config import load_config, numeric_setting


# Các trạng thái của hệ thống cảnh báo
class AlertState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    COOLDOWN = "COOLDOWN"


class AlertManager:
    """Quản lý trạng thái cảnh báo và thời gian phục hồi."""

    def __init__(self, config=None):
        # Khởi tạo cấu hình và thiết lập thời gian cooldown, âm thanh cảnh báo
        if config is None:
            config = load_config()

        alert_config = config.get("alert", {})
        self.cooldown = numeric_setting(config, "alert", "cooldown")

        sound_enabled = alert_config.get("sound_enabled", True)
        if not isinstance(sound_enabled, bool):
            raise ValueError("alert.sound_enabled must be a boolean")
        self.alarm = Alarm(sound_enabled=sound_enabled)

        # Khởi tạo trạng thái ban đầu và thời điểm bắt đầu cooldown
        self.state = AlertState.NORMAL
        self.cooldown_start_time = None

    def update(self, decision, current_time=None):
        """Cập nhật trạng thái cảnh báo từ Decision Engine."""

        # Lấy thời gian hiện tại và chuẩn hóa trạng thái đầu vào
        if current_time is None:
            current_time = time.monotonic()

        decision_state = self._get_decision_state(decision)

        # Phát hiện buồn ngủ: kích hoạt cảnh báo và hủy cooldown
        if decision_state == "DROWSY":
            self.state = AlertState.WARNING
            self.cooldown_start_time = None
            self.alarm.activate()

        # Trạng thái chưa xác định: quay lại WARNING nếu đang cooldown
        elif decision_state == "UNKNOWN":
            if self.state == AlertState.COOLDOWN:
                self.state = AlertState.WARNING
                self.cooldown_start_time = None

        # Không buồn ngủ: bắt đầu hoặc kiểm tra thời gian cooldown
        else:
            if self.state == AlertState.WARNING:
                self.state = AlertState.COOLDOWN
                self.cooldown_start_time = current_time

            if self.state == AlertState.COOLDOWN:
                elapsed = current_time - self.cooldown_start_time

                # Hết thời gian cooldown thì trở về NORMAL và tắt cảnh báo
                if elapsed >= self.cooldown:
                    self.state = AlertState.NORMAL
                    self.cooldown_start_time = None
                    self.alarm.deactivate()

        return self.state

    @staticmethod
    def _get_decision_state(decision):
        """Trích xuất và kiểm tra trạng thái từ kết quả đầu vào."""

        # Hỗ trợ đầu vào dạng chuỗi, Mapping hoặc đối tượng có thuộc tính state
        if isinstance(decision, str):
            state = decision
        elif isinstance(decision, Mapping):
            state = decision.get("state")
        else:
            state = getattr(decision, "state", None)

        # Kiểm tra tính hợp lệ và chuẩn hóa trạng thái
        if not isinstance(state, str):
            raise ValueError(f"Invalid drowsiness decision state: {state!r}")
        if state == "NORMAL":
            state = "NON_DROWSY"
        if state not in {"DROWSY", "NON_DROWSY", "UNKNOWN"}:
            raise ValueError(f"Invalid drowsiness decision state: {state!r}")

        return state

    def reset(self):
        """Đưa hệ thống về trạng thái ban đầu."""

        # Tắt cảnh báo và đặt lại toàn bộ trạng thái quản lý
        self.alarm.deactivate()
        self.state = AlertState.NORMAL
        self.cooldown_start_time = None
