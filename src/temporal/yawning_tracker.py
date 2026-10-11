import math
import time
from collections import deque
from src.utils.config import load_config


# Lớp theo dõi thời lượng và tần suất ngáp theo chuỗi thời gian.
class YawningTracker:
    def __init__(self):
        # Đọc cấu hình từ file YAML
        config = load_config()

        mouth_config = config.get("mouth", {})

        # Lấy ngưỡng thời lượng ngáp từ cấu hình (mặc định 2.0 giây)
        self.yawn_duration_threshold = float(
            mouth_config.get("yawn_duration_threshold", 2.0)
        )

        # Lấy khoảng thời gian trượt (time window) để tính tần suất ngáp (mặc định 60 giây)
        self.time_window = float(
            mouth_config.get("yawn_time_window", 60.0)
        )

        # Kiểm tra các ngưỡng phải là số hữu hạn và hợp lệ
        if (
            not math.isfinite(self.yawn_duration_threshold)
            or self.yawn_duration_threshold < 0
        ):
            raise ValueError(
                "mouth.yawn_duration_threshold must be a finite, non-negative number"
            )

        if not math.isfinite(self.time_window) or self.time_window <= 0:
            raise ValueError(
                "mouth.yawn_time_window must be a finite, positive number"
            )

        # Thời điểm bắt đầu lần há miệng / ngáp hiện tại
        self.open_start_time = None

        # Tổng số lần ngáp ghi nhận được
        self.yawn_count = 0

        # Cờ đánh dấu sự kiện ngáp đã được tính cho chu kỳ mở miệng hiện tại chưa
        self.counted_current_yawn = False

        # Hàng đợi lưu vết các mốc thời gian xảy ra sự kiện ngáp trong phạm vi time_window
        self.yawn_timestamps = deque()

    # Cập nhật trạng thái miệng theo từng frame.
    def update(self, mouth_state, current_time=None):
        # Sử dụng thời gian hiện tại của hệ thống nếu không truyền vào
        if current_time is None:
            current_time = time.time()

        # Chuẩn hóa thời gian về kiểu float
        current_time = float(current_time)

        # Thời gian phải là số hữu hạn
        if not math.isfinite(current_time):
            raise ValueError("current_time must be a finite number")

        # Khởi tạo kết quả cho frame hiện tại
        yawn_event = False
        yawn_duration = 0.0

        if mouth_state == "MOUTH_OPEN":
            # Bắt đầu ghi nhận thời điểm mở miệng nếu đây là frame MOUTH_OPEN đầu tiên
            if self.open_start_time is None:
                self.open_start_time = current_time
                self.counted_current_yawn = False

            # Tính thời gian mở miệng liên tục
            yawn_duration = current_time - self.open_start_time

            # Xác định sự kiện ngáp khi thời gian đạt hoặc vượt ngưỡng và chưa được đếm trước đó trong chu kỳ này
            if (
                yawn_duration >= self.yawn_duration_threshold
                and not self.counted_current_yawn
            ):
                yawn_event = True
                self.yawn_count += 1
                self.yawn_timestamps.append(current_time)
                self.counted_current_yawn = True

        elif mouth_state == "MOUTH_CLOSED":
            # Nếu trước đó miệng đang mở, kết thúc chu kỳ mở miệng hiện tại
            if self.open_start_time is not None:
                yawn_duration = current_time - self.open_start_time

            # Reset trạng thái sau khi miệng khép lại
            self.open_start_time = None
            self.counted_current_yawn = False
            yawn_duration = 0.0

        elif mouth_state == "UNKNOWN":
            # Trạng thái UNKNOWN làm gián đoạn chuỗi theo dõi ngáp
            self.open_start_time = None
            self.counted_current_yawn = False
            yawn_duration = 0.0

        else:
            # Từ chối trạng thái miệng không hợp lệ
            raise ValueError(
                "mouth_state must be 'MOUTH_OPEN', 'MOUTH_CLOSED', or 'UNKNOWN'"
            )

        # Dọn dẹp các mốc thời gian ngáp cũ nằm ngoài khoảng thời gian trượt (time_window)
        while self.yawn_timestamps and (
            current_time - self.yawn_timestamps[0] > self.time_window
        ):
            self.yawn_timestamps.popleft()

        # Tính tần suất ngáp hiện tại trong cửa sổ thời gian
        yawn_frequency = len(self.yawn_timestamps)

        # Trả về kết quả theo dõi của frame hiện tại
        return {
            "yawn_event": yawn_event,
            "yawn_duration": yawn_duration,
            "yawn_count": self.yawn_count,
            "yawn_frequency": yawn_frequency,
        }