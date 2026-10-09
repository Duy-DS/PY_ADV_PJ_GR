import math
import time

from src.utils.config import load_config

# Lớp theo dõi trạng thái nhắm mắt qua các frame video.
class EyeClosureTracker:
    def __init__(self):
        # Đọc cấu hình từ file YAML
        config = load_config()

        # Lấy ngưỡng thời gian nhắm mắt từ cấu hình
        self.closure_time_threshold = float(
            config["eye"]["closure_time_threshold"]
        )

        # Kiểm tra ngưỡng phải là số hữu hạn và không âm
        if (
            not math.isfinite(self.closure_time_threshold)
            or self.closure_time_threshold < 0
        ):
            raise ValueError(
                "eye.closure_time_threshold must be a finite, "
                "non-negative number"
            )

        # Thời điểm bắt đầu lần nhắm mắt hiện tại
        self.closed_start_time = None

        # Số frame CLOSED liên tiếp
        self.consecutive_closed_frames = 0


# Cập nhật trạng thái mắt theo từng frame.
    def update(self, eye_state, current_time=None):
        # Sử dụng thời gian hiện tại của hệ thống
        if current_time is None:
            current_time = time.time()

        # Chuẩn hóa thời gian về kiểu float
        current_time = float(current_time)

        # Thời gian phải là số hữu hạn
        if not math.isfinite(current_time):
            raise ValueError("current_time must be a finite number")

        # Khởi tạo kết quả cho frame hiện tại
        prolonged_eye_closure = False
        blink_detected = False
        closure_duration = 0.0

        if eye_state == "CLOSED":
            # Bắt đầu ghi nhận thời điểm nhắm mắt
            # nếu đây là frame CLOSED đầu tiên
            if self.closed_start_time is None:
                self.closed_start_time = current_time

            # Tăng số frame CLOSED liên tiếp
            self.consecutive_closed_frames += 1

            # Tính thời gian mắt đã nhắm liên tục
            closure_duration = (
                current_time - self.closed_start_time
            )

            # Xác định prolonged closure khi thời gian
            # nhắm mắt đạt hoặc vượt ngưỡng cấu hình
            prolonged_eye_closure = (
                closure_duration >= self.closure_time_threshold
            )

        elif eye_state == "OPEN":
            # Nếu trước đó mắt đang nhắm, kết thúc lần nhắm mắt hiện tại
            if self.closed_start_time is not None:
                closure_duration = (
                    current_time - self.closed_start_time
                )

                # Kiểm tra lần nhắm mắt vừa kết thúc có phải prolonged closure
                prolonged_eye_closure = (
                    closure_duration >= self.closure_time_threshold
                )

                # Chỉ xác định blink nếu thời gian nhắm mắt chưa vượt ngưỡng
                blink_detected = not prolonged_eye_closure

            # Reset trạng thái sau khi mắt mở
            self.closed_start_time = None
            self.consecutive_closed_frames = 0

        elif eye_state == "UNKNOWN":
            # UNKNOWN làm gián đoạn việc theo dõi nhắm mắt liên tục
            if self.closed_start_time is not None:
                closure_duration = (
                    current_time - self.closed_start_time
                )

            # Reset trạng thái theo dõi
            self.closed_start_time = None
            self.consecutive_closed_frames = 0

        else:
            # Từ chối trạng thái mắt không hợp lệ
            raise ValueError(
                "eye_state must be 'OPEN', 'CLOSED', or 'UNKNOWN'"
            )

        # Trả về kết quả theo dõi của frame hiện tại
        return {
            "prolonged_eye_closure": prolonged_eye_closure,
            "blink_detected": blink_detected,
            "closed_start_time": self.closed_start_time,
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "closure_duration": closure_duration,
        }