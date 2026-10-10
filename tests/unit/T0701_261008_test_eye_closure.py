"""
Unit test cho EyeClosureTracker.

Kiểm tra các chức năng:
- Theo dõi trạng thái OPEN/CLOSED/UNKNOWN.
- Đếm số frame CLOSED liên tiếp.
- Phát hiện blink khi nhắm mắt trong thời gian ngắn.
- Phát hiện prolonged eye closure khi vượt ngưỡng thời gian.
- Reset trạng thái sau khi mở mắt hoặc gặp UNKNOWN.
- Đọc closure_time_threshold từ file YAML.
- Xử lý trạng thái mắt không hợp lệ.
- Kiểm tra hành vi tại đúng ngưỡng thời gian.
"""

import pytest

from src.temporal import eye_closure
from src.temporal.eye_closure import EyeClosureTracker
from src.utils.config import load_config


# Kiểm tra trường hợp mắt đang mở.
def test_open_eye():
    tracker = EyeClosureTracker()

    result = tracker.update(
        eye_state="OPEN",
        current_time=0.0
    )
    # Tracker không được ghi nhận nhắm mắt, blink hoặc prolonged closure.
    assert result["prolonged_eye_closure"] is False
    assert result["blink_detected"] is False
    assert result["closed_start_time"] is None
    assert result["consecutive_closed_frames"] == 0


# Kiểm tra trường hợp nhắm mắt trong thời gian ngắn và được xác định là blink.
def test_short_blink():
    tracker = EyeClosureTracker()

    # Bắt đầu một lần nhắm mắt
    result = tracker.update(
        eye_state="CLOSED",
        current_time=1.0
    )

    assert result["closed_start_time"] == 1.0
    assert result["consecutive_closed_frames"] == 1
    assert result["prolonged_eye_closure"] is False
    assert result["blink_detected"] is False

    # Tiếp tục nhắm mắt, thời gian nhắm mắt được duy trì
    result = tracker.update(
        eye_state="CLOSED",
        current_time=1.1
    )

    assert result["closed_start_time"] == 1.0
    assert result["consecutive_closed_frames"] == 2
    assert result["prolonged_eye_closure"] is False
    assert result["blink_detected"] is False

    # Mở mắt lại, kết thúc lần nhắm mắt ngắn
    result = tracker.update(
        eye_state="OPEN",
        current_time=1.2
    )

    # Thời gian nhắm mắt = 1.2 - 1.0 = 0.2 giây
    assert result["prolonged_eye_closure"] is False
    assert result["blink_detected"] is True
    assert result["closure_duration"] == pytest.approx(0.2)

    # Trạng thái theo dõi được reset sau khi mở mắt
    assert result["closed_start_time"] is None
    assert result["consecutive_closed_frames"] == 0


# Kiểm tra trường hợp mắt nhắm liên tục vượt quá ngưỡng prolonged closure.
def test_prolonged_eye_closure():
    tracker = EyeClosureTracker()

    # Bắt đầu nhắm mắt
    result = tracker.update(
        eye_state="CLOSED",
        current_time=1.0
    )

    assert result["prolonged_eye_closure"] is False

    # Tiếp tục nhắm mắt sau 1 giây nhưng chưa vượt ngưỡng
    result = tracker.update(
        eye_state="CLOSED",
        current_time=2.0
    )

    assert result["prolonged_eye_closure"] is False

    # Thời gian nhắm mắt vượt ngưỡng 2 giây
    result = tracker.update(
        eye_state="CLOSED",
        current_time=3.1
    )

    assert result["prolonged_eye_closure"] is True
    assert result["blink_detected"] is False
    assert result["closed_start_time"] == 1.0
    assert result["consecutive_closed_frames"] == 3


# Kiểm tra số frame CLOSED liên tiếp được đếm chính xác.
def test_consecutive_closed_frames():
    tracker = EyeClosureTracker()

    tracker.update("CLOSED", current_time=1.0)
    tracker.update("CLOSED", current_time=1.1)
    tracker.update("CLOSED", current_time=1.2)

    assert tracker.consecutive_closed_frames == 3


# Kiểm tra tracker reset trạng thái sau khi mắt mở lại.
def test_reset_after_open():
    tracker = EyeClosureTracker()

    tracker.update("CLOSED", current_time=1.0)
    tracker.update("CLOSED", current_time=1.1)

    result = tracker.update(
        eye_state="OPEN",
        current_time=1.2
    )

    # Sau khi mở mắt, thông tin của lần nhắm mắt trước được reset
    assert result["closed_start_time"] is None
    assert result["consecutive_closed_frames"] == 0


# Kiểm tra tracker có thể bắt đầu theo dõi một lần nhắm mắt mới
# sau khi lần nhắm mắt trước đã được reset.
def test_new_closure_after_reset():
    tracker = EyeClosureTracker()

    # Lần nhắm mắt thứ nhất
    tracker.update("CLOSED", current_time=1.0)
    tracker.update("OPEN", current_time=1.2)

    # Lần nhắm mắt thứ hai
    result = tracker.update(
        eye_state="CLOSED",
        current_time=2.0
    )

    # Tracker bắt đầu tính lại từ thời điểm 2.0
    assert result["closed_start_time"] == 2.0
    assert result["consecutive_closed_frames"] == 1
    assert result["prolonged_eye_closure"] is False


# Kiểm tra tracker từ chối trạng thái mắt không hợp lệ.
def test_invalid_eye_state():
    tracker = EyeClosureTracker()

    # Trạng thái không thuộc OPEN, CLOSED hoặc UNKNOWN phải gây ValueError
    with pytest.raises(ValueError):
        tracker.update(
            eye_state="INVALID",
            current_time=1.0
        )


# Kiểm tra ngưỡng thời gian nhắm mắt được đọc từ file YAML,
# không được hard-code trực tiếp trong EyeClosureTracker.
def test_threshold_is_loaded_from_yaml(monkeypatch):
    # Thay thế load_config() bằng cấu hình giả lập trong test
    monkeypatch.setattr(
        eye_closure,
        "load_config",
        lambda: {"eye": {"closure_time_threshold": 0.75}},
    )

    tracker = EyeClosureTracker()

    # Tracker phải sử dụng đúng giá trị từ config
    assert tracker.closure_time_threshold == pytest.approx(0.75)


# Kiểm tra trường hợp thời gian nhắm mắt đúng bằng threshold.
# Tại ngưỡng, trạng thái được xem là prolonged eye closure.
def test_closure_at_threshold_is_prolonged():
    tracker = EyeClosureTracker()
    start_time = 10.0

    # Lấy threshold thực tế từ file YAML
    threshold = load_config()["eye"]["closure_time_threshold"]

    tracker.update("CLOSED", current_time=start_time)

    # Thời gian nhắm mắt đúng bằng threshold
    result = tracker.update(
        "CLOSED",
        current_time=start_time + threshold,
    )

    assert result["prolonged_eye_closure"] is True
    assert result["blink_detected"] is False


# Kiểm tra khi mở mắt sau một lần prolonged closure.
# Sự kiện prolonged closure vẫn phải được trả về ở frame mở mắt.
def test_reopening_after_prolonged_closure_keeps_event_visible():
    tracker = EyeClosureTracker()
    start_time = 10.0

    # Lấy threshold từ file YAML
    threshold = load_config()["eye"]["closure_time_threshold"]

    tracker.update("CLOSED", current_time=start_time)

    # Mở mắt sau khi đã vượt threshold
    result = tracker.update(
        "OPEN",
        current_time=start_time + threshold + 0.1,
    )

    # Sự kiện prolonged closure vẫn phải được ghi nhận
    assert result["prolonged_eye_closure"] is True
    assert result["blink_detected"] is False

    # Sau đó tracker được reset
    assert result["closed_start_time"] is None
    assert result["consecutive_closed_frames"] == 0


# Kiểm tra trạng thái UNKNOWN làm gián đoạn chuỗi nhắm mắt liên tục.
def test_unknown_breaks_continuous_closure_tracking():
    tracker = EyeClosureTracker()

    # Bắt đầu và tiếp tục nhắm mắt
    tracker.update("CLOSED", current_time=1.0)
    tracker.update("CLOSED", current_time=1.1)

    # UNKNOWN làm mất tính liên tục của trạng thái CLOSED
    result = tracker.update("UNKNOWN", current_time=1.2)

    assert result["prolonged_eye_closure"] is False
    assert result["blink_detected"] is False
    assert result["closed_start_time"] is None
    assert result["consecutive_closed_frames"] == 0

    # CLOSED sau UNKNOWN phải được xem là một lần nhắm mắt mới
    result = tracker.update("CLOSED", current_time=1.3)

    assert result["closed_start_time"] == 1.3
    assert result["consecutive_closed_frames"] == 1