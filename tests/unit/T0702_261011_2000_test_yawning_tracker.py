import time
import pytest
from src.temporal.yawning_tracker import YawningTracker

def test_yawning_tracker_initialization():
    """Kiểm tra khởi tạo tracker thành công với các giá trị hợp lệ từ config."""
    tracker = YawningTracker()
    assert tracker.yawn_count == 0
    assert tracker.open_start_time is None

def test_short_mouth_opening():
    """Kiểm tra trường hợp mở miệng ngắn (dưới ngưỡng threshold) -> không tính là ngáp."""
    tracker = YawningTracker()
    # Giả lập thời gian bắt đầu
    start_t = time.time()
    
    # Mở miệng ngắn (chỉ cách nhau 0.5 giây, nhỏ hơn ngưỡng mặc định 2.0s)
    res1 = tracker.update("MOUTH_OPEN", current_time=start_t)
    assert res1["yawn_event"] is False
    assert res1["yawn_duration"] == 0.0

    res2 = tracker.update("MOUTH_OPEN", current_time=start_t + 0.5)
    assert res2["yawn_event"] is False
    assert res2["yawn_duration"] == 0.5

    # Khép miệng lại
    res3 = tracker.update("MOUTH_CLOSED", current_time=start_t + 0.6)
    assert res3["yawn_event"] is False
    assert tracker.yawn_count == 0

def test_prolonged_yawn_event():
    """Kiểm tra trường hợp mở miệng liên tục vượt ngưỡng -> kích hoạt yawn_event."""
    tracker = YawningTracker()
    start_t = 100.0  # Dùng một mốc số thực cố định để tránh lệch thời gian hệ thống nếu muốn, hoặc giữ time.time()
    
    # Mở miệng tại thời điểm t = 100.0
    tracker.update("MOUTH_OPEN", current_time=start_t)

    # Mở miệng kéo dài vượt ngưỡng (truyền 2.2 thay vì 2.1 để bù trừ sai số float)
    res = tracker.update("MOUTH_OPEN", current_time=start_t + 2.2)
    
    assert res["yawn_event"] is True
    assert res["yawn_duration"] >= 2.0  # Hoặc kiểm tra lớn hơn ngưỡng
    assert res["yawn_count"] == 1
    assert res["yawn_frequency"] == 1

def test_invalid_mouth_state():
    """Kiểm tra ngoại lệ khi truyền trạng thái không hợp lệ."""
    tracker = YawningTracker()
    with pytest.raises(ValueError):
        tracker.update("INVALID_STATE")