import pytest

from src.alert.alert_manager import AlertManager, AlertState


# Tạo AlertManager với cấu hình tùy chỉnh phục vụ kiểm thử
def create_manager(cooldown=3.0, sound_enabled=True):
    config = {
        "alert": {
            "cooldown": cooldown,
            "sound_enabled": sound_enabled,
        }
    }
    return AlertManager(config)


# Kiểm tra cấu hình mặc định được đọc từ YAML
def test_default_alert_settings_load_from_yaml():
    manager = AlertManager()

    assert manager.cooldown == 3.0
    assert manager.alarm.sound_enabled is True


# Kiểm tra toàn bộ chu trình NORMAL -> WARNING -> COOLDOWN -> NORMAL
def test_alert_cycle_normal_warning_cooldown_normal():
    manager = create_manager(cooldown=3.0)

    # NORMAL: Trạng thái ban đầu
    assert manager.state == AlertState.NORMAL
    assert manager.alarm.is_playing is False

    # NORMAL -> WARNING: Phát hiện buồn ngủ
    state = manager.update("DROWSY", current_time=1.0)

    assert state == AlertState.WARNING
    assert manager.alarm.is_playing is True
    assert manager.cooldown_start_time is None

    # WARNING -> COOLDOWN: Không còn phát hiện buồn ngủ
    decision = {
        "state": "NON_DROWSY",
        "reason": "Normal behavior",
    }
    state = manager.update(decision, current_time=2.0)

    assert state == AlertState.COOLDOWN
    assert manager.alarm.is_playing is True
    assert manager.cooldown_start_time == 2.0

    # Kiểm tra trạng thái khi chưa hết thời gian cooldown
    state = manager.update("NON_DROWSY", current_time=4.9)

    assert state == AlertState.COOLDOWN
    assert manager.alarm.is_playing is True

    # Kiểm tra trạng thái khi đã hết thời gian cooldown
    state = manager.update("NON_DROWSY", current_time=5.0)

    assert state == AlertState.NORMAL
    assert manager.alarm.is_playing is False
    assert manager.cooldown_start_time is None

# Kiểm tra trạng thái NORMAL được chấp nhận và chuẩn hóa
def test_normal_string_is_accepted():
    manager = create_manager()

    manager.update("DROWSY", current_time=1.0)

    state = manager.update("NORMAL", current_time=2.0)

    assert state == AlertState.COOLDOWN
    assert manager.cooldown_start_time == 2.0
    assert manager.alarm.is_playing is True


# Kiểm tra DROWSY hoặc UNKNOWN xuất hiện trong thời gian cooldown
def test_drowsy_during_cooldown_reactivates_warning():
    manager = create_manager(cooldown=3.0)

    manager.update("DROWSY", current_time=1.0)
    manager.update("NON_DROWSY", current_time=2.0)

    state = manager.update("DROWSY", current_time=3.0)

    assert state == AlertState.WARNING
    assert manager.alarm.is_playing is True
    assert manager.cooldown_start_time is None


def test_unknown_during_cooldown_returns_to_warning():
    manager = create_manager(cooldown=3.0)

    manager.update("DROWSY", current_time=1.0)
    manager.update("NON_DROWSY", current_time=2.0)

    state = manager.update("UNKNOWN", current_time=3.0)

    assert state == AlertState.WARNING
    assert manager.alarm.is_playing is True
    assert manager.cooldown_start_time is None


# Kiểm tra UNKNOWN khi hệ thống đang WARNING hoặc NORMAL
def test_unknown_while_warning_keeps_warning():
    manager = create_manager()

    manager.update("DROWSY", current_time=1.0)

    state = manager.update("UNKNOWN", current_time=2.0)

    assert state == AlertState.WARNING
    assert manager.alarm.is_playing is True
    assert manager.cooldown_start_time is None


def test_unknown_while_normal_keeps_normal():
    manager = create_manager()

    state = manager.update("UNKNOWN", current_time=1.0)

    assert state == AlertState.NORMAL
    assert manager.alarm.is_playing is False


# Kiểm tra chức năng bật/tắt âm thanh độc lập với trạng thái cảnh báo
def test_sound_can_be_disabled_without_disabling_warning_state():
    manager = create_manager(sound_enabled=False)

    state = manager.update("DROWSY", current_time=1.0)

    assert state == AlertState.WARNING
    assert manager.alarm.sound_enabled is False
    assert manager.alarm.is_playing is False


# Kiểm tra các giá trị cooldown hợp lệ và không hợp lệ
def test_zero_cooldown():
    manager = create_manager(cooldown=0.0)

    manager.update("DROWSY", current_time=1.0)

    state = manager.update("NON_DROWSY", current_time=2.0)

    assert state == AlertState.NORMAL
    assert manager.alarm.is_playing is False
    assert manager.cooldown_start_time is None


def test_negative_cooldown_is_invalid():
    with pytest.raises(ValueError):
        create_manager(cooldown=-1.0)


@pytest.mark.parametrize("cooldown", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_cooldown_is_invalid(cooldown):
    with pytest.raises(ValueError):
        create_manager(cooldown=cooldown)


# Kiểm tra xử lý cấu hình âm thanh không đúng kiểu dữ liệu
def test_invalid_sound_enabled_type():
    with pytest.raises(ValueError, match="sound_enabled"):
        AlertManager({
            "alert": {
                "cooldown": 3.0,
                "sound_enabled": "false",
            }
        })


# Kiểm tra xử lý trạng thái đầu vào không hợp lệ hoặc bị thiếu
def test_invalid_drowsiness_state():
    manager = create_manager()

    with pytest.raises(
        ValueError,
        match="Invalid drowsiness decision state",
    ):
        manager.update("INVALID", current_time=1.0)


def test_missing_decision_state_is_invalid():
    manager = create_manager()

    with pytest.raises(
        ValueError,
        match="Invalid drowsiness decision state",
    ):
        manager.update(
            {"reason": "Missing state"},
            current_time=1.0,
        )


# Kiểm tra reset đưa hệ thống từ WARNING hoặc COOLDOWN về NORMAL
def test_reset_returns_warning_to_normal():
    manager = create_manager()

    manager.update("DROWSY", current_time=1.0)
    manager.reset()

    assert manager.state == AlertState.NORMAL
    assert manager.alarm.is_playing is False
    assert manager.cooldown_start_time is None


def test_reset_returns_cooldown_to_normal():
    manager = create_manager()

    manager.update("DROWSY", current_time=1.0)
    manager.update("NON_DROWSY", current_time=2.0)
    manager.reset()

    assert manager.state == AlertState.NORMAL
    assert manager.alarm.is_playing is False
    assert manager.cooldown_start_time is None