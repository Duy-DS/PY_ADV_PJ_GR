import pytest

from src.features import eye
from src.utils.config import load_config


# Tạo dữ liệu landmark mắt giả lập với EAR mong muốn
# Dùng để kiểm thử các hàm tính EAR và trạng thái mắt
def make_eye_landmarks(ear):
    vertical_distance = 4 * ear
    half_vertical_distance = vertical_distance / 2
    return [
        (0, 0),
        (1, half_vertical_distance),
        (3, half_vertical_distance),
        (4, 0),
        (3, -half_vertical_distance),
        (1, -half_vertical_distance),
    ]


# Kiểm tra hàm calculate_ear() có tính EAR đúng theo
# công thức Soukupová & Čech hay không
def test_calculate_ear_uses_soukupova_cech_formula():
    landmarks = make_eye_landmarks(0.5)

    # EAR mong đợi bằng 0.5
    assert eye.calculate_ear(landmarks) == pytest.approx(0.5)


# Kiểm tra hàm calculate_eye_state():
# - Tính EAR trung bình của hai mắt
# - Đọc ear_threshold từ cấu hình
# - Phân loại trạng thái mắt dựa trên ngưỡng
def test_calculate_eye_state_averages_both_eyes_and_uses_config(monkeypatch):
    config = load_config()
    config["eye"]["ear_threshold"] = 0.38

    # Thay thế load_config() trong lúc test bằng cấu hình giả lập
    monkeypatch.setattr(eye, "load_config", lambda: config)

    result = eye.calculate_eye_state(
        make_eye_landmarks(0.5),
        make_eye_landmarks(0.25),
    )

    # EAR trung bình = (0.5 + 0.25) / 2 = 0.375
    # 0.375 < 0.38 nên trạng thái là CLOSED
    assert result == {"ear": pytest.approx(0.375), "eye_state": "CLOSED"}


# Kiểm tra trường hợp EAR bằng đúng ngưỡng.
# Theo yêu cầu, EAR >= threshold được phân loại là OPEN.
def test_calculate_eye_state_classifies_threshold_as_open(monkeypatch):
    # Thiết lập threshold giả lập bằng 0.5
    monkeypatch.setattr(
        eye,
        "load_config",
        lambda: {"eye": {"ear_threshold": 0.5}},
    )

    result = eye.calculate_eye_state(
        make_eye_landmarks(0.5),
        make_eye_landmarks(0.5),
    )

    # EAR = threshold nên trạng thái phải là OPEN
    assert result == {"ear": pytest.approx(0.5), "eye_state": "OPEN"}


# Kiểm tra khả năng xử lý các dữ liệu landmark không hợp lệ.
# Test được chạy nhiều lần với các trường hợp khác nhau:
# 1. left_eye = None
# 2. Mắt trái chỉ có 5 landmark
# 3. Mắt phải có dữ liệu không hợp lệ
# 4. Landmark chứa giá trị NaN
@pytest.mark.parametrize(
    ("left_eye", "right_eye"),
    [
        (None, make_eye_landmarks(0.5)),
        ([(0, 0)] * 5, make_eye_landmarks(0.5)),
        (make_eye_landmarks(0.5), [(0, 0)] * 6),
        (
            [(0, 0), (1, float("nan")), (3, 1), (4, 0), (3, -1), (1, -1)],
            make_eye_landmarks(0.5),
        ),
    ],
)
def test_calculate_eye_state_returns_unknown_for_invalid_landmarks(
    monkeypatch, left_eye, right_eye
):
    # Thiết lập threshold giả lập
    monkeypatch.setattr(
        eye,
        "load_config",
        lambda: {"eye": {"ear_threshold": 0.25}},
    )

    result = eye.calculate_eye_state(left_eye, right_eye)

    # Dữ liệu không hợp lệ phải trả về UNKNOWN
    # và EAR được đặt về 0.0
    assert result == {"ear": 0.0, "eye_state": "UNKNOWN"}


# Kiểm tra trường hợp khoảng cách ngang của mắt bằng 0.
# Trường hợp này có thể gây chia cho 0 khi tính EAR,
# vì vậy hàm phải xử lý an toàn và trả về UNKNOWN.
def test_calculate_eye_state_returns_unknown_for_zero_horizontal_distance(
    monkeypatch,
):
    # Thiết lập threshold giả lập
    monkeypatch.setattr(
        eye,
        "load_config",
        lambda: {"eye": {"ear_threshold": 0.25}},
    )

    # Tạo landmark có hai điểm ngang trùng nhau,
    # làm khoảng cách ngang bằng 0
    degenerate_eye = [
        (1, 0),
        (1, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (1, -1),
    ]

    result = eye.calculate_eye_state(
        degenerate_eye,
        make_eye_landmarks(0.5),
    )

    # Dữ liệu không hợp lệ phải trả về UNKNOWN
    assert result == {"ear": 0.0, "eye_state": "UNKNOWN"}