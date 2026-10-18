import math

from src.utils.geometry import euclidean_distance_2d
from src.utils.config import load_config


def calculate_mar(mouth_landmarks):
    """
    Tính Mouth Aspect Ratio (MAR) cho vùng miệng.
    Yêu cầu mouth_landmarks chứa đúng 8 điểm theo thứ tự chuẩn:
    - p1 (mép trái), p5 (mép phải) -> xác định chiều ngang
    - (p2, p8), (p3, p7), (p4, p6) -> 3 cặp điểm môi trên/môi dưới xác định
    chiều dọc
    """
    if len(mouth_landmarks) != 8:
        raise ValueError("mouth_landmarks must contain exactly 8 points")

    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_landmarks

    # Tính khoảng cách dọc (Vertical distances)
    vertical_1 = euclidean_distance_2d(p2, p8)
    vertical_2 = euclidean_distance_2d(p3, p7)
    vertical_3 = euclidean_distance_2d(p4, p6)

    # Tính khoảng cách ngang giữa 2 mép miệng (Horizontal distance)
    horizontal = euclidean_distance_2d(p1, p5)

    if horizontal == 0:
        return 0.0

    # Công thức MAR chuẩn
    mar = (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)

    return float(mar)


def classify_mouth_state(mar, mar_threshold):
    """Phân loại trạng thái miệng dựa trên MAR và mar_threshold."""
    if mar is None or not math.isfinite(mar):
        return "UNKNOWN"

    if mar >= mar_threshold:
        return "MOUTH_OPEN"

    return "MOUTH_CLOSED"


def calculate_mouth_state(mouth_points, landmark_valid=True):
    """
    Tính MAR và phân loại trạng thái miệng cấp frame.
    Trả về dictionary theo đúng hợp đồng: {"mar": float, "mouth_state": str}
    """
    # Nếu cờ landmark bị lỗi từ bước detector
    if not landmark_valid or mouth_points is None:
        return {"mar": 0.0, "mouth_state": "UNKNOWN"}

    # Đọc ngưỡng MAR trực tiếp từ tệp cấu hình YAML (key: mouth ->
    # mar_threshold)
    config = load_config()
    try:
        mar_threshold = float(config["mouth"]["mar_threshold"])
    except (KeyError, TypeError, ValueError):
        mar_threshold = 0.6  # Giá trị dự phòng nếu lỗi đọc config

    if not math.isfinite(mar_threshold):
        raise ValueError("mouth.mar_threshold must be a finite number")

    # Chuẩn hóa và kiểm tra dữ liệu landmark của miệng (cần đúng 8 điểm)
    try:
        if len(mouth_points) != 8:
            return {"mar": 0.0, "mouth_state": "UNKNOWN"}

        normalized_points = []
        for point in mouth_points:
            if len(point) != 2:
                return {"mar": 0.0, "mouth_state": "UNKNOWN"}

            coordinates = tuple(float(value) for value in point)

            if not all(math.isfinite(value) for value in coordinates):
                return {"mar": 0.0, "mouth_state": "UNKNOWN"}

            normalized_points.append(coordinates)

    except (TypeError, ValueError, OverflowError):
        return {"mar": 0.0, "mouth_state": "UNKNOWN"}

    # Tránh chia cho 0 nếu chiều ngang miệng bằng 0
    if euclidean_distance_2d(normalized_points[0], normalized_points[4]) == 0:
        return {"mar": 0.0, "mouth_state": "UNKNOWN"}

    # Tính chỉ số MAR
    mar = calculate_mar(normalized_points)

    # Phân loại trạng thái MOUTH_OPEN / MOUTH_CLOSED / UNKNOWN
    mouth_state = classify_mouth_state(mar, mar_threshold)

    return {"mar": float(mar), "mouth_state": mouth_state}


class MouthFeatureExtractor:
    def __init__(self, mar_threshold=None):
        self.mar_threshold = mar_threshold

    def compute(self, mouth_pts, landmark_valid=True):
        return calculate_mouth_state(mouth_pts, landmark_valid=landmark_valid)
