import math

from src.utils.geometry import euclidean_distance_2d
from src.utils.config import load_config

# Tính Eye Aspect Ratio (EAR) cho một mắt
def calculate_ear(eye_landmarks):
    if len(eye_landmarks) != 6:
        raise ValueError("eye_landmarks must contain exactly 6 points")

    # Danh sách 6 điểm landmark của mắt theo thứ tự
    p1, p2, p3, p4, p5, p6 = eye_landmarks

    # Tính khoảng cách Euclidean giữa các điểm landmark
    vertical_1 = euclidean_distance_2d(p2, p6)
    vertical_2 = euclidean_distance_2d(p3, p5)
    horizontal = euclidean_distance_2d(p1, p4)

    if horizontal == 0:
        return 0.0

    # Tính EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return float(ear)


# Phân loại trạng thái mắt dựa trên EAR
def classify_eye_state(ear, ear_threshold):
    if ear is None or not math.isfinite(ear):
        return "UNKNOWN"

    if ear >= ear_threshold:
        return "OPEN"

    return "CLOSED"


#  Tính EAR trung bình của hai mắt và phân loại trạng thái mắt
def calculate_eye_state(left_eye_points, right_eye_points):
    # Đọc ngưỡng EAR từ tệp cấu hình YAML
    config = load_config()
    ear_threshold = float(config["eye"]["ear_threshold"])

    if not math.isfinite(ear_threshold):
        raise ValueError("eye.ear_threshold must be a finite number")

    normalized_eyes = []

    # Chuẩn hóa và kiểm tra dữ liệu landmark của hai mắt
    for eye_points in (left_eye_points, right_eye_points):
        try:
            # Mỗi mắt có đúng 6 landmark, mỗi landmark gồm 2 tọa độ
            if len(eye_points) != 6:
                return {"ear": 0.0, "eye_state": "UNKNOWN"}

            normalized_points = []

            for point in eye_points:
                if len(point) != 2:
                    return {"ear": 0.0, "eye_state": "UNKNOWN"}

                # Chuyển tọa độ sang kiểu float
                coordinates = tuple(float(value) for value in point)

                # Kiểm tra tọa độ có phải số hữu hạn hay không
                if not all(math.isfinite(value) for value in coordinates):
                    return {"ear": 0.0, "eye_state": "UNKNOWN"}

                normalized_points.append(coordinates)

        except (TypeError, ValueError, OverflowError):
            return {"ear": 0.0, "eye_state": "UNKNOWN"}

        # Khoảng cách ngang bằng 0 sẽ làm công thức EAR bị chia cho 0
        if euclidean_distance_2d(normalized_points[0], normalized_points[3]) == 0:
            return {"ear": 0.0, "eye_state": "UNKNOWN"}

        normalized_eyes.append(normalized_points)

    # Tính EAR riêng cho mắt trái và mắt phải
    left_ear = calculate_ear(normalized_eyes[0])
    right_ear = calculate_ear(normalized_eyes[1])

    # Tính EAR trung bình của hai mắt
    average_ear = (left_ear + right_ear) / 2.0

    # Phân loại trạng thái mắt dựa trên ngưỡng EAR
    eye_state = classify_eye_state(average_ear, ear_threshold)

    return {
        "ear": float(average_ear),
        "eye_state": eye_state
    }

class EyeFeatureExtractor:
    def __init__(self, ear_threshold=None):
        self.ear_threshold = ear_threshold
    def compute(self, left_pts, right_pts):
        return calculate_eye_state(left_pts, right_pts)
