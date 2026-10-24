"""Adapter typed cho hình học mắt/miệng, dùng cùng cấu hình coordinator."""

import math

from src.features.eye import calculate_ear, classify_eye_state
from src.features.mouth import calculate_mar, classify_mouth_state
from src.utils.config import TemporalSettings
from src.utils.contracts import (
    EyeMeasurement,
    MouthMeasurement,
    unknown_eye,
    unknown_mouth,
)
from src.utils.geometry import euclidean_distance_2d


def _points_valid(points, count, opposite):
    """Kiểm tra số điểm, tọa độ hữu hạn và chiều ngang khác zero.

    Parameters: points, số count và chỉ số opposite.
    Returns: bool.
    Raises: Không có; dữ liệu hình học lỗi trả False.
    """
    try:
        return (
            len(points) == count
            and all(
                len(p) == 2 and all(math.isfinite(v) for v in p)
                for p in points
            )
            and euclidean_distance_2d(points[0], points[opposite]) > 0
        )
    except (TypeError, ValueError, OverflowError):
        return False


def measure_eye(left_points, right_points, settings: TemporalSettings):
    """Tính EAR thực của hai mắt và phân loại bằng YAML.

    Parameters: left_points, right_points và settings typed.
    Returns: EyeMeasurement; UNKNOWN nếu hình học lỗi.
    Raises: Không có đối với dữ liệu landmark lỗi.
    """
    if not (
        _points_valid(left_points, 6, 3) and _points_valid(right_points, 6, 3)
    ):
        return unknown_eye()
    left = calculate_ear(left_points)
    right = calculate_ear(right_points)
    average = (left + right) / 2
    if not all(math.isfinite(v) for v in (left, right, average)):
        return unknown_eye()
    return EyeMeasurement(
        left_ear=left,
        right_ear=right,
        avg_ear=average,
        eye_state=classify_eye_state(average, settings.ear_threshold),
    )


def measure_mouth(points, settings: TemporalSettings, landmark_valid=True):
    """Tính MAR và phân loại bằng ngưỡng đã kiểm tra từ YAML.

    Parameters: points, settings typed và landmark_valid.
    Returns: MouthMeasurement; UNKNOWN nếu hình học lỗi.
    Raises: Không có đối với landmark lỗi.
    """
    if not landmark_valid or not _points_valid(points, 8, 4):
        return unknown_mouth()
    mar = calculate_mar(points)
    if not math.isfinite(mar):
        return unknown_mouth()
    return MouthMeasurement(
        mar=mar, mouth_state=classify_mouth_state(mar, settings.mar_threshold)
    )
