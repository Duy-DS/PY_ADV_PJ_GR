import math


def check_eye_closure_rule(temporal_data: dict) -> bool:
    """Luật A: Kiểm tra xem có xảy ra hiện tượng nhắm mắt kéo dài hay không."""
    if not temporal_data:
        return False
    # Truy xuất qua behavior
    behavior = temporal_data.get("behavior", {})
    return bool(behavior.get("is_prolonged_closure", False))


def check_yawning_rule(temporal_data: dict, yawn_frequency_threshold: float) -> bool:
    """Luật B: Kiểm tra xem tần suất ngáp có vượt ngưỡng cho phép trong cửa sổ."""
    if not temporal_data:
        return False

    # Truy xuất qua behavior
    behavior = temporal_data.get("behavior", {})
    yawn_frequency = behavior.get("yawn_frequency_in_window", 0.0)

    try:
        yawn_freq_val = float(yawn_frequency)
        threshold_val = float(yawn_frequency_threshold)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(yawn_freq_val) or not math.isfinite(threshold_val):
        return False

    return yawn_freq_val >= threshold_val


def evaluate_rules(
    temporal_data: dict, mode: str, yawn_frequency_threshold: float
) -> tuple:
    """Hàm tổng hợp đánh giá luật dựa trên chế độ."""
    # Kiểm tra key 'is_valid' của Pydantic contract thay vì 'face_detected'
    if not temporal_data or not temporal_data.get("is_valid", True):
        return False, ["Face not detected or invalid temporal data"]

    reasons = []
    is_drowsy = False

    # Kiểm tra luật mắt
    if check_eye_closure_rule(temporal_data):
        is_drowsy = True
        reasons.append("Prolonged eye closure")

    # Kiểm tra luật ngáp
    if mode == "eye_yawn":
        if check_yawning_rule(temporal_data, yawn_frequency_threshold):
            is_drowsy = True
            reasons.append(f"High yawn frequency (>= {yawn_frequency_threshold})")

    return is_drowsy, reasons
