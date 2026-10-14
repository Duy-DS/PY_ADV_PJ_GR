import math

def check_eye_closure_rule(temporal_data: dict) -> bool:
    """
    Luật A: Kiểm tra xem có xảy ra hiện tượng nhắm mắt kéo dài hay không.
    """
    if not temporal_data:
        return False
    return bool(temporal_data.get("prolonged_eye_closure", False))


def check_yawning_rule(temporal_data: dict, yawn_frequency_threshold: float) -> bool:
    """
    Luật B: Kiểm tra xem tần suất ngáp có vượt ngưỡng cho phép trong cửa sổ thời gian hay không.
    """
    if not temporal_data:
        return False
    
    yawn_frequency = temporal_data.get("yawn_frequency", 0)
    try:
        yawn_freq_val = float(yawn_frequency)
        threshold_val = float(yawn_frequency_threshold)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(yawn_freq_val) or not math.isfinite(threshold_val):
        return False

    return yawn_freq_val >= threshold_val


def evaluate_rules(temporal_data: dict, mode: str, yawn_frequency_threshold: float) -> tuple:
    """
    Hàm tổng hợp đánh giá luật dựa trên chế độ (Mode A hoặc Mode B).
    Trả về: (is_drowsy: bool, reasons: list)
    """
    if not temporal_data or not temporal_data.get("face_detected", True):
        return False, ["Face not detected"]

    reasons = []
    is_drowsy = False

    # Kiểm tra luật mắt (áp dụng cho cả Mode A và Mode B)
    if check_eye_closure_rule(temporal_data):
        is_drowsy = True
        reasons.append("Prolonged eye closure")

    # Kiểm tra luật ngáp (chỉ áp dụng nếu mode là 'eye_yawn')
    if mode == "eye_yawn":
        if check_yawning_rule(temporal_data, yawn_frequency_threshold):
            is_drowsy = True
            reasons.append(f"High yawn frequency (>= {yawn_frequency_threshold})")

    return is_drowsy, reasons