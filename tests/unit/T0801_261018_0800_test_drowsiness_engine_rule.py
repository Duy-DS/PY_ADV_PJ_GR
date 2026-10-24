from unittest.mock import patch
from src.drowsiness.decision_engine import DecisionEngine
from src.drowsiness.rule_based import (
    check_eye_closure_rule,
    check_yawning_rule,
)


def test_rule_based_eye_closure():
    """Kiểm tra luật nhận diện nhắm mắt kéo dài."""
    assert check_eye_closure_rule({"prolonged_eye_closure": True}) is True
    assert check_eye_closure_rule({"prolonged_eye_closure": False}) is False
    assert check_eye_closure_rule({}) is False


def test_rule_based_yawning():
    """Kiểm tra luật tần suất ngáp so với ngưỡng."""
    assert (
        check_yawning_rule({"yawn_frequency": 3}, yawn_frequency_threshold=3.0)
        is True
    )
    assert (
        check_yawning_rule({"yawn_frequency": 2}, yawn_frequency_threshold=3.0)
        is False
    )
    assert check_yawning_rule({}, yawn_frequency_threshold=3.0) is False


@patch("src.drowsiness.decision_engine.load_config")
def test_decision_engine_mode_a_eye_only(mock_load_config):
    """Kiểm tra Decision Engine ở Mode A (eye_only)."""
    mock_load_config.return_value = {
        "drowsiness": {"mode": "eye_only"},
        "mouth": {"yawn_frequency_threshold": 3.0},
    }
    engine = DecisionEngine()

    # Kịch bản 1: Mắt bình thường -> NON_DROWSY
    res = engine.evaluate(
        {
            "face_detected": True,
            "prolonged_eye_closure": False,
            "yawn_frequency": 5,
        }
    )
    assert res["state"] == "NON_DROWSY"

    # Kịch bản 2: Nhắm mắt kéo dài -> DROWSY
    res = engine.evaluate(
        {
            "face_detected": True,
            "prolonged_eye_closure": True,
            "yawn_frequency": 0,
        }
    )
    assert res["state"] == "DROWSY"

    # Kịch bản 3: Ngáp nhiều nhưng ở Mode A nên không quan tâm -> Vẫn
    # NON_DROWSY
    res = engine.evaluate(
        {
            "face_detected": True,
            "prolonged_eye_closure": False,
            "yawn_frequency": 10,
        }
    )
    assert res["state"] == "NON_DROWSY"


@patch("src.drowsiness.decision_engine.load_config")
def test_decision_engine_mode_b_eye_yawn(mock_load_config):
    """Kiểm tra Decision Engine ở Mode B (eye_yawn)."""
    mock_load_config.return_value = {
        "drowsiness": {"mode": "eye_yawn"},
        "mouth": {"yawn_frequency_threshold": 3.0},
    }
    engine = DecisionEngine()

    # Kịch bản 1: Cả mắt và ngáp đều bình thường -> NON_DROWSY
    res = engine.evaluate(
        {
            "face_detected": True,
            "prolonged_eye_closure": False,
            "yawn_frequency": 1,
        }
    )
    assert res["state"] == "NON_DROWSY"

    # Kịch bản 2: Ngáp đạt ngưỡng trong time window -> DROWSY
    res = engine.evaluate(
        {
            "face_detected": True,
            "prolonged_eye_closure": False,
            "yawn_frequency": 3,
        }
    )
    assert res["state"] == "DROWSY"

    # Kịch bản 3: Mất dấu khuôn mặt -> UNKNOWN
    res = engine.evaluate({"face_detected": False})
    assert res["state"] == "UNKNOWN"
