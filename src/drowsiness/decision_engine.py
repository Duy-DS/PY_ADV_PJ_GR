import math
from typing import Any
from src.utils.config import load_config
from src.drowsiness.rule_based import evaluate_rules


class DecisionEngine:
    """
    Bộ điều phối quyết định trạng thái buồn ngủ (Drowsiness Decision Engine)
    sử dụng các quy tắc được định nghĩa trong rule_based.py.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Nạp cấu hình được truyền vào, giữ API mặc định cho caller cũ.

        Parameters: config là mapping từ load_config hoặc None.
        Returns: Không có.
        Raises: ValueError nếu mode hoặc ngưỡng ngáp không hợp lệ.
        """
        if config is None:
            config = load_config()

        # 1. Đọc chế độ từ YAML: "eye_only" (Mode A) hoặc "eye_yawn" (Mode B)
        drowsiness_config = config.get("drowsiness", {})
        self.mode = drowsiness_config.get("mode", "eye_yawn")

        # 2. Đọc ngưỡng tần suất ngáp từ YAML
        mouth_config = config.get("mouth", {})
        self.yawn_frequency_threshold = float(
            mouth_config.get("yawn_frequency_threshold", 3.0)
        )

        # Kiểm tra tính hợp lệ
        if self.mode not in ["eye_only", "eye_yawn"]:
            raise ValueError(
                "drowsiness.mode must be either 'eye_only' or 'eye_yawn'"
            )

        if (
            not math.isfinite(self.yawn_frequency_threshold)
            or self.yawn_frequency_threshold < 0
        ):
            raise ValueError(
                "mouth.yawn_frequency_threshold must be a non-negative number"
            )

    def evaluate(self, temporal_data: dict) -> dict:
        """
        Đánh giá trạng thái buồn ngủ từ kết quả temporal analyzer thông qua
        rule engine.
        """
        # Nếu mất dấu khuôn mặt
        if not temporal_data or not temporal_data.get("face_detected", True):
            return {
                "state": "UNKNOWN",
                "reason": "Face not detected or insufficient temporal data",
            }

        # Gọi module rule_based để đánh giá
        is_drowsy, reasons = evaluate_rules(
            temporal_data=temporal_data,
            mode=self.mode,
            yawn_frequency_threshold=self.yawn_frequency_threshold,
        )

        if is_drowsy:
            mode_label = (
                "Mode A (Eye-only)"
                if self.mode == "eye_only"
                else "Mode B (Eye + Yawning)"
            )
            return {
                "state": "DROWSY",
                "reason": " & ".join(reasons) + f" [{mode_label}]",
            }
        else:
            return {"state": "NON_DROWSY", "reason": "Normal behavior"}
