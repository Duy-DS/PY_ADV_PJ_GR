"""Hợp đồng Pydantic v2 cho phép đo và hành vi T0704."""

from typing import Literal

from pydantic import BaseModel, Field


class EyeMeasurement(BaseModel):
    """Phép đo mắt theo AGENTS.md.

    Parameters: EAR hai mắt, trung bình, trạng thái và is_valid.
    Returns: Phép đo typed.
    Raises: ValueError nếu dữ liệu không khớp schema.
    """

    left_ear: float = Field(..., ge=0.0)
    right_ear: float = Field(..., ge=0.0)
    avg_ear: float = Field(..., ge=0.0)
    eye_state: Literal["OPEN", "CLOSED", "UNKNOWN"]
    is_valid: bool = True


class MouthMeasurement(BaseModel):
    """Phép đo miệng theo AGENTS.md.

    Parameters: MAR, trạng thái và is_valid.
    Returns: Phép đo typed.
    Raises: ValueError nếu dữ liệu không khớp schema.
    """

    mar: float = Field(..., ge=0.0)
    mouth_state: Literal["MOUTH_OPEN", "MOUTH_CLOSED", "UNKNOWN"]
    is_valid: bool = True


class TemporalBehaviorState(BaseModel):
    """Snapshot hành vi không chứa quyết định buồn ngủ.

    Parameters: Thời lượng, bộ đếm và cờ theo AGENTS.md.
    Returns: Hành vi typed.
    Raises: ValueError nếu thời lượng hoặc bộ đếm âm.
    """

    eye_closed_duration: float = Field(default=0.0, ge=0.0)
    consecutive_closed_frames: int = Field(default=0, ge=0)
    is_prolonged_closure: bool = False
    blink_count: int = Field(default=0, ge=0)
    mouth_open_duration: float = Field(default=0.0, ge=0.0)
    yawn_count: int = Field(default=0, ge=0)
    yawn_frequency_in_window: float = Field(default=0.0, ge=0.0)
    is_yawning_event: bool = False


class TemporalAnalysisResult(BaseModel):
    """Bao phép đo hiện tại, hành vi và metadata khoảng mất dữ liệu.

    Parameters: eye, mouth, behavior, action, validity và số frame thiếu.
    Returns: Kết quả typed; consumer phải kiểm tra is_valid.
    Raises: ValueError nếu dữ liệu không khớp schema.
    """

    eye: EyeMeasurement
    mouth: MouthMeasurement
    behavior: TemporalBehaviorState
    action: Literal["CONTINUE", "HOLD", "RESET"]
    is_valid: bool
    consecutive_missing_frames: int = Field(default=0, ge=0)


def unknown_eye() -> EyeMeasurement:
    """Tạo phép đo mắt không khả dụng.

    Parameters: Không có.
    Returns: UNKNOWN với is_valid=False; số không không phải bằng chứng.
    Raises: Không có.
    """
    return EyeMeasurement(
        left_ear=0.0,
        right_ear=0.0,
        avg_ear=0.0,
        eye_state="UNKNOWN",
        is_valid=False,
    )


def unknown_mouth() -> MouthMeasurement:
    """Tạo phép đo miệng không khả dụng.

    Parameters: Không có.
    Returns: UNKNOWN với is_valid=False.
    Raises: Không có.
    """
    return MouthMeasurement(mar=0.0, mouth_state="UNKNOWN", is_valid=False)
