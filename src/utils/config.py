"""Đọc YAML và kiểm tra cấu hình thời gian, không dùng ngưỡng dự phòng."""

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import yaml


def load_config(config_path=None):
    """Đọc YAML độc lập với thư mục làm việc hiện tại.

    Parameters: config_path là đường dẫn YAML hoặc None cho mặc định.
    Returns: Mapping YAML.
    Raises: OSError, yaml.YAMLError hoặc ValueError nếu tệp sai.
    """
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def numeric_setting(config, section, key, *, positive=False, integer=False):
    """Kiểm tra một giá trị cấu hình bắt buộc.

    Parameters: Mapping YAML, tên section/key và ràng buộc positive/integer.
    Returns: float hoặc int đã kiểm tra.
    Raises: ValueError nếu thiếu trường hoặc sai kiểu/miền giá trị.
    """
    name = f"{section}.{key}"
    try:
        value = config[section][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing required configuration: {name}") from exc
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or (positive and value <= 0)
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(f"Invalid numeric configuration: {name}")
    return int(value) if integer else float(value)


@dataclass(frozen=True)
class TemporalSettings:
    """Cấu hình typed dùng chung giữa coordinator và tracker.

    Parameters: Các ngưỡng float và giới hạn frame int từ YAML.
    Returns: Cấu hình bất biến.
    Raises: Không có; load_temporal_settings chịu trách nhiệm kiểm tra.
    """

    ear_threshold: float
    closure_time_threshold: float
    mar_threshold: float
    yawn_duration_threshold: float
    yawn_frequency_window: float
    missing_face_tolerance_frames: int
    missing_face_timeout_sec: float


def load_temporal_settings(config_path=None) -> TemporalSettings:
    """Đọc và kiểm tra đầy đủ cấu hình T0704.

    Parameters: config_path là tệp YAML hoặc None.
    Returns: TemporalSettings không có ngưỡng dự phòng.
    Raises: OSError, ValueError, yaml.YAMLError nếu cấu hình sai.
    """
    config = load_config(config_path)
    return TemporalSettings(
        ear_threshold=numeric_setting(config, "eye", "ear_threshold"),
        closure_time_threshold=numeric_setting(
            config, "eye", "closure_time_threshold"
        ),
        mar_threshold=numeric_setting(config, "mouth", "mar_threshold"),
        yawn_duration_threshold=numeric_setting(
            config, "mouth", "yawn_duration_threshold"
        ),
        yawn_frequency_window=numeric_setting(
            config, "mouth", "yawn_frequency_window", positive=True
        ),
        missing_face_tolerance_frames=numeric_setting(
            config, "temporal", "missing_face_tolerance_frames", integer=True
        ),
        missing_face_timeout_sec=numeric_setting(
            config, "temporal", "missing_face_timeout_sec", positive=True
        ),
    )


def validate_timestamp(timestamp, previous=None) -> float:
    """Kiểm tra thời gian hữu hạn, không lùi; cho phép trùng mốc.

    Parameters: timestamp dạng số, previous là mốc trước hoặc None.
    Returns: float đã kiểm tra.
    Raises: ValueError nếu sai kiểu, vô hạn hoặc thời gian lùi.
    """
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, Real)
        or not math.isfinite(timestamp)
        or (previous is not None and timestamp < previous)
    ):
        raise ValueError(
            "timestamp must be finite, numeric and non-decreasing"
        )
    return float(timestamp)
