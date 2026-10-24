"""Kiểm thử tích hợp hai nhánh và gap policy của T0704."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
import yaml

from src.features.measurement_adapter import measure_eye, measure_mouth
from src.temporal.temporal_analyzer import TemporalAnalyzer
from src.utils.config import load_config, load_temporal_settings
from src.utils.contracts import (
    EyeMeasurement,
    MouthMeasurement,
    TemporalAnalysisResult,
    TemporalBehaviorState,
    unknown_eye,
    unknown_mouth,
)


@pytest.fixture
def config():
    """Cấu hình thử xác định; không sửa YAML của user.

    Parameters: Không có.
    Returns: Mapping fixture.
    Raises: Không có.
    """
    return {
        "eye": {"ear_threshold": 0.25, "closure_time_threshold": 1.0},
        "mouth": {
            "mar_threshold": 0.6,
            "yawn_duration_threshold": 1.0,
            "yawn_frequency_window": 2.0,
        },
        "temporal": {
            "missing_face_tolerance_frames": 2,
            "missing_face_timeout_sec": 1.0,
        },
    }


@pytest.fixture
def make_analyzer(tmp_path, config):
    """Factory đọc YAML tạm bằng utility thật.

    Parameters: tmp_path và config từ pytest.
    Returns: Factory tạo coordinator.
    Raises: ValueError nếu cấu hình fixture sai.
    """

    def factory(changes=None):
        """Ghi cấu hình thử và khởi tạo.

        Parameters: changes là mapping thay section hoặc None.
        Returns: TemporalAnalyzer.
        Raises: ValueError nếu cấu hình không hợp lệ.
        """
        data = deepcopy(config)
        if changes:
            for section, values in changes.items():
                data[section].update(values)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return TemporalAnalyzer(path)

    return factory


def frame(analyzer, timestamp, eye="OPEN", mouth="MOUTH_CLOSED", face=True):
    """Tạo phép đo synthetic và gửi qua API thật.

    Parameters: analyzer, timestamp, trạng thái hai nhánh và face.
    Returns: TemporalAnalysisResult.
    Raises: ValueError nếu timestamp hoặc schema sai.
    """
    ear = 0.1 if eye == "CLOSED" else 0.4
    mar = 0.9 if mouth == "MOUTH_OPEN" else 0.1
    return analyzer.update(
        face,
        timestamp,
        EyeMeasurement(
            left_ear=ear, right_ear=ear, avg_ear=ear, eye_state=eye
        ),
        MouthMeasurement(mar=mar, mouth_state=mouth),
    )


def test_normal_blink(make_analyzer):
    """Chớp ngắn không phải nhắm kéo dài.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu sai hành vi.
    """
    a = make_analyzer()
    assert isinstance(frame(a, 0), TemporalAnalysisResult)
    frame(a, 0.1, "CLOSED")
    frame(a, 0.2, "CLOSED")
    result = frame(a, 0.3)
    assert result.behavior.blink_count == 1
    assert not result.behavior.is_prolonged_closure
    assert frame(a, 0.4).behavior.blink_count == 1


@pytest.mark.parametrize(
    "duration,expected", [(0.9, False), (1, True), (1.1, True)]
)
def test_closure_boundaries(make_analyzer, duration, expected):
    """Nhắm mắt được xác nhận tại biên YAML.

    Parameters: Fixture, duration và expected.
    Returns: Không có.
    Raises: AssertionError nếu sai biên.
    """
    a = make_analyzer()
    assert not frame(a, 0, "CLOSED").behavior.is_prolonged_closure
    result = frame(a, duration, "CLOSED")
    assert result.behavior.is_prolonged_closure is expected
    assert result.behavior.consecutive_closed_frames == 2
    assert result.behavior.eye_closed_duration == pytest.approx(duration)


def test_single_frame_and_zero_threshold(make_analyzer):
    """Frame đơn và timestamp trùng không tạo sự kiện kéo dài.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu vi phạm guardrail.
    """
    a = make_analyzer()
    frame(a, 0, "CLOSED")
    assert not frame(a, 10).behavior.is_prolonged_closure
    a = make_analyzer(
        {
            "eye": {"closure_time_threshold": 0},
            "mouth": {"yawn_duration_threshold": 0},
        }
    )
    r = frame(a, 0, "CLOSED", "MOUTH_OPEN")
    assert not r.behavior.is_prolonged_closure
    assert not r.behavior.is_yawning_event
    r = frame(a, 0, "CLOSED", "MOUTH_OPEN")
    assert not r.behavior.is_prolonged_closure
    assert not r.behavior.is_yawning_event
    r = frame(a, 0.1, "CLOSED", "MOUTH_OPEN")
    assert r.behavior.is_prolonged_closure
    assert r.behavior.is_yawning_event


def test_speech_yawn_window(make_analyzer):
    """Mở ngắn không ngáp; mỗi chu kỳ chỉ đếm một lần, window độc lập.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu đếm sai.
    """
    a = make_analyzer()
    frame(a, 0, mouth="MOUTH_OPEN")
    assert not frame(a, 0.2, mouth="MOUTH_OPEN").behavior.is_yawning_event
    assert frame(a, 0.3).behavior.yawn_count == 0
    frame(a, 1, mouth="MOUTH_OPEN")
    r = frame(a, 2, mouth="MOUTH_OPEN")
    assert r.behavior.is_yawning_event
    assert r.behavior.yawn_count == 1
    assert not frame(a, 2.1, mouth="MOUTH_OPEN").behavior.is_yawning_event
    frame(a, 2.2)
    frame(a, 2.3, mouth="MOUTH_OPEN")
    assert frame(a, 3.4, mouth="MOUTH_OPEN").behavior.yawn_count == 2
    assert frame(a, 4).behavior.yawn_frequency_in_window == 2
    assert frame(a, 4.1).behavior.yawn_frequency_in_window == 1
    r = frame(a, 5.5)
    assert r.behavior.yawn_frequency_in_window == 0
    assert r.behavior.yawn_count == 2


def test_hold_excludes_gap_and_suppresses_events(make_analyzer):
    """Hold giữ counter, UNKNOWN ngay và loại toàn bộ gap khỏi timer.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu hold tích lũy bằng chứng lỗi.
    """
    a = make_analyzer()
    frame(a, 0, "CLOSED", "MOUTH_OPEN")
    frame(a, 0.5, "CLOSED", "MOUTH_OPEN")
    for t in (0.6, 0.7):
        r = frame(a, t, face=False)
        assert r.action == "HOLD" and not r.is_valid
        assert r.eye.eye_state == r.mouth.mouth_state == "UNKNOWN"
        assert r.behavior.consecutive_closed_frames == 2
        assert not r.behavior.is_prolonged_closure
        assert not r.behavior.is_yawning_event
    r = frame(a, 0.9, "CLOSED", "MOUTH_OPEN")
    assert r.behavior.eye_closed_duration == pytest.approx(0.5)
    assert r.behavior.mouth_open_duration == pytest.approx(0.5)
    r = frame(a, 1.5, "CLOSED", "MOUTH_OPEN")
    assert r.behavior.is_prolonged_closure and r.behavior.is_yawning_event
    held = frame(a, 1.6, face=False)
    assert held.behavior.yawn_count == 1
    assert not held.behavior.is_yawning_event
    recovered = frame(a, 1.7, "CLOSED", "MOUTH_OPEN")
    assert not recovered.behavior.is_yawning_event


@pytest.mark.parametrize(
    "time,action", [(0.9, "HOLD"), (1, "RESET"), (1.1, "RESET")]
)
def test_seconds_timeout(make_analyzer, time, action):
    """Timeout tại đúng ngưỡng reset toàn bộ buffers.

    Parameters: Fixture, time và action.
    Returns: Không có.
    Raises: AssertionError nếu sai biên timeout.
    """
    a = make_analyzer()
    frame(a, 0, "CLOSED", "MOUTH_OPEN")
    r = frame(a, time, face=False)
    assert r.action == action
    if action == "RESET":
        assert r.behavior == TemporalBehaviorState()
        assert a.eye_tracker.closed_start_time is None
        assert not a.yawn_tracker.yawn_timestamps


def test_frame_overflow_and_late_recovery(make_analyzer):
    """Frame thứ N+1 reset; phục hồi muộn không nối timer cũ.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu buffers cũ còn dùng được.
    """
    a = make_analyzer()
    frame(a, 0, "CLOSED")
    frame(a, 0.1, face=False)
    assert frame(a, 0.2, face=False).action == "HOLD"
    assert frame(a, 0.3, face=False).action == "RESET"
    r = frame(a, 0.4, "CLOSED")
    assert r.action == "RESET" and r.is_valid
    assert r.behavior.consecutive_closed_frames == 1
    a.reset()
    frame(a, 0, "CLOSED")
    frame(a, 0.1, face=False)
    r = frame(a, 2, "CLOSED")
    assert r.action == "RESET"
    assert r.behavior.eye_closed_duration == 0


@pytest.mark.parametrize("invalid", ["face", "eye", "mouth", "inf", "nan"])
def test_invalid_measurement(make_analyzer, invalid):
    """Thiếu face hoặc bất kỳ nhánh lỗi đều UNKNOWN toàn frame.

    Parameters: Fixture và dạng invalid.
    Returns: Không có.
    Raises: AssertionError nếu dữ liệu lỗi tạo sự kiện.
    """
    a = make_analyzer()
    e = EyeMeasurement(
        left_ear=0.1, right_ear=0.1, avg_ear=0.1, eye_state="CLOSED"
    )
    m = MouthMeasurement(mar=0.9, mouth_state="MOUTH_OPEN")
    if invalid == "eye":
        e = unknown_eye()
    elif invalid == "mouth":
        m = unknown_mouth()
    elif invalid in ("inf", "nan"):
        e = e.model_copy(update={"avg_ear": float(invalid)})
    r = a.update(invalid != "face", 0, e, m)
    assert r.action == "RESET" and not r.is_valid
    assert r.eye.eye_state == r.mouth.mouth_state == "UNKNOWN"
    assert r.behavior == TemporalBehaviorState()


@pytest.mark.parametrize(
    "timestamp", [-1, float("nan"), float("inf"), True, "1", None]
)
def test_invalid_timestamp_atomic(make_analyzer, timestamp):
    """Timestamp lỗi không làm biến đổi trạng thái.

    Parameters: Fixture và timestamp sai.
    Returns: Không có.
    Raises: AssertionError nếu validation không nguyên tử.
    """
    a = make_analyzer()
    frame(a, 0, "CLOSED")
    before = deepcopy(a.__dict__)
    with pytest.raises(ValueError):
        frame(a, timestamp, "CLOSED")
    assert a._behavior == before["_behavior"]
    assert a._last_timestamp == before["_last_timestamp"]
    assert a.eye_tracker.consecutive_closed_frames == 1


def test_reset_and_default_config(make_analyzer, monkeypatch, tmp_path):
    """Reset idempotent và YAML mặc định không phụ thuộc cwd.

    Parameters: Fixtures make_analyzer, monkeypatch, tmp_path.
    Returns: Không có.
    Raises: AssertionError nếu config/reset sai.
    """
    a = make_analyzer()
    frame(a, 0, "CLOSED", "MOUTH_OPEN")
    frame(a, 1, "CLOSED", "MOUTH_OPEN")
    a.reset()
    a.reset()
    assert frame(a, 0).behavior == TemporalBehaviorState()
    assert not a.yawn_tracker.yawn_timestamps
    monkeypatch.chdir(tmp_path)
    settings = load_temporal_settings()
    assert (
        settings.yawn_frequency_window
        == load_config()["mouth"]["yawn_frequency_window"]
    )
    assert TemporalAnalyzer().settings == settings


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("eye", "closure_time_threshold", -1),
        ("mouth", "yawn_frequency_window", 0),
        ("mouth", "yawn_duration_threshold", float("inf")),
        ("temporal", "missing_face_tolerance_frames", 2.5),
        ("temporal", "missing_face_tolerance_frames", True),
        ("temporal", "missing_face_timeout_sec", "1.5"),
        ("temporal", "missing_face_timeout_sec", float("nan")),
    ],
)
def test_bad_configuration(make_analyzer, section, key, value):
    """Cấu hình sai dừng ngay với tên trường cụ thể.

    Parameters: Fixture, section/key/value.
    Returns: Không có.
    Raises: AssertionError nếu ngưỡng lỗi được chấp nhận.
    """
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        make_analyzer({section: {key: value}})


def test_missing_configuration(tmp_path, config):
    """Thiếu trường không dùng ngưỡng dự phòng.

    Parameters: tmp_path và config fixtures.
    Returns: Không có.
    Raises: AssertionError nếu lỗi không rõ tên trường.
    """
    del config["mouth"]["yawn_frequency_window"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="mouth.yawn_frequency_window"):
        TemporalAnalyzer(path)


def test_geometry_to_coordinator(make_analyzer):
    """Hình học EAR hai mắt thật và MAR đi xuyên cả hai tracker.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu adapter hoặc integration sai.
    """
    a = make_analyzer()
    left = [(0, 0), (1, 0.1), (3, 0.1), (4, 0), (3, -0.1), (1, -0.1)]
    right = [(0, 0), (1, 0.2), (3, 0.2), (4, 0), (3, -0.2), (1, -0.2)]
    mouth = [(0, 0), (1, 1), (2, 1), (3, 1), (4, 0), (3, -1), (2, -1), (1, -1)]
    e = measure_eye(left, right, a.settings)
    m = measure_mouth(mouth, a.settings)
    assert e.left_ear != e.right_ear
    assert e.avg_ear == pytest.approx((e.left_ear + e.right_ear) / 2)
    assert e.eye_state == "CLOSED" and m.mouth_state == "MOUTH_OPEN"
    a.update(True, 0, e, m)
    r = a.update(True, 1, e, m)
    assert r.behavior.is_prolonged_closure and r.behavior.is_yawning_event
    assert not measure_eye(None, right, a.settings).is_valid
    assert not measure_mouth(None, a.settings).is_valid
    assert not measure_mouth(mouth, a.settings, False).is_valid


def test_yaml_changes_behavior(make_analyzer):
    """Thay ngưỡng YAML tác động cả nhánh mắt và miệng.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu tracker dùng config khác.
    """
    a = make_analyzer(
        {
            "eye": {"closure_time_threshold": 2},
            "mouth": {"yawn_duration_threshold": 2},
        }
    )
    frame(a, 0, "CLOSED", "MOUTH_OPEN")
    r = frame(a, 1, "CLOSED", "MOUTH_OPEN")
    assert not r.behavior.is_prolonged_closure
    assert not r.behavior.is_yawning_event
    r = frame(a, 2, "CLOSED", "MOUTH_OPEN")
    assert r.behavior.is_prolonged_closure and r.behavior.is_yawning_event


def test_window_ages_during_hold(make_analyzer):
    """Window dùng thời gian gốc trong HOLD và không phát lại sự kiện.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu window bị đóng băng theo timer.
    """
    a = make_analyzer({"mouth": {"yawn_frequency_window": 0.2}})
    frame(a, 0, mouth="MOUTH_OPEN")
    frame(a, 1, mouth="MOUTH_OPEN")
    held = frame(a, 1.3, face=False)
    assert held.action == "HOLD"
    assert held.behavior.yawn_count == 1
    assert held.behavior.yawn_frequency_in_window == 0
    assert not held.behavior.is_yawning_event
    recovered = frame(a, 1.4, mouth="MOUTH_OPEN")
    assert recovered.behavior.yawn_count == 1
    assert recovered.behavior.mouth_open_duration == pytest.approx(1)
    assert not recovered.behavior.is_yawning_event


def test_zero_tolerance(make_analyzer):
    """Không cho phép giữ frame thiếu khi dung sai YAML bằng zero.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu vẫn HOLD.
    """
    a = make_analyzer({"temporal": {"missing_face_tolerance_frames": 0}})
    frame(a, 0)
    assert frame(a, 0.1, face=False).action == "RESET"


@pytest.mark.parametrize("failure", [None, "initialization", "iteration"])
def test_caller_cleanup(monkeypatch, make_analyzer, failure):
    """Demo thật cleanup ở EOF, lỗi khởi tạo và lỗi giữa vòng lặp.

    Parameters: Fixtures và failure chọn lỗi synthetic.
    Returns: Không có.
    Raises: AssertionError nếu tài nguyên không được đóng.
    """
    from scripts import test_detection_face_detected as demo

    analyzer = make_analyzer()
    reset = MagicMock(wraps=analyzer.reset)
    monkeypatch.setattr(analyzer, "reset", reset)
    monkeypatch.setattr(demo, "TemporalAnalyzer", lambda: analyzer)
    detector = MagicMock()
    monkeypatch.setattr(demo, "FaceDetector", MagicMock(return_value=detector))
    landmarks = MagicMock()
    landmark_factory = MagicMock(return_value=landmarks)
    if failure == "initialization":
        landmark_factory.side_effect = RuntimeError("initialization")
    monkeypatch.setattr(demo, "LandmarkDetector", landmark_factory)
    stream = MagicMock()
    source = MagicMock()
    stream.__enter__.return_value = source
    if failure == "iteration":
        source.__iter__.side_effect = RuntimeError("iteration")
    else:
        source.__iter__.return_value = iter([])
    monkeypatch.setattr(demo, "VideoReader", MagicMock(return_value=stream))
    windows = MagicMock()
    monkeypatch.setattr(demo.cv2, "destroyAllWindows", windows)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            demo.run_pipeline(False, "synthetic.mp4")
    else:
        demo.run_pipeline(False, "synthetic.mp4")
    reset.assert_called_once()
    detector.release.assert_called_once()
    windows.assert_called_once()
    if failure != "initialization":
        landmarks.release.assert_called_once()
        stream.__exit__.assert_called_once()


def test_type_contract_and_snapshot_isolation(make_analyzer):
    """Không nhận loose dict và snapshot không sửa được buffers bên trong.

    Parameters: make_analyzer fixture.
    Returns: Không có.
    Raises: AssertionError nếu sai hợp đồng hoặc snapshot dùng chung.
    """
    a = make_analyzer()
    with pytest.raises(TypeError):
        a.update(True, 0, {}, unknown_mouth())
    with pytest.raises(TypeError):
        a.update(1, 0, unknown_eye(), unknown_mouth())
    result = frame(a, 0, "CLOSED")
    result.behavior.blink_count = 999
    assert frame(a, 0.1, "CLOSED").behavior.blink_count == 0
