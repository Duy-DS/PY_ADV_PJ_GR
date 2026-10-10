# TASK INSTRUCTION FOR CODEX AGENT: T1001 — Core Facial Analysis Prototype Integration (PoC)

## 1. Role & Context
You are the **Lead Computer Vision & Software Engineer (Codex Agent)** on the **`PY_ADV_PJ_GR`** repository (**Driver Drowsiness Detection System**).
You are implementing **Task T1001** (`Core Facial Analysis Prototype Integration - PoC`) under **Work Package WP10 (Pipeline Integration)** to successfully close **Milestone M03 (Facial Analysis Prototype)**.

Your work bridges upstream acquisition and feature extraction modules completed by team members:
- **T0301 (Acquisition)**: `src/acquisition/video_reader.py` (Member A)
- **T0402 (Landmark Detection)**: `src/detection/landmark_detector.py` (Member B)
- **T0501 (Eye Feature)**: `src/features/eye.py` & `src/utils/geometry.py` (Member C)
- **T0601 (Mouth Feature)**: `src/features/mouth.py` (Member D)
- **T0701 (Temporal Eye Closure)**: `src/temporal/eye_closure.py` (Member D)

---

## 2. Objectives & Deliverables
1. **Remove / Clean Up Misplaced Stub**:
   - Delete the misplaced 0-byte file `tests/unit/integration/T1001_test_frame_pipeline.py` and its directory if empty.
2. **Implement Integration Test Suite & Runner**:
   - Create `tests/integration/T1001_261010_1030_test_frame_pipeline.py` adhering to `AGENTS.md` naming standards.
   - The script must support **dual-mode execution**:
     - **Headless Mode (`pytest`)**: Automated regression tests without GUI calls (`cv2.imshow`), verifying end-to-end data contracts, numeric bounds, safety fallbacks, and resource teardown.
     - **Interactive Visual Mode (`CLI / Demo`)**: Invoked via `python tests/integration/T1001_261010_1030_test_frame_pipeline.py --visual [--source <path|camera_id>] [--max-frames <N>]`, rendering a comprehensive on-screen HUD (FPS, latency, EAR, MAR, states, and landmark overlays) with multi-layered exit handling (`q`, `Q`, `ESC`, `Ctrl+C`).
3. **Fulfill System Evaluation & Performance Profiling (`SCri014`)**:
   - Measure frame processing latency (mean and p95 in ms) and effective FPS.
   - Ensure the integrated facial analysis pipeline operates at $\ge 20$ FPS (latency $\le 50$ ms/frame) on standard hardware.

---

## 3. Strict Invariants & Guardrails
- **Invalid Measurement Safety (`FReq004`, `FReq030`)**:
  When `landmark_valid == False` (no face detected or corrupted landmarks), the pipeline MUST immediately yield `eye_state = "UNKNOWN"`, `mouth_state = "UNKNOWN"`.
  **CRITICAL RED LINE**: NEVER classify a missing face or single missing detection as `DROWSY`.
- **Stream & Resource Safety (`FReq032`)**:
  Always manage video captures and detectors using context managers or explicit `try...finally` teardown (`reader.release()`, `detector.release()`, `cv2.destroyAllWindows()`).
- **Configuration Decoupling (`FReq024`)**:
  Operational thresholds (`ear_threshold`, `mar_threshold`, `closure_time_threshold`) must be loaded dynamically via `src/utils/config.py` from `configs/default.yaml`. Do NOT hardcode thresholds.
- **Repository Standards & Coding Style**:
  - 100% PEP 8 compliant, type hints on all function signatures.
  - Comprehensive Vietnamese docstrings (`Mô tả`, `Tham số (Parameters)`, `Giá trị trả về (Returns)`, `Ngoại lệ (Raises)`).
  - Path resolution strictly via `pathlib.Path` relative to repository root (`REPO_ROOT = Path(__file__).resolve().parents[2]`).
- **Regression Invariant**:
  All 49 existing unit tests under `tests/unit/` must continue to pass 100% green.

---

## 4. Technical Architecture & Integration Pipeline Flow

```text
[Input: VideoReader (T0301)]
           | (frame_id, timestamp, raw_frame BGR)
           v
[LandmarkDetector (T0402)]
           |
   +-------+-----------------------------+
   | (landmark_valid == False)           | (landmark_valid == True)
   v                                     v
[Emit UNKNOWN States]                 [Extract 6 Eye & 8 Mouth Landmarks]
   - ear: 0.0, eye_state: UNKNOWN        |
   - mar: 0.0, mouth_state: UNKNOWN      +---------------------+
   - tracker: reset closure duration     |                     |
                                         v                     v
                             [calculate_eye_state]   [calculate_mouth_state]
                                         | (T0501)             | (T0601)
                                         v                     v
                              [EyeClosureTracker]     {mar, mouth_state}
                                         | (T0701)
                                         v
                         {prolonged_closure, blink, duration}
```

### Data Contract Verification Matrix:
| Component | Function / Class | Expected Output Schema |
|---|---|---|
| Landmark Detector | `LandmarkDetector.process(frame)` | `{"landmark_valid": bool, "left_eye_points": list[tuple] \| None, "right_eye_points": list[tuple] \| None, "mouth_points": list[tuple] \| None}` |
| Eye Feature | `calculate_eye_state(left_pts, right_pts)` | `{"ear": float, "eye_state": "OPEN" \| "CLOSED" \| "UNKNOWN"}` |
| Mouth Feature | `calculate_mouth_state(mouth_pts, landmark_valid)` | `{"mar": float, "mouth_state": "MOUTH_OPEN" \| "MOUTH_CLOSED" \| "UNKNOWN"}` |
| Temporal Tracker | `EyeClosureTracker.update(eye_state, current_time)` | `{"prolonged_eye_closure": bool, "blink_detected": bool, "closed_start_time": float \| None, "consecutive_closed_frames": int, "closure_duration": float}` |

---

## 5. Complete Implementation Blueprint

Create the integration file at:
`tests/integration/T1001_261010_1030_test_frame_pipeline.py`

```python
"""Script kiểm thử tích hợp Pipeline phân tích khuôn mặt (Task T1001 - Milestone M03).

Quy trình tích hợp:
    VideoReader (T0301) -> LandmarkDetector (T0402) -> Eye & Mouth Feature Extractor (T0501, T0601)
    -> Temporal Eye Closure Tracker (T0701).

Hỗ trợ 2 chế độ:
    1. Headless test (pytest): Kiểm thử tính toàn vẹn hợp đồng dữ liệu, ngưỡng và xử lý lỗi FReq030/032.
    2. Interactive visual test (CLI): Chạy trực tiếp có giao diện HUD thời gian thực, FPS và phím thoát.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytest

# Thiết lập đường dẫn gốc dự án để đảm bảo import nhất quán
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.video_reader import VideoReader
from src.detection.landmark_detector import LandmarkDetector
from src.features.eye import calculate_eye_state
from src.features.mouth import calculate_mouth_state
from src.temporal.eye_closure import EyeClosureTracker
from src.utils.config import load_config


class IntegratedFacialPipeline:
    """Lớp điều phối tích hợp luồng xử lý khuôn mặt PoC (Task T1001)."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Khởi tạo toàn bộ các thành phần trong pipeline.

        Parameters
        ----------
        config_path : Optional[str]
            Đường dẫn tới file cấu hình YAML (mặc định nạp configs/default.yaml).
        """
        self.config = load_config(config_path) if config_path else load_config()
        model_path = REPO_ROOT / "models" / "pretrained" / "face_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy mô hình MediaPipe: {model_path}")

        min_conf = float(self.config.get("face_detection", {}).get("min_detection_confidence", 0.5))
        self.detector = LandmarkDetector(
            model_path=str(model_path),
            min_detection_confidence=min_conf,
        )
        self.eye_tracker = EyeClosureTracker()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Xử lý đơn frame qua toàn bộ các tầng trích xuất và trạng thái thời gian.

        Parameters
        ----------
        frame : np.ndarray
            Khung hình BGR từ nguồn video.
        timestamp : float
            Thời điểm tương ứng của khung hình tính theo giây.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa đầy đủ thông tin mốc, EAR, MAR, trạng thái mắt/miệng và temporal tracker.
        """
        start_time = time.perf_counter()

        # 1. Trích xuất Landmark từ MediaPipe Tasks API
        detection_result = self.detector.process(frame)
        landmark_valid = bool(detection_result.get("landmark_valid", False))
        left_eye_points = detection_result.get("left_eye_points")
        right_eye_points = detection_result.get("right_eye_points")
        mouth_points = detection_result.get("mouth_points")

        # 2. Trích xuất đặc trưng Mắt (T0501) với chốt an toàn FReq004/FReq030
        if landmark_valid and left_eye_points and right_eye_points:
            eye_result = calculate_eye_state(left_eye_points, right_eye_points)
        else:
            eye_result = {"ear": 0.0, "eye_state": "UNKNOWN"}

        # 3. Trích xuất đặc trưng Miệng (T0601)
        mouth_result = calculate_mouth_state(
            mouth_points=mouth_points,
            landmark_valid=landmark_valid,
        )

        # 4. Cập nhật trạng thái thời gian nhắm mắt (T0701)
        temporal_result = self.eye_tracker.update(
            eye_state=eye_result["eye_state"],
            current_time=timestamp,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "timestamp": timestamp,
            "latency_ms": latency_ms,
            "landmark_valid": landmark_valid,
            "landmarks": {
                "left_eye": left_eye_points,
                "right_eye": right_eye_points,
                "mouth": mouth_points,
            },
            "eye": eye_result,
            "mouth": mouth_result,
            "temporal": temporal_result,
        }

    def release(self) -> None:
        """Giải phóng tài nguyên detector."""
        self.detector.release()


def render_hud(
    frame: np.ndarray,
    frame_id: int,
    result: Dict[str, Any],
    fps: float,
) -> np.ndarray:
    """Vẽ giao diện hiển thị thông tin đo đạc (HUD) trực quan lên khung hình.

    Parameters
    ----------
    frame : np.ndarray
        Khung hình BGR gốc.
    frame_id : int
        Chỉ số khung hình hiện tại.
    result : Dict[str, Any]
        Kết quả đầu ra từ IntegratedFacialPipeline.process_frame.
    fps : float
        Tốc độ khung hình tức thời (FPS).

    Returns
    -------
    np.ndarray
        Khung hình đã được vẽ HUD và các điểm mốc.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Vẽ các điểm mốc giải phẫu nếu hợp lệ
    if result["landmark_valid"]:
        landmarks = result["landmarks"]
        if landmarks["left_eye"]:
            for pt in landmarks["left_eye"]:
                cv2.circle(canvas, pt, 2, (0, 255, 255), -1)  # Vàng
        if landmarks["right_eye"]:
            for pt in landmarks["right_eye"]:
                cv2.circle(canvas, pt, 2, (255, 200, 0), -1)  # Cam nhạt
        if landmarks["mouth"]:
            for pt in landmarks["mouth"]:
                cv2.circle(canvas, pt, 2, (0, 165, 255), -1)  # Cam đậm

    # Tạo bảng thông số mờ phía trên (Overlay Banner)
    banner_height = 95
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # 1. Hàng thông số hệ thống
    status_text = (
        f"Frame: {frame_id:04d} | Time: {result['timestamp']:.2f}s | "
        f"FPS: {fps:.1f} | Latency: {result['latency_ms']:.1f}ms"
    )
    cv2.putText(canvas, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. Hàng chỉ số Mắt & Miệng
    ear_val = result["eye"]["ear"]
    eye_state = result["eye"]["eye_state"]
    mar_val = result["mouth"]["mar"]
    mouth_state = result["mouth"]["mouth_state"]

    eye_color = (0, 255, 0) if eye_state == "OPEN" else ((0, 140, 255) if eye_state == "CLOSED" else (180, 180, 180))
    mouth_color = (0, 255, 255) if mouth_state == "MOUTH_OPEN" else ((0, 255, 0) if mouth_state == "MOUTH_CLOSED" else (180, 180, 180))

    feat_text = f"EAR: {ear_val:.3f} [{eye_state}]  |  MAR: {mar_val:.3f} [{mouth_state}]"
    cv2.putText(canvas, feat_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)

    # 3. Hàng trạng thái thời gian (Temporal Alert)
    temporal = result["temporal"]
    prolonged = temporal["prolonged_eye_closure"]
    closure_dur = temporal["closure_duration"]
    consec_frames = temporal["consecutive_closed_frames"]

    temporal_color = (0, 0, 255) if prolonged else (0, 255, 0)
    temporal_text = (
        f"Closure: {closure_dur:.2f}s (Frames: {consec_frames}) | "
        f"PROLONGED: {'YES (WARNING)' if prolonged else 'NO'}"
    )
    cv2.putText(canvas, temporal_text, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, temporal_color, 2)

    # Báo động đỏ lớn giữa màn hình nếu nhắm mắt kéo dài
    if prolonged:
        alert_box_y = int(h * 0.45)
        cv2.putText(
            canvas,
            "WARNING: PROLONGED EYE CLOSURE!",
            (w // 2 - 280, alert_box_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
        )

    return canvas


def run_interactive_pipeline(
    video_source: str,
    max_frames: Optional[int] = None,
    visual: bool = True,
) -> Dict[str, float]:
    """Khởi chạy pipeline ở chế độ trực quan hoặc benchmarking có báo cáo hiệu năng.

    Parameters
    ----------
    video_source : str
        Đường dẫn file video hoặc chỉ số camera ('0').
    max_frames : Optional[int]
        Giới hạn số khung hình xử lý (None = chạy hết).
    visual : bool
        Có hiển thị cửa sổ OpenCV GUI hay không.

    Returns
    -------
    Dict[str, float]
        Báo cáo hiệu năng gồm average_latency_ms, effective_fps, total_frames.
    """
    pipeline = IntegratedFacialPipeline()
    latencies = []
    processed_count = 0

    video_path = REPO_ROOT / video_source if not Path(video_source).is_absolute() else Path(video_source)
    if not video_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video nguồn: {video_path}")

    reader = VideoReader(video_path)
    print(f"\n[INFO] Đang chạy Pipeline trên nguồn: {video_path}")
    print(f"[INFO] Độ phân giải: {reader.width}x{reader.height} | FPS gốc: {reader.fps:.1f}")
    if visual:
        print("[INFO] Bấm 'q', 'Q', hoặc 'ESC' để dừng chương trình.\n")

    start_total_time = time.perf_counter()
    window_name = "Driver Drowsiness - Facial Pipeline Integration (Task T1001)"

    try:
        with reader as stream:
            for frame_id, timestamp, frame in stream:
                if max_frames and processed_count >= max_frames:
                    break

                res = pipeline.process_frame(frame, timestamp)
                latencies.append(res["latency_ms"])
                processed_count += 1

                current_fps = 1000.0 / res["latency_ms"] if res["latency_ms"] > 0 else 0.0

                if visual:
                    hud_frame = render_hud(frame, frame_id, res, current_fps)
                    cv2.imshow(window_name, hud_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), ord("Q"), 27]:
                        print("\n[INFO] Nhận tín hiệu dừng từ người dùng.")
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Nhận ngắt tín hiệu Ctrl+C.")
    finally:
        total_elapsed = time.perf_counter() - start_total_time
        pipeline.release()
        if visual:
            cv2.destroyAllWindows()

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    effective_fps = processed_count / total_elapsed if total_elapsed > 0 else 0.0

    print("\n" + "=" * 50)
    print(" BÁO CÁO HIỆU NĂNG TÍCH HỢP (SCri014 PROFILING)")
    print("=" * 50)
    print(f" Tổng số frame xử lý       : {processed_count}")
    print(f" Thời gian thực thi        : {total_elapsed:.2f}s")
    print(f" Độ trễ trung bình (Mean)  : {avg_latency:.2f} ms")
    print(f" Độ trễ phân vị 95 (P95)   : {p95_latency:.2f} ms")
    print(f" Tốc độ xử lý (FPS)        : {effective_fps:.1f} FPS")
    print("=" * 50 + "\n")

    return {
        "total_frames": processed_count,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "effective_fps": effective_fps,
    }


# ==============================================================================
# PYTEST AUTOMATED INTEGRATION TESTS (HEADLESS)
# ==============================================================================

@pytest.fixture(scope="module")
def facial_pipeline():
    """Khởi tạo pipeline chung cho các bài kiểm thử headless."""
    pipeline = IntegratedFacialPipeline()
    yield pipeline
    pipeline.release()


def test_pipeline_on_synthetic_blank_frame_safety(facial_pipeline):
    """Kiểm thử chốt an toàn FReq004 & FReq030 khi không có khuôn mặt (ảnh đen)."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = facial_pipeline.process_frame(blank_frame, timestamp=0.0)

    # Khẳng định mốc không hợp lệ và trạng thái UNKNOWN
    assert result["landmark_valid"] is False
    assert result["landmarks"]["left_eye"] is None
    assert result["landmarks"]["right_eye"] is None
    assert result["landmarks"]["mouth"] is None

    assert result["eye"]["ear"] == 0.0
    assert result["eye"]["eye_state"] == "UNKNOWN"

    assert result["mouth"]["mar"] == 0.0
    assert result["mouth"]["mouth_state"] == "UNKNOWN"

    # Red Line: Không được báo DROWSY khi mất dấu khuôn mặt
    assert result["temporal"]["prolonged_eye_closure"] is False
    assert result["temporal"]["consecutive_closed_frames"] == 0


def test_pipeline_data_contracts_on_d3s_video(facial_pipeline):
    """Kiểm thử tính toàn vẹn của hợp đồng dữ liệu trên video D3S thực tế."""
    sample_video = REPO_ROOT / "data" / "raw" / "d3s" / "d3s_video_dataset.mp4"
    if not sample_video.exists():
        pytest.skip("Bỏ qua kiểm thử do không có video mẫu D3S tại data/raw/d3s/")

    reader = VideoReader(sample_video)
    tested_frames = 0

    with reader as stream:
        for frame_id, timestamp, frame in stream:
            result = facial_pipeline.process_frame(frame, timestamp)

            # Kiểm tra kiểu dữ liệu các trường chính
            assert isinstance(result["landmark_valid"], bool)
            assert isinstance(result["latency_ms"], float)
            assert result["latency_ms"] >= 0.0

            assert "ear" in result["eye"]
            assert "eye_state" in result["eye"]
            assert result["eye"]["eye_state"] in ["OPEN", "CLOSED", "UNKNOWN"]

            assert "mar" in result["mouth"]
            assert "mouth_state" in result["mouth"]
            assert result["mouth"]["mouth_state"] in ["MOUTH_OPEN", "MOUTH_CLOSED", "UNKNOWN"]

            assert "prolonged_eye_closure" in result["temporal"]
            assert "blink_detected" in result["temporal"]
            assert "closure_duration" in result["temporal"]

            tested_frames += 1
            if tested_frames >= 20:
                break

    assert tested_frames == 20


def test_pipeline_resource_release_and_context_teardown():
    """Kiểm định giải phóng an toàn tài nguyên (FReq032)."""
    pipe = IntegratedFacialPipeline()
    sample_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _ = pipe.process_frame(sample_frame, timestamp=0.0)
    # Gọi giải phóng nhiều lần không được gây lỗi exception
    pipe.release()
    pipe.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy kiểm thử tích hợp Pipeline phân tích khuôn mặt (Task T1001).")
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw/d3s/d3s_video_dataset.mp4",
        help="Đường dẫn file video hoặc webcam ID (mặc định: data/raw/d3s/d3s_video_dataset.mp4)",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        default=True,
        help="Bật giao diện hiển thị đồ họa OpenCV HUD",
    )
    parser.add_argument(
        "--no-visual",
        dest="visual",
        action="store_false",
        help="Chạy chế độ ngầm (headless benchmark)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Số lượng khung hình tối đa cần xử lý (None = hết video)",
    )

    args = parser.parse_args()
    run_interactive_pipeline(
        video_source=args.source,
        max_frames=args.max_frames,
        visual=args.visual,
    )
```

---

## 6. Verification & Definition of Done (DoD) Checklist

Sau khi viết mã, Codex Agent bắt buộc phải thực hiện tuần tự các bước kiểm tra sau trong PowerShell:

1. **Dọn dẹp tệp tin rác / sai quy cách**:
   ```powershell
   if (Test-Path "tests/unit/integration") { Remove-Item -Recurse -Force "tests/unit/integration" }
   ```
2. **Kiểm tra hồi quy toàn diện Unit Tests (Bắt buộc 49/49 Pass 100%)**:
   ```powershell
   .venv\Scripts\pytest.exe tests/unit/ -v
   ```
3. **Chạy kiểm thử tích hợp mới (Integration Test - Headless Mode)**:
   ```powershell
   .venv\Scripts\pytest.exe tests/integration/T1001_261010_1030_test_frame_pipeline.py -v
   ```
4. **Kiểm tra hiệu năng và đo lường Profiling (`SCri014`) trên 100 frames**:
   ```powershell
   .venv\Scripts\python.exe tests/integration/T1001_261010_1030_test_frame_pipeline.py --source data/raw/d3s/d3s_video_dataset.mp4 --no-visual --max-frames 100
   ```
   *Yêu cầu*: Độ trễ trung bình $\le 50$ ms, FPS $\ge 20$.
5. **Kiểm tra chuẩn cú pháp và linting**:
   ```powershell
   .venv\Scripts\flake8.exe tests/integration/T1001_261010_1030_test_frame_pipeline.py
   ```
6. **Kiểm tra trạng thái Git**:
   ```powershell
   git status
   ```
