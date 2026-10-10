# BÁO CÁO TÍCH HỢP NHÁNH & KIỂM ĐỊNH TÍNH TOÀN VẸN (TASK T1001)

- **Mã định danh nhiệm vụ (Task ID)**: `T1001`
- **Gói công việc (Work Package)**: `WP10` (Core Facial Analysis Prototype Integration - PoC)
- **Cột mốc (Milestone)**: `M03` (Facial Analysis Prototype)
- **Nhánh tích hợp mục tiêu**: `feature/T1001-frame-pipeline-poc`
- **Thời gian khởi tạo**: `2026-10-10 10:30:00+07:00`
- **Vai trò thực hiện**: Lead AI DevOps & Systems Architect (Antigravity Agent)

---

## 1. TỔNG QUAN VÀ MỤC TIÊU TÍCH HỢP

Nhằm chuẩn bị môi trường chuẩn xác, sạch và đầy đủ cho việc phát triển và kiểm thử tích hợp toàn bộ pipeline phân tích khuôn mặt (Task `T1001`), toàn bộ các nhánh tính năng upstream đã hoàn thành từ các thành viên trong nhóm đã được rà soát, kiểm toán và hợp nhất tuần tự vào nhánh làm việc chuyên dụng `feature/T1001-frame-pipeline-poc`.

### Nguyên tắc bảo đảm (Guardrails Enforcement):
1. **Tuân thủ ranh giới Đỏ (Red Line)**: Tuyệt đối không can thiệp, commit hay push trực tiếp vào nhánh `main`.
2. **Không phân tán dữ liệu / file rác**: Các dữ liệu video dung lượng lớn (`data/raw/d3s/*.mp4`) và các thư mục cache (`__pycache__`, `.pytest_cache`) được kiểm soát chặt qua `.gitignore`.
3. **Bảo toàn hợp đồng giao tiếp (Interface Contracts)**: Mọi cấu trúc dữ liệu Pydantic / Dictionary (`frame_id, timestamp, frame`, `{"ear", "eye_state"}`, `{"mar", "mouth_state"}`) và xử lý ngoại lệ mất mặt (`UNKNOWN`) được giữ nguyên vẹn.

---

## 2. DANH SÁCH CÁC NHÁNH TÍNH NĂNG ĐƯỢC HỢP NHẤT

Nhánh nền tảng xuất phát (Base commit): `feat/T0301-Input-Acquisition-Modules` (Commit `dd5249a`), đã kế thừa đầy đủ `T0201` (Tải dataset D3S) và `T0202` (Kiểm kê metadata D3S).

Thứ tự hợp nhất kiến trúc tuần tự từ Upstream -> Downstream:

| STT | Tên nhánh (Branch Name) | Commit Hash | Module chính phụ trách | Tác vụ (Task ID) |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `origin/feat/T1201-Euclidean2D` | `ca6affe` | `src/utils/geometry.py` | `T1201` |
| 2 | `origin/feature/T0401-Models-Detection` | `9eba461` | `src/detection/face_detector.py`<br>`src/detection/landmark_detector.py`<br>`models/pretrained/*.task` | `T0401`, `T0402` |
| 3 | `origin/feat/T0501-Eye-Feature` | `acab2b7` | `src/features/eye.py`<br>`src/utils/config.py` | `T0501` |
| 4 | `origin/feat/T0601-Mouth-Feature` | `33f3a2c` | `src/features/mouth.py` | `T0601` |
| 5 | `origin/feat/T0701-Eye-Closure` | `1390dcf` | `src/temporal/eye_closure.py` | `T0701` |

---

## 3. CHI TIẾT TẬP TIN HỢP NHẤT & XỬ LÝ XUNG ĐỘT (CONFLICT RESOLUTION)

### 3.1. Tập tin hợp nhất
- **Tiện ích hình học & cấu hình (`WP12`)**:
  - `src/utils/geometry.py`: Hàm `euclidean_distance_2d(p1, p2)` phục vụ tính toán khoảng cách pixel cho EAR và MAR.
  - `src/utils/config.py`: Hàm `load_config()` đọc cấu hình ngưỡng từ `configs/default.yaml`.
  - `tests/unit/T1201_041026_test_geometry.py`: Bộ kiểm thử khoảng cách Euclidean.
  - `tests/unit/T0501_061026_test_config.py`: Kiểm thử nạp YAML config.
- **Phát hiện khuôn mặt & Mốc giải phẫu (`WP04`)**:
  - `src/detection/face_detector.py`: Lớp `FaceDetector` (MediaPipe Tasks API BlazeFace).
  - `src/detection/landmark_detector.py`: Lớp `LandmarkDetector` (MediaPipe FaceLandmarker, xuất 6 mốc mắt trái, 6 mốc mắt phải, 8 mốc môi trong/ngoài).
  - `models/pretrained/blaze_face_short_range.tflite` & `models/pretrained/face_landmarker.task`: Mô hình trọng số pretrained chính thức.
  - `scripts/test_detection_*.py`: Các script tiền trạm và UI demo.
- **Đặc trưng Mắt (`WP05`)**:
  - `src/features/eye.py`: Tính `calculate_ear()` và `calculate_eye_state()` (OPEN / CLOSED / UNKNOWN).
  - `tests/unit/T0501_261006_test_eye.py`: Bộ kiểm thử đơn vị bao phủ toàn diện công thức Soukupová-Čech.
- **Đặc trưng Miệng (`WP06`)**:
  - `src/features/mouth.py`: Tính `calculate_mar()` và `calculate_mouth_state()` (MOUTH_OPEN / MOUTH_CLOSED / UNKNOWN).
- **Trạng thái Thời gian (`WP08`)**:
  - `src/temporal/eye_closure.py`: Lớp `EyeClosureTracker` theo dõi thời gian nhắm mắt liên tục, phân biệt chớp mắt ngắn (blink) với nhắm mắt kéo dài (prolonged eye closure).
  - `tests/unit/T0701_261008_test_eye_closure.py`: Bộ kiểm thử chuyển trạng thái thời gian và reset khi mở mắt.

### 3.2. Tình trạng xung đột
- Toàn bộ 5 lượt merge đều thực hiện thành công thông qua chiến lược `ort` tự động của Git.
- **Số lượng xung đột**: **0 xung đột (Zero conflicts)**.
- **Bảo toàn giao tiếp**: Không có bất kỳ logic kiểm thử hoặc đoạn mã nào của các thành viên bị ghi đè hay thất thoát.

---

## 4. KẾT QUẢ KIỂM THỬ HỒI QUY (QUALITY GATE VERIFICATION)

Đã cấu hình tương thích môi trường Windows cho `pytest.ini` (`addopts = --basetemp=.pytest_cache/tmp`) nhằm khắc phục triệt để lỗi phân quyền `PermissionError` trên thư mục tạm của hệ điều hành.

Lệnh thực thi:
```powershell
.venv\Scripts\pytest.exe tests/unit/ -v
```

Kết quả thực tế (100% GREEN PASS):
```text
============================= test session starts =============================
platform win32 -- Python 3.13.2, pytest-9.1.1, pluggy-1.6.0 -- F:\Study Materials\Môn Ngành\PY NC\PY_ADV_PJ_GR\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: F:\Study Materials\Môn Ngành\PY NC\PY_ADV_PJ_GR
configfile: pytest.ini
plugins: anyio-4.15.1, platformdirs-4.12.1
collecting ... collected 49 items

tests/unit/T0201_260928_1830_test_download_dataset.py::test_default_download_name_matches_published_video PASSED [  2%]
tests/unit/T0201_260928_1830_test_download_dataset.py::test_prepare_existing_zip_preserves_layout_and_verifies_video PASSED [  4%]
tests/unit/T0201_260928_1830_test_download_dataset.py::test_extract_archive_rejects_path_traversal PASSED [  6%]
tests/unit/T0201_260928_1830_test_download_dataset.py::test_verify_dataset_videos_fails_for_unreadable_video PASSED [  8%]
tests/unit/T0201_260928_1830_test_download_dataset.py::test_skip_download_verifies_existing_dataset PASSED [ 10%]
tests/unit/T0201_260928_1830_test_download_dataset.py::test_google_drive_confirmation_form_uses_modern_download_url PASSED [ 12%]
tests/unit/T0201_260928_1830_test_download_dataset.py::test_download_retries_and_replaces_invalid_archive_atomically PASSED [ 14%]
tests/unit/T0202_260929_2131_test_d3s_inventory.py::test_inventory_csv_exists_with_expected_schema PASSED [ 16%]
tests/unit/T0202_260929_2131_test_d3s_inventory.py::test_inventory_csv_contains_non_empty_records PASSED [ 18%]
tests/unit/T0202_260929_2131_test_d3s_inventory.py::test_inventory_csv_has_valid_video_measurements PASSED [ 20%]
tests/unit/T0202_260929_2131_test_d3s_inventory.py::test_subject_and_label_parser_supports_structured_d3s_paths PASSED [ 22%]
tests/unit/T0301_260930_0830_test_acquisition.py::test_video_reader_reads_synthetic_video_with_metadata PASSED [ 24%]
tests/unit/T0301_260930_0830_test_acquisition.py::test_video_reader_rejects_missing_video PASSED [ 26%]
tests/unit/T0301_260930_0830_test_acquisition.py::test_video_reader_stops_at_eof_and_releases_capture PASSED [ 28%]
tests/unit/T0301_260930_0830_test_acquisition.py::test_video_reader_treats_empty_frame_as_eof PASSED [ 30%]
tests/unit/T0301_260930_0830_test_acquisition.py::test_video_reader_context_manager_releases_without_iteration PASSED [ 32%]
tests/unit/T0301_260930_0830_test_acquisition.py::test_camera_stream_yields_monotonic_elapsed_timestamps PASSED [ 34%]
tests/unit/T0301_260930_0830_test_acquisition.py::test_camera_stream_rejects_unavailable_device PASSED [ 36%]
tests/unit/T0301_260930_0830_test_acquisition.py::test_camera_stream_context_manager_releases_on_exit PASSED [ 38%]
tests/unit/T0301_260930_0903_test_video.py::test_video_reader_emits_metadata_timestamps_and_frames PASSED [ 40%]
tests/unit/T0301_260930_0903_test_video.py::test_video_reader_rejects_a_missing_file PASSED [ 42%]
tests/unit/T0301_260930_0904_test_camera.py::test_camera_stream_emits_monotonic_timestamps_and_releases PASSED [ 44%]
tests/unit/T0301_260930_0904_test_camera.py::test_camera_stream_rejects_an_unavailable_device PASSED [ 46%]
tests/unit/T0501_061026_test_config.py::test_load_config PASSED          [ 48%]
tests/unit/T0501_261006_test_eye.py::test_calculate_ear_uses_soukupova_cech_formula PASSED [ 51%]
tests/unit/T0501_261006_test_eye.py::test_calculate_eye_state_averages_both_eyes_and_uses_config PASSED [ 53%]
tests/unit/T0501_261006_test_eye.py::test_calculate_eye_state_classifies_threshold_as_open PASSED [ 55%]
tests/unit/T0501_261006_test_eye.py::test_calculate_eye_state_returns_unknown_for_invalid_landmarks[None-right_eye0] PASSED [ 57%]
tests/unit/T0501_261006_test_eye.py::test_calculate_eye_state_returns_unknown_for_invalid_landmarks[left_eye1-right_eye1] PASSED [ 59%]
tests/unit/T0501_261006_test_eye.py::test_calculate_eye_state_returns_unknown_for_invalid_landmarks[left_eye2-right_eye2] PASSED [ 61%]
tests/unit/T0501_261006_test_eye.py::test_calculate_eye_state_returns_unknown_for_invalid_landmarks[left_eye3-right_eye3] PASSED [ 63%]
tests/unit/T0501_261006_test_eye.py::test_calculate_eye_state_returns_unknown_for_zero_horizontal_distance PASSED [ 65%]
tests/unit/T0701_261008_test_eye_closure.py::test_open_eye PASSED        [ 67%]
tests/unit/T0701_261008_test_eye_closure.py::test_short_blink PASSED     [ 69%]
tests/unit/T0701_261008_test_eye_closure.py::test_prolonged_eye_closure PASSED [ 71%]
tests/unit/T0701_261008_test_eye_closure.py::test_consecutive_closed_frames PASSED [ 73%]
tests/unit/T0701_261008_test_eye_closure.py::test_reset_after_open PASSED [ 75%]
tests/unit/T0701_261008_test_eye_closure.py::test_new_closure_after_reset PASSED [ 77%]
tests/unit/T0701_261008_test_eye_closure.py::test_invalid_eye_state PASSED [ 79%]
tests/unit/T0701_261008_test_eye_closure.py::test_threshold_is_loaded_from_yaml PASSED [ 81%]
tests/unit/T0701_261008_test_eye_closure.py::test_closure_at_threshold_is_prolonged PASSED [ 83%]
tests/unit/T0701_261008_test_eye_closure.py::test_reopening_after_prolonged_closure_keeps_event_visible PASSED [ 85%]
tests/unit/T0701_261008_test_eye_closure.py::test_unknown_breaks_continuous_closure_tracking PASSED [ 87%]
tests/unit/T1201_041026_test_geometry.py::test_euclidean_distance_2d_with_numeric_points[point10-point20-0] PASSED [ 89%]
tests/unit/T1201_041026_test_geometry.py::test_euclidean_distance_2d_with_numeric_points[point11-point21-3] PASSED [ 91%]
tests/unit/T1201_041026_test_geometry.py::test_euclidean_distance_2d_with_numeric_points[point12-point22-4] PASSED [ 93%]
tests/unit/T1201_041026_test_geometry.py::test_euclidean_distance_2d_with_numeric_points[point13-point23-5] PASSED [ 95%]
tests/unit/T1201_041026_test_geometry.py::test_euclidean_distance_2d_with_numeric_points[point14-point24-5] PASSED [ 97%]
tests/unit/T1201_041026_test_geometry.py::test_euclidean_distance_2d_with_numeric_points[point15-point25-5] PASSED [100%]

============================= 49 passed in 1.81s ==============================
```

---

## 5. KẾT LUẬN & XÁC NHẬN SẴN SÀNG TRIỂN KHAI T1001

1. **Trạng thái kho mã**: Hoàn toàn sạch sẽ, không có file nhị phân lớn hoặc file untracked vi phạm quy ước.
2. **Trạng thái hồi quy**: Đạt chuẩn chất lượng 100% Pass (49/49 tests passed).
3. **Mức độ sẵn sàng**: Nhánh `feature/T1001-frame-pipeline-poc` đã sẵn sàng 100% cho việc triển khai kịch bản kiểm thử tích hợp pipeline PoC (Task `T1001` - Milestone `M03`).
