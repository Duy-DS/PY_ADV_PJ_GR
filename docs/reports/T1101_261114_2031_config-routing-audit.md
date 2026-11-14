# Báo cáo kiểm toán định tuyến cấu hình và xác minh runtime

- **Task:** T1101 — audit chỉ-đọc `default.yaml` và `experiment.yaml`
- **Thời điểm:** 2026-11-14 20:31 (giờ local của môi trường)
- **Phạm vi:** Truy vết loader/entrypoint, xác minh parser và giá trị cấu hình runtime.
- **Kết quả:** Đã xác minh; không sửa production code, scripts hoặc configs.

## 1. Tóm tắt kết luận

1. `src/utils/config.py::load_config()` không có đối số sẽ đọc `configs/default.yaml`; nếu truyền đường dẫn thì đọc trực tiếp tệp đó. Hàm không deep-merge hay kế thừa cấu hình.
2. CLI `src/main.py` đặt `--config` mặc định là `None`; `RealtimePipeline` chuyển `None` thành `configs/default.yaml`. Vì vậy đây là mặc định của luồng realtime.
3. `scripts/evaluate.py` có loader YAML riêng, đặt `--config` mặc định là `configs/experiment.yaml`, và đọc trực tiếp tệp được chọn. Chạy `evaluate.py` không kèm `--config` vẫn chọn experiment config.
4. `experiment.yaml` đang chứa trực tiếp cả `alert.cooldown` và hai giá trị `temporal.*`; giá trị trùng với default nhưng đây là khai báo lặp, không phải kế thừa. `drowsiness.mode` không có trong experiment config.
5. `scripts/run_demo.py` và thư mục `experiments/` không tồn tại trong checkout được kiểm tra. Không thể xác nhận hành vi của một script không có mặt.

## 2. Bảng định tuyến entrypoint

| Entry point / API | Mặc định hoặc đường đi | Bằng chứng mã nguồn |
|---|---|---|
| `load_config()` | `configs/default.yaml`, tính từ vị trí module; đối số tường minh được dùng trực tiếp, không merge | `src/utils/config.py:11-26` |
| `python src/main.py --source ... --mode ...` | Parser đặt `args.config=None`; truyền vào `RealtimePipeline`; pipeline chọn `configs/default.yaml` | `src/main.py:124-127, 137`; `src/pipeline/realtime_pipeline.py:172-180` |
| `python src/main.py --source ... --mode ... --config configs/experiment.yaml` | Đọc trực tiếp `experiment.yaml`; CLI `--mode` được gán vào `drowsiness.mode` trong dict sau khi đọc | `src/main.py:124-137`; `src/pipeline/realtime_pipeline.py:172-180` |
| `python scripts/evaluate.py` | `configs/experiment.yaml` | `scripts/evaluate.py:22-30`; parser runtime xác nhận bên dưới |
| `python scripts/evaluate.py --config PATH` | Loader riêng ghép `PATH` vào `REPO_ROOT` rồi gọi `yaml.safe_load`; không gọi `src.utils.config.load_config` | `scripts/evaluate.py:33-40` |
| `scripts/run_demo.py` | Không có trong checkout; không có entrypoint để kiểm tra | `Test-Path scripts/run_demo.py` trả `False` |
| `src/pipeline/offline_pipeline.py` | Constructor mặc định `config_path=None`, chuyển nguyên giá trị tới `TemporalAnalyzer`; bản thân module không đặt một đường dẫn experiment | `src/pipeline/offline_pipeline.py:25-32` |

**Lưu ý về tham số CLI:** `src/main.py` yêu cầu cả `--source` và `--mode` khi chạy CLI (`src/main.py:130-137`). Do đó ví dụ `python src/main.py --config configs/experiment.yaml` đứng riêng sẽ thoát với lỗi thiếu tham số, không khởi chạy pipeline. Với `--source ... --mode eye_yawn --config configs/experiment.yaml`, config experiment được đọc, sau đó mode CLI ghi đè `drowsiness.mode` thành `eye_yawn`.

## 3. Xác minh runtime

### 3.1. Main/realtime với cấu hình mặc định

Snippet chỉ gọi loader, không khởi động camera, video, cửa sổ GUI hoặc pipeline:

```text
ENTRYPOINT=src/main.py default config
resolved_path=F:\Study Materials\specialized_subject\PY NC\PY_ADV_PJ_GR\configs\default.yaml
eye.ear_threshold=0.25
eye.closure_time_threshold=1.5
mouth.mar_threshold=0.6
mouth.yawn_duration_threshold=2.0
temporal.missing_face_tolerance_frames=5
temporal.missing_face_timeout_sec=1.5
alert.cooldown=3.0
drowsiness.mode=eye_yawn
```

### 3.2. Evaluation parser và loader

Lệnh `python scripts/evaluate.py --help` chạy thành công; phần trợ giúp cho thấy chỉ có `-h/--help` và `--config CONFIG` (không có tùy chọn `--output`). Gọi `parse_args()` với `sys.argv=['evaluate.py']` cho kết quả:

```text
evaluate.parse_args() with no --config => 'configs/experiment.yaml'
resolved=F:\Study Materials\specialized_subject\PY NC\PY_ADV_PJ_GR\configs\experiment.yaml
```

Đọc bằng đúng `scripts.evaluate.load_config('configs/experiment.yaml')`:

```text
ENTRYPOINT=scripts/evaluate.py config default/explicit experiment
resolved_path=F:\Study Materials\specialized_subject\PY NC\PY_ADV_PJ_GR\configs\experiment.yaml
modes=['eye_only', 'eye_yawn']
eye.ear_threshold=0.22
eye.closure_time_threshold=1.2
mouth.mar_threshold=0.55
mouth.yawn_duration_threshold=1.5
temporal.missing_face_tolerance_frames=5
temporal.missing_face_timeout_sec=1.5
alert.cooldown=3.0
drowsiness.mode=<MISSING>
loader_merge=none
```

Luồng lặp mode được xác nhận trong `scripts/evaluate.py:180-196`: lấy `modes = config.get("modes", ["eye_only", "eye_yawn"])`, sau đó `for mode in modes`, gọi `run_experiment(..., config_path=args.config)`. `run_experiment` tạo `OfflinePipeline(config_path=config_path, mode=mode)` (`scripts/evaluate.py:51-53`).

### 3.3. Main khi truyền experiment config

Đọc trực tiếp cùng tệp bằng loader của `src/utils/config.py` cho kết quả:

```text
ENTRYPOINT=src/main.py --config configs/experiment.yaml loader verification
resolved_path=F:\Study Materials\specialized_subject\PY NC\PY_ADV_PJ_GR\configs\experiment.yaml
eye.ear_threshold=0.22
eye.closure_time_threshold=1.2
mouth.mar_threshold=0.55
mouth.yawn_duration_threshold=1.5
temporal.missing_face_tolerance_frames=5
temporal.missing_face_timeout_sec=1.5
alert.cooldown=3.0
drowsiness.mode=<MISSING>
```

Đây là kiểm chứng loader-level, không phải chạy CLI xử lý video. `RealtimePipeline` dùng `load_config(self.config_path)`, gửi cùng đường dẫn cho `TemporalAnalyzer`, và truyền config đã nạp cho `DecisionEngine`/`AlertManager` (`src/pipeline/realtime_pipeline.py:172-188`). Nếu chạy đủ CLI, `--mode` là bắt buộc và mode đó sẽ được đặt vào config ở dòng 179.

## 4. Inheritance, override và xử lý key thiếu

- **Không có deep-merge:** `src/utils/config.py:18-26` chỉ chọn đường dẫn (nếu `None`) rồi `yaml.safe_load()` đúng một file. `scripts/evaluate.py:33-40` cũng đọc đúng một YAML bằng loader khác; không có bước merge.
- **Không có fallback sang default cho key bị lược khỏi experiment:** `TemporalAnalyzer` nhận đúng config path và `load_temporal_settings()` truy xuất các key bắt buộc (`src/temporal/temporal_analyzer.py:30-39`; `src/utils/config.py:71-97`). Nếu thiếu `temporal.*` hoặc các ngưỡng bắt buộc, kiểm tra cấu hình sẽ raise `ValueError`, không lấy từ default.
- **Cooldown:** `AlertManager` lấy `alert.cooldown` từ config qua `numeric_setting` (`src/alert/alert_manager.py:19-30`). Nếu key không có thì không có fallback default; cấu hình lỗi sẽ bị báo.
- **Trạng thái hiện tại của YAML:** `configs/experiment.yaml:26-32` khai báo cooldown `3.0` và temporal `5`/`1.5` trực tiếp. So sánh với `configs/default.yaml:11-16` cho thấy các giá trị trùng nhau, nhưng sự trùng khớp không phải bằng chứng kế thừa.
- **`drowsiness.mode`:** Không có trong experiment YAML. Với realtime CLI, mode được truyền qua `--mode`; với evaluation, mỗi mode được đặt vào `DecisionEngine` bởi `OfflinePipeline` (`src/pipeline/offline_pipeline.py:26-32`). Vì vậy `<MISSING>` trong dump experiment không tự nó là lỗi khi các entrypoint này cung cấp mode riêng.

## 5. Quét tham chiếu và phát hiện

Quét literal `default.yaml` / `experiment.yaml` trong các thư mục tồn tại `src/`, `scripts/`, `tests/` cho kết quả:

```text
src/pipeline/realtime_pipeline.py:173: ROOT / "configs/default.yaml" if config_path is None else Path(config_path)
src/utils/config.py:20: Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
scripts/evaluate.py:27: default="configs/experiment.yaml",
scripts/test_camera_features.py:30: CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
scripts/test_video_features.py:22: CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
tests/unit/T1001_261010_1030_test_frame_pipeline.py:141: """Mô tả: nạp default.yaml qua API thực tế load_config().
```

`experiments/` không tồn tại trong workspace nên không có nội dung để quét. `scripts/run_demo.py` cũng không tồn tại. Không thấy một entrypoint `run_demo.py` nào trong checkout.

Các điểm cần lưu ý:

1. **Hai loader độc lập:** `scripts/evaluate.py` không dùng loader chuẩn trong `src/utils/config.py`; quy tắc merge/validation vì thế không được chia sẻ tự động.
2. **Tài liệu lệnh evaluate lệch CLI:** ví dụ `AGENTS.md:251` đưa `--output`, nhưng `evaluate.py --help` không khai báo tùy chọn này; chạy nguyên ví dụ đó sẽ bị argparse từ chối.
3. **File không thuộc task:** Git status ban đầu và kiểm tra cuối cho thấy `configs/default.txt` và `configs/experiment.txt` là file chưa theo dõi. Báo cáo không đọc chúng làm cấu hình và không sửa/xóa chúng.
4. **Không chạy toàn bộ evaluation:** `evaluate.py` có thể tạo thư mục/kết quả và xóa `comparison_report.csv` cũ (`scripts/evaluate.py:187-192`), nên để giữ audit không phá hủy, chỉ chạy `--help`, parser và loader. Không chạy inference trên dataset.

## 6. Thay đổi và xác minh an toàn

- Artifact duy nhất được tạo: `docs/reports/T1101_261114_2031_config-routing-audit.md`.
- Không thay đổi `src/`, `scripts/`, `configs/`, dữ liệu hoặc kết quả evaluation.
- `git status` trước/sau vẫn ghi nhận hai file `.txt` chưa theo dõi nêu trên; chúng không do audit tạo ra.
- Thời gian chạy lệnh là chỉ đọc cấu hình/parser; không mở GUI/camera, không ghi log/kết quả pipeline.