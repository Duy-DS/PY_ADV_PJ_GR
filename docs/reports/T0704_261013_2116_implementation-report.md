# T0704 — Báo cáo triển khai và kiểm chứng

Ngày: 13/10/2026. Phạm vi: WP07 và WP12.
Workspace: `f:\Study Materials\specialized_subject\PY NC\PY_ADV_PJ_GR`.

## Kết quả bàn giao

Đã triển khai coordinator thật trong
`f:\Study Materials\specialized_subject\PY NC\PY_ADV_PJ_GR\src\temporal\temporal_analyzer.py`.

API:

```python
analyzer = TemporalAnalyzer(config_path=None)
result = analyzer.update(face_detected, timestamp, eye, mouth)
behavior = result.behavior
analyzer.reset()
```

eye và mouth phải là EyeMeasurement/MouthMeasurement. Kết quả là
TemporalAnalysisResult, không phải dictionary. Consumer cần kiểm tra
result.is_valid trước khi diễn giải behavior. action là CONTINUE/HOLD/RESET,
không phải kết luận buồn ngủ.

Các schema measurement/behavior khớp trường AGENTS.md, không thêm verdict.
TemporalAnalysisResult là envelope riêng cho thông tin validity/gap.

## Tệp triển khai

Tất cả đường dẫn dưới đây tính từ workspace tuyệt đối ghi ở đầu báo cáo:

- `src/temporal/temporal_analyzer.py`: kết hợp hai tracker, gap/timeout/reset.
- `src/temporal/eye_closure.py`: typed result, blink counter, suspend/resume/reset.
- `src/temporal/yawning_tracker.py`: typed result, window cấu hình, reset/gap.
- `src/utils/contracts.py`: ba schema đã quy định và envelope/helper UNKNOWN.
- `src/utils/config.py`: kiểm tra YAML, cấu hình typed và timestamp validation.
- `src/features/measurement_adapter.py`: EAR hai mắt thực, MAR và typed adapter.
- `scripts/test_detection_face_detected.py`: migrate caller GUI, ExitStack cleanup.
- `tests/unit/T0704_261013_2116_test_temporal.py`: 39 test cases khi parametrized.
- `notebooks/03_yawning_analysis.ipynb`: visualization synthetic, không lưu output.

Không sửa AGENTS.md, dependencies, data, models hoặc root layout. YAML do user
cập nhật trước phiên triển khai; agent giữ nguyên cấu hình đó. Không commit/push.
Audit/directive T0704_261013_2100 là bản ghi lịch sử trước khi YAML được cập nhật;
báo cáo này mô tả trạng thái triển khai mới nhất.

## Cấu hình đã nạp thành công

| Trường | Giá trị |
| --- | --- |
| eye.ear_threshold | 0.25 |
| eye.closure_time_threshold | 1.5 giây |
| mouth.mar_threshold | 0.6 |
| mouth.yawn_duration_threshold | 2.0 giây |
| mouth.yawn_frequency_window | 60.0 giây |
| temporal.missing_face_tolerance_frames | 5 frame |
| temporal.missing_face_timeout_sec | 1.5 giây |

Ngưỡng được nạp qua src/utils/config.py một lần khi khởi tạo coordinator và dùng
chung cho hai tracker/adapter. Thay đổi YAML cần tạo coordinator mới; không có
hot-reload trong lúc video đang chạy. Thiếu key hoặc giá trị sai dừng với lỗi
chứa tên trường, không fallback số trong đường xử lý T0704.

## Chính sách an toàn đã kiểm thử

- Bất kỳ nhánh invalid hoặc face thiếu làm cả hai phép đo hiện tại UNKNOWN.
- HOLD chỉ giữ buffers khi số frame thiếu <= N và thời gian từ frame hợp lệ
  cuối < timeout. Invalid output luôn is_valid=False, tắt cờ kéo dài/ngáp.
- Khoảng từ quan sát hợp lệ cuối đến frame phục hồi bị loại khỏi active timer.
  Không di chuyển timestamps sự kiện ngáp; window tiếp tục hết hạn khi HOLD.
- Tại timeout hoặc frame N+1, reset mọi timer, counter và lịch sử sự kiện.
  Kiểm tra timeout cả khi phục hồi; frame hợp lệ sau reset bắt đầu chu kỳ mới.
- UNKNOWN đầu luồng không có bằng chứng được trả RESET với hành vi zero.
- Typed API cần ít nhất hai quan sát và thời lượng dương để xác nhận hành vi,
  kể cả khi ngưỡng duration cấu hình bằng zero. Một CLOSED cô lập rồi OPEN
  muộn không tạo prolonged flag.
- Timestamp không hữu hạn, sai kiểu, bool hoặc lùi bị từ chối trước mutation.
  Timestamp trùng được phép nhưng không làm tăng thời lượng.
- reset() idempotent, xóa mốc thời gian, cho phép video mới bắt đầu tại zero.
- Coordinator không chứa quyết định DROWSY/NON_DROWSY, capture hay log giả.

## Tương thích và giới hạn

Wrapper update() của hai tracker vẫn trả dictionary legacy cho caller/test cũ.
Eye wrapper giữ semantics T0701 (frame mở mắt có thể trả cờ prolonged sau một
sample CLOSED và khoảng thời gian dài). Đây là giới hạn legacy được giữ có chủ
đích để không âm thầm phá test cũ; không được dùng đường này để quyết định cuối.
Coordinator dùng update_typed(confirm=True), không dùng wrapper legacy.
Những consumer mới phải migrate sang typed API bảo vệ frame đơn.

Legacy feature calculate_mouth_state() vẫn có fallback MAR cũ. Adapter mới không
đi qua fallback đó; đọc ngưỡng typed đã kiểm tra và sử dụng calculate_mar().

Không thể nhận biết decoder bỏ frame nếu caller không gửi UNKNOWN. Khoảng giữa
hai frame hợp lệ không tự động bị coi là missing vì không có metadata/threshold
được phê duyệt cho trường hợp đó. Invalid một nhánh gây invalid cả frame theo
chính sách bảo thủ, không duy trì inference mắt độc lập khi miệng invalid.

yawn_count là tổng số từ lần reset; yawn_frequency_in_window là số sự kiện trong
cửa sổ (giữ biên đúng window), không phải rate mỗi giây/phút. Mouth duration bằng
zero khi miệng đóng, phù hợp tracker cũ. Không thay đổi schema để thêm alias.

## Lệnh kiểm chứng và kết quả

- `python -m pytest tests/ -v`: exit 0, **133 passed in 8.02s**.
  Bao gồm 94 test baseline và 39 T0704 mới; không skipped/failed.
- `python -m black --line-length 79 <8 absolute changed Python paths>`:
  exit 0; lần cuối 8 files left unchanged.
- `python -m flake8 <8 absolute changed Python paths>`: exit 0, không findings.
- Kiểm tra AST docstrings Parameters/Returns/Raises: không thiếu; parse thành công.
- `python -m flake8 src/ tests/ --statistics --count`: exit 1,
  **44 findings ở tệp không sửa**, giảm từ 67 baseline.
- `git -c core.whitespace=cr-at-eol --no-pager diff --check`: exit 0.
  Git Windows có CRLF normalization; cấu hình này công nhận CR cuối dòng,
  không bỏ qua space/tab cuối dòng. Không sửa cấu hình Git hoặc gitattributes.
- nbformat.validate và NotebookClient.execute: exit 0, 3 code cells, 0 errors.
  Chỉ thực thi trong bộ nhớ, giữ notebook tracked không chứa output hình nhị phân.
  Kernel phát warning Windows event loop và TCP không mã hóa; không cell error.
- Smoke test D3S headless: exit 0, 20/20 frame hợp lệ qua landmark detector thật,
  typed geometry adapters và coordinator. Capture/detector được release,
  destroyAllWindows gọi và reset bộ đếm mắt/ngáp về zero.
  MediaPipe phát thông báo XNNPACK/feedback; không có lỗi xử lý frame.

Smoke test chỉ chứng minh đường dữ liệu chạy được, không kiểm chứng annotation,
ground truth hoặc accuracy. Không claim ứng dụng full pipeline vì src/main.py và
decision/alert/pipeline modules vẫn chưa được triển khai trong task này.

## Acceptance

**T0704 core implementation và regression/integration verification đã hoàn tất.**
FReq024 được kiểm chứng bằng default/custom YAML, missing/bad config và thay ngưỡng.
FReq027 được kiểm chứng qua typed duration, blink/yawn count và flags.
Các chính sách invalid/reset/cleanup có test boundary và mocked caller lifecycle.

M04 formal sign-off vẫn cần team review. Không claim SCri004/008/009 chính thức
khi tài liệu định nghĩa chuẩn chưa có trên workspace. Repo-wide zero-lint gate
chưa đạt do 44 findings ngoài phạm vi; mọi Python file sửa trong task lint-clean.
