# 🐛 SYSTEM DEBUGGING & BUG FIX MANDATE

## 1. ALERT &症状 (Hiện trạng lỗi)
* **Bug ID / Mã lỗi:** [Ví dụ: BUG-TOKEN-01]
* **Error Log Stacktrace:** ```text
  [Dán log lỗi hệ thống tại đây]

## 2. ROOT CAUSE ANALYSIS GUIDATED TASK
* Hãy cô lập vùng mã nguồn gây lỗi trong thư mục src/.

* Xác định xem lỗi do tràn cửa sổ ngữ cảnh (Context Exhaustion), lỗi floating-point tài chính, hay infinite loop của agent.

## 3. HOTFIX REQUIREMENTS
* Đề xuất bản vá thay đổi tối thiểu, bảo toàn tính tương thích ngược.

* Viết tệp báo cáo sửa lỗi lưu vào docs/bug-fixes/ theo quy tắc [YYMMDD]-[HHMM]-[Short_Name].md.