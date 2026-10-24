# 🏗️ Kiến Trúc Hệ Thống: [Tiêu đề Thiết kế / Số ADR]

## 1. 🎯 Bối Cảnh & Bài Toán (Context)
* [Tại sao chúng ta cần đưa ra thiết kế hoặc thay đổi kiến trúc này? Nó giải quyết rào cản tài chính hay kỹ thuật nào?]

## 2. 🗺️ Sơ Đồ Khái Niệm / Sơ Đồ Thực Thể (Design & Schema)
* **Sơ đồ luồng (Data Flow ASCII / Mermaid):**
  ```mermaid
  graph TD
      A[User Telegram] -->|Voice/Text| B(Reporting Agent)
      B --> C{Orchestrator}

* Cấu trúc bảng dữ liệu (Nếu có): [Mô tả chi tiết các trường, kiểu dữ liệu khi lưu vào PostgreSQL/pgvector]

## 3. ⚖️ Đánh Giá Hệ Quả (Consequences)
* Điểm đạt được: [Tối ưu chi phí 0 USD, tăng tốc độ truy vấn RAG dưới 2 giây]

* Hạn chế tồn tại: [Tăng dung lượng lưu trữ cục bộ]

* Trạng thái quyết định: Đang đề xuất (Proposed) / Đã chấp thuận (Accepted)