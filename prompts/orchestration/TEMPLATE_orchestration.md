# 🎛️ ORCHESTRATION & ROUTING PROTOCOL

## 1. GLOBAL STATE REVIEW
* Đọc dữ liệu từ file bộ nhớ động trạng thái phiên: `.memory/short_term/current_state.json`.

## 2. CONFLICT CHECKING TASK
* Hãy đóng vai trò Trọng tài Hệ thống (Orchestrator). Kiểm tra xem có sự mâu thuẫn tín hiệu giữa Technical Analyst Agent (TA) và Fundamental Analyst Agent (FA) không?

## 3. NEXT-ACTION ROUTING LOGIC
* Nếu `loop_count < 3` và chưa đạt được đồng thuận: Kích hoạt định tuyến quay lại phân hệ Analyst thích hợp kèm theo lý do phản biện.
* Nếu `loop_count == 3` (Circuit Breaker kích hoạt): Ép buộc chốt verdict dựa trên Ma trận trọng số cố định được cấu hình trong `AGENTS.md`.