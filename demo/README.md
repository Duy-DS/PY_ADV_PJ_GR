## 1. Nội dung thư mục

```text
demo/
├── README.md
└── videos/
    ├── demo_video_dataset.mp4   # Demo 1: chạy trên video từ dataset
    └── demo_video_camera.mp4    # Demo 2: chạy trực tiếp trên camera
```

## 2. Mô tả hai video demo

| | Demo 1 – Video dataset | Demo 2 – Camera thời gian thực |
|---|---|---|
| **Nguồn đầu vào** | Video có sẵn từ bộ dữ liệu | Webcam, quay trực tiếp |
| **Mục đích** | Kiểm chứng hệ thống trên dữ liệu có kịch bản hành vi đã biết | Kiểm chứng khả năng hoạt động thời gian thực trong điều kiện thực tế |
| **File** | `videos/demo_video_dataset.mp4` | `videos/demo_video_camera.mp4` |

Các video minh họa 5 kịch bản hành vi chính của hệ thống:

1. **Normal** – trạng thái tỉnh táo bình thường
2. **Blink** – chớp mắt thông thường
3. **Prolonged Closure** – nhắm mắt kéo dài
4. **Yawn** – ngáp
5. **Alert** – hệ thống phát cảnh báo bằng âm thanh

## 3. Kết quả chạy

### 3.1. Demo 1 – Chạy trên video dataset

| Chỉ số | Giá trị |
|---|---|
| Tổng số khung hình đã xử lý | 997 |
| Số khung hình báo buồn ngủ | 74  |
| Số lần bật âm thanh cảnh báo | 4 |
| Tốc độ xử lý (FPS trung bình) | 20,1 |
| Độ trễ trung bình | 32,7 ms |
| Tổng thời gian hoạt động | 49,7 giây |

### 3.2. Demo 2 – Chạy trên camera

| Chỉ số | Giá trị |
|---|---|
| Tổng số khung hình đã xử lý | 823 |
| Số lần bật âm thanh cảnh báo | 2 |
| Tốc độ xử lý (FPS trung bình) | 21,7 |
| Độ trễ trung bình | 29,8 ms |
| Tổng thời gian hoạt động | 37,9 giây |

### 3.3. Nhận xét

- Cả hai lần chạy đều đạt tốc độ xử lý trên **20 FPS** với độ trễ trung bình dưới **33 ms** mỗi khung hình, đáp ứng yêu cầu xử lý thời gian thực.
- Chế độ camera nhanh hơn một chút (21,7 FPS, 29,8 ms) so với chế độ video dataset (20,1 FPS, 32,7 ms).
- Hệ thống phát cảnh báo âm thanh đúng khi phát hiện trạng thái buồn ngủ: 4 lần trong demo dataset và 2 lần trong demo camera.

## 4. Ghi chú

- Số liệu trên đo trong một lần chạy cụ thể; FPS và độ trễ có thể thay đổi tùy cấu hình phần cứng.
- Video demo có độ phân giải cao để thể hiện rõ các kịch bản hành vi.