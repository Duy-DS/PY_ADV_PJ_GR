import numpy as np
import matplotlib.pyplot as plt

# 1. Khởi tạo dữ liệu thời gian (từ 0 đến 6 giây, tần số 100Hz để mượt mà)
t = np.linspace(0, 6, 600)
ear = np.full_like(t, 0.3)  # EAR trạng thái bình thường (mắt mở) là 0.3

# 2. Mô phỏng Phản xạ chớp mắt (Blink) từ giây 0.8 đến 1.1 (Khoảng 300ms)
# Tạo dạng sóng lõm chữ V/U nhanh
blink_mask = (t >= 0.8) & (t <= 1.1)
# Sử dụng hàm cosine để tạo độ lõm mượt mà xuống gần 0.03
ear[blink_mask] = 0.3 - 0.27 * (1 + np.cos(2 * np.pi * (t[blink_mask] - 0.95) / 0.3)) / 2

# 3. Mô phỏng Cơn ngủ gật (Drowsiness) kéo dài từ giây 2.5 đến 4.3 (Khoảng 1.8s >= 1.5s)
# Sóng trũng kéo dài vượt ngưỡng T_closure
drowsy_start, drowsy_end = 2.5, 4.3
drowsy_mask = (t >= drowsy_start) & (t <= drowsy_end)

for i, ti in enumerate(t):
    if drowsy_start <= ti <= drowsy_end:
        # Đoạn sụp mí mắt (xuống đáy trong 0.2s)
        if ti < drowsy_start + 0.2:
            ear[i] = 0.3 - 0.28 * (ti - drowsy_start) / 0.2
        # Đoạn mắt nhắm chặt kéo dài (nằm im dưới đáy 0.02)
        elif ti > drowsy_end - 0.2:
            ear[i] = 0.02 + 0.28 * (ti - (drowsy_end - 0.2)) / 0.2
        # Đoạn mở mắt trở lại (phục hồi trong 0.2s)
        else:
            ear[i] = 0.02

# 4. Cấu hình vẽ đồ thị
plt.figure(figsize=(11, 6))
plt.plot(t, ear, label='Tín hiệu EAR (Eye Aspect Ratio)', color='#1f77b4', linewidth=2.5)

# Vẽ đường ngưỡng nhắm mắt T_closure_threshold
plt.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Ngưỡng nhắm mắt (T_threshold = 0.15)')

# Vùng xám biểu thị thời gian nhắm mắt của cơn ngủ gật
plt.axvspan(drowsy_start, drowsy_end, color='orange', alpha=0.15, label='Thời gian nhắm mắt thực tế')

# 5. Thêm các ghi chú trực quan (Annotations)
# Ghi chú cho chớp mắt
plt.annotate('Phản xạ chớp mắt cực ngắn\n(100 - 400 ms)', 
             xy=(0.95, 0.03), xytext=(0.4, 0.08),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

# Ghi chú cho ngủ gật
plt.annotate('Sóng trũng kéo dài của ngủ gật\nT_closure = 1.8s (≥ 1.5s)', 
             xy=(3.4, 0.02), xytext=(2.6, 0.08),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))

# 6. Định dạng trục và hiển thị
plt.title('MÔ PHỎNG PHÂN TÍCH EAR: PHẢN XẠ CHỚP MẮT VS CƠN NGỦ GẬT', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Thời gian (giây)', fontsize=12)
plt.ylabel('Chỉ số EAR', fontsize=12)
plt.xlim(0, 6)
plt.ylim(0, 0.4)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)

# Hiển thị đồ thị lên màn hình
plt.tight_layout()
plt.show()