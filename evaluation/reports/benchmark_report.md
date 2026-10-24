# Báo cáo Benchmark Hệ thống Phát hiện Buồn ngủ

_Tạo tự động từ dữ liệu trong `results/`_

## 1. Tổng quan tập đánh giá
- **Tổng số ảnh**: 1319
- **Số đối tượng (subject)**: 3 — Sub1 (381), Sub2 (610), Sub3 (328)
- **Tỉ lệ phát hiện được khuôn mặt**: 100.00%
- **Lớp dương (positive)**: `DROWSY`

| Lớp | Nhãn | Số ảnh |
|---|---|---|
| Eyeclose | DROWSY | 618 |
| Neutral | NON_DROWSY | 421 |
| Yawn | DROWSY | 169 |
| Happy | NON_DROWSY | 111 |

## 2. So sánh chế độ phát hiện
| Chế độ | Accuracy | Precision | Recall | Specificity | F1-score |
|---|---|---|---|---|---|
| Eye only (EAR) | 0.7074 | 0.9239 | 0.5553 | 0.9323 | 0.6937 |
| Eye + Yawn (EAR + MAR) | 0.7074 | 0.9239 | 0.5553 | 0.9323 | 0.6937 |

**Ma trận nhầm lẫn** (hàng = nhãn thật, cột = dự đoán)

*Eye only (EAR)*

|  | Dự đoán: DROWSY | Dự đoán: NON_DROWSY |
|---|---|---|
| Thật: DROWSY | 437 | 350 |
| Thật: NON_DROWSY | 36 | 496 |

*Eye + Yawn (EAR + MAR)*

|  | Dự đoán: DROWSY | Dự đoán: NON_DROWSY |
|---|---|---|
| Thật: DROWSY | 437 | 350 |
| Thật: NON_DROWSY | 36 | 496 |

**Tỉ lệ bị dự đoán là DROWSY theo từng lớp**

| Lớp | Nhãn thật | Eye only (EAR) | Eye + Yawn (EAR + MAR) |
|---|---|---|---|
| Eyeclose | DROWSY | 70.71% (437/618) | 70.71% (437/618) |
| Yawn | DROWSY | 0.00% (0/169) | 0.00% (0/169) |
| Neutral | NON_DROWSY | 0.00% (0/421) | 0.00% (0/421) |
| Happy | NON_DROWSY | 32.43% (36/111) | 32.43% (36/111) |

> ⚠️ **Lưu ý**: hai chế độ cho dự đoán giống hệt nhau trên toàn bộ 1319 ảnh, tức là việc thêm MAR **chưa cải thiện** kết quả. Cụ thể, 169 ảnh `Yawn` có MAR trung bình 0.98 nhưng không ảnh nào bị gán `DROWSY`.

## 3. Hiệu năng thời gian thực
| Chế độ | Độ trễ TB (ms) | Trung vị (ms) | P95 (ms) | Max (ms) | FPS (1000/TB) | FPS (1000/P95) |
|---|---|---|---|---|---|---|
| Eye only (EAR) | 20.14 | 19.29 | 26.32 | 90.00 | 49.6 | 38.0 |
| Eye + Yawn (EAR + MAR) | 19.97 | 19.17 | 25.02 | 106.25 | 50.1 | 40.0 |

*FPS ở đây chỉ tính thời gian xử lý của pipeline (face → landmark → decision), chưa gồm thời gian đọc camera/ảnh và hiển thị.*

**Phân rã độ trễ theo module — Eye only (EAR)**

| Module | TB (ms) | Trung vị (ms) | P95 (ms) | Max (ms) | Tỉ trọng |
|---|---|---|---|---|---|
| Face Detection | 6.17 | 5.47 | 11.37 | 35.82 | 30.64% |
| Landmark / Feature Extraction (EAR, MAR) | 13.85 | 13.17 | 18.05 | 69.87 | 68.77% |
| Decision Engine | 0.12 | 0.10 | 0.16 | 1.13 | 0.59% |
| **Tổng pipeline** | 20.14 | 19.29 | 26.32 | 90.00 | 100.00% |

**Phân rã độ trễ theo module — Eye + Yawn (EAR + MAR)**

| Module | TB (ms) | Trung vị (ms) | P95 (ms) | Max (ms) | Tỉ trọng |
|---|---|---|---|---|---|
| Face Detection | 5.83 | 5.48 | 8.52 | 27.24 | 29.20% |
| Landmark / Feature Extraction (EAR, MAR) | 14.01 | 13.27 | 18.22 | 78.70 | 70.17% |
| Decision Engine | 0.12 | 0.10 | 0.19 | 1.11 | 0.62% |
| **Tổng pipeline** | 19.97 | 19.17 | 25.02 | 106.25 | 100.00% |

## 4. Đánh giá tính bền bỉ (Robustness)
Dòng đầu (`Baseline_Lighting`) được dùng làm mốc so sánh.

| Điều kiện | Accuracy | Precision | Recall | F1-score | Độ trễ (ms) | ΔF1 so với baseline | ΔRecall so với baseline |
|---|---|---|---|---|---|---|---|
| Baseline_Lighting | 0.7074 | 0.9239 | 0.5553 | 0.6937 | 19.97 | — | — |
| Low_Lighting | 0.6437 | 0.8869 | 0.4553 | 0.6035 | 20.37 | -0.0902 | -0.1000 |
| Glasses_On | 0.6861 | 0.9054 | 0.5275 | 0.6659 | 20.17 | -0.0278 | -0.0278 |
| Side_Angle_Face | 0.6083 | 0.8315 | 0.4276 | 0.5688 | 20.97 | -0.1249 | -0.1277 |

- Điều kiện làm giảm hiệu năng nhiều nhất: **Side_Angle_Face** (F1 0.5688, giảm 0.1249; Recall giảm 0.1277).
- Độ trễ gần như không đổi giữa các điều kiện (chênh lệch tối đa 1.00 ms).

## 5. Kết quả theo từng đối tượng (Eye + Yawn (EAR + MAR))
| Subject | Số ảnh | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Sub1 | 381 | 0.6588 | 1.0000 | 0.3953 | 0.5667 |
| Sub2 | 610 | 0.7984 | 0.8861 | 0.7629 | 0.8199 |
| Sub3 | 328 | 0.5945 | 1.0000 | 0.3512 | 0.5199 |

## 6. Kết luận
- Độ chính xác tổng thể 70.74%; Precision cao (92.39%) nhưng Recall thấp (55.53%) — hệ thống **ít báo động nhầm nhưng bỏ sót nhiều trường hợp buồn ngủ** (350/787 ca).
- Pipeline chạy khoảng 50 FPS (độ trễ TB 20.0 ms), đáp ứng ngưỡng thời gian thực 15 FPS.
- Bottleneck nằm ở module landmark (70.17% thời gian); Decision Engine gần như không đáng kể.
