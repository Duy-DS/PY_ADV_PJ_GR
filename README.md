# Driver Drowsiness Detection System

> Image Processing & Computer Vision - Python Nâng Cao

[![Status](https://img.shields.io/badge/status-active-brightgreen)](#)
[![Team](https://img.shields.io/badge/team-4%20members-blue)](#)
[![Duration](https://img.shields.io/badge/duration-6%20weeks-orange)](#)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB)](#)
[![Repository](https://img.shields.io/badge/repository-PY_ADV_PJ_GR-black)](https://github.com/Duy-DS/PY_ADV_PJ_GR)

**Repository:** <https://github.com/Duy-DS/PY_ADV_PJ_GR>

## Tổng Quan

Driver Drowsiness Detection System là hệ thống phát hiện dấu hiệu buồn ngủ hoặc ngủ gật của tài xế thông qua video/camera. Hệ thống tập trung vào các tín hiệu thị giác trên khuôn mặt, đặc biệt là trạng thái mắt và miệng, sau đó phân tích theo chuỗi thời gian để đưa ra quyết định cảnh báo.

Thách thức trung tâm của bài toán là phân biệt hành vi bình thường như chớp mắt ngắn, mở miệng tự nhiên với hành vi buồn ngủ như nhắm mắt kéo dài (*prolonged eye closure*) hoặc ngáp lặp lại/kéo dài. Vì vậy, hệ thống không chỉ dựa vào một frame đơn lẻ mà cần theo dõi hành vi liên tục qua nhiều frame.

## Problem Statement & Motivation

| Hạng mục | Nội dung |
| --- | --- |
| Bài toán | Phát hiện trạng thái buồn ngủ/ngủ gật của tài xế từ video hoặc camera thời gian thực. |
| Đầu vào | Video file, webcam hoặc camera stream. |
| Đầu ra | Trạng thái tài xế: `NORMAL`, `DROWSY`, `UNKNOWN` và tín hiệu cảnh báo nếu cần. |
| Giá trị ứng dụng | Hỗ trợ cảnh báo sớm trong ngữ cảnh lái xe, giảm rủi ro do mất tập trung và buồn ngủ. |
| Trọng tâm kỹ thuật | Computer Vision, Facial Landmark Detection, EAR/MAR, temporal behavior analysis. |

## Research Questions

1. Đặc trưng mắt như Eye Aspect Ratio (EAR) và thời gian nhắm mắt liên tục có đủ để phân biệt chớp mắt bình thường với buồn ngủ hay không?
2. Việc kết hợp thêm đặc trưng miệng như Mouth Aspect Ratio (MAR) và yawning duration có cải thiện khả năng phát hiện so với eye-only detection không?
3. Cần thiết lập ngưỡng và cửa sổ thời gian như thế nào để giảm false positive trong các tình huống mất khuôn mặt, landmark lỗi, ánh sáng yếu hoặc góc nhìn thay đổi?

## Core Scope

| Nhóm đặc trưng | Phạm vi xử lý | Trạng thái |
| --- | --- | --- |
| Eyes | Eye state, EAR, blink/prolonged closure, closure duration | Core |
| Mouth | Mouth state, MAR, yawning duration, yawning frequency | Core |
| Head Pose | Ước lượng góc nghiêng đầu và head movement | Deferred / Extension |

Hệ thống ưu tiên hai cấu hình thực nghiệm chính:

| Experiment | Mô tả | Mục tiêu |
| --- | --- | --- |
| Experiment A | Eye-only detection | Đánh giá khả năng phát hiện buồn ngủ chỉ bằng đặc trưng mắt. |
| Experiment B | Eye + Yawning multi-feature combination | So sánh mức cải thiện khi kết hợp đặc trưng mắt và ngáp. |

## System Pipeline

```text
Input Acquisition
        |
        v
Face Detection
        |
        v
Facial Landmark Detection
        |
        v
Feature Extraction (EAR & MAR)
        |
        v
Temporal Behavior Analysis
        |
        v
Drowsiness Decision Engine (Mode A & B)
        |
        v
Alert & Recovery Manager
        |
        v
Visualization / Logging
```

| Tầng xử lý | Vai trò |
| --- | --- |
| Input Acquisition | Đọc webcam, camera stream hoặc video dataset. |
| Face Detection | Xác định vùng khuôn mặt của tài xế trong từng frame. |
| Facial Landmark Detection | Trích xuất điểm mốc khuôn mặt bằng Face Mesh. |
| Feature Extraction | Tính EAR cho mắt và MAR cho miệng. |
| Temporal Behavior Analysis | Theo dõi liên tục thời gian nhắm mắt, thời lượng ngáp và tần suất ngáp. |
| Decision Engine | Đưa ra kết luận theo Mode A (Eye-only) hoặc Mode B (Eye + Yawn). |
| Alert & Recovery Manager | Quản lý cảnh báo, cooldown và trạng thái phục hồi. |
| Visualization / Logging | Vẽ overlay, chỉ số EAR/MAR, trạng thái và ghi log thực nghiệm. |

**Nguyên tắc xử lý lỗi:** khi không phát hiện được khuôn mặt (`Face not detected`) hoặc landmarks không hợp lệ, hệ thống phải trả về trạng thái `UNKNOWN`. Hệ thống không được suy luận các frame lỗi thành `DROWSY`.

## Project Structure

```text
PY_ADV_PJ_GR/
├── configs/                            # Thư mục cấu hình tham số hệ thống
│   ├── default.yaml                    # Tham số mặc định: ngưỡng EAR, MAR, thời gian closure, alert
│   ├── development.yaml                # Cấu hình môi trường dev/debug cục bộ
│   └── experiment.yaml                 # Cấu hình tham số cho các kịch bản thực nghiệm
├── data/                               # Quản lý dữ liệu dự án, bỏ qua dữ liệu nặng bằng .gitignore
│   ├── raw/                            # Chứa dữ liệu gốc, không chỉnh sửa trực tiếp
│   │   ├── d3s/                        # Dataset chính Driver Drowsiness Dataset D3S
│   │   ├── mrl_eye/                    # Dataset phụ trợ trạng thái mắt, dùng tham khảo/dự phòng
│   │   └── roboflow_drowsiness/        # Dataset tham khảo từ Roboflow cho bài toán drowsiness
│   ├── processed/                      # Dữ liệu đã trích xuất hoặc tiền xử lý sẵn
│   ├── samples/                        # Ảnh/video ngắn phục vụ test nhanh
│   └── self_collected/                 # Video/ảnh tự thu thập để kiểm thử thực tế
├── demo/                               # Tài nguyên phục vụ demo cuối kỳ
│   ├── images/                         # Ảnh chụp màn hình hoặc minh họa kết quả demo
│   ├── videos/                         # Video clip demo hoạt động của hệ thống
│   └── README.md                       # Hướng dẫn trình diễn demo
├── docs/                               # Tài liệu dự án chi tiết
│   ├── architecture/                   # Tài liệu thiết kế kiến trúc hệ thống và pipeline
│   ├── dataset/                        # Phân tích D3S, label mapping và split strategy
│   ├── experiments/                    # Kế hoạch và quy trình thực nghiệm
│   ├── meetings/                       # Biên bản các cuộc họp nhóm
│   ├── project/                        # Quản lý dự án: decision log, progress log, open questions
│   └── testing/                        # Chiến lược kiểm thử phần mềm
├── evaluation/                         # Đánh giá và đo lường hệ thống
│   ├── metrics/                        # Code tính metrics: Accuracy, F1, Recall, FPS, Latency
│   ├── reports/                        # Báo cáo đánh giá hiệu năng
│   ├── results/                        # File kết quả thực nghiệm dạng CSV/Log
│   └── test_cases/                     # Bộ test case đánh giá hệ thống
├── experiments/                        # Code và kịch bản chạy thực nghiệm
│   ├── configs/                        # Cấu hình riêng cho từng bài test
│   ├── eye_only/                       # Experiment A: chỉ sử dụng đặc trưng mắt
│   ├── eye_yawn/                       # Experiment B: kết hợp đặc trưng mắt và ngáp
│   ├── multimodal/                     # Thực nghiệm mở rộng đa đặc trưng
│   └── robustness/                     # Kiểm thử độ ổn định: ánh sáng yếu, đeo kính, góc nghiêng
├── models/                             # Lưu trữ trọng số mô hình nếu sử dụng ML/DL
│   ├── checkpoints/                    # Checkpoint trong quá trình huấn luyện
│   ├── pretrained/                     # Model weights có sẵn từ nguồn mở
│   └── trained/                        # Trọng số mô hình sau huấn luyện
├── notebooks/                          # Jupyter Notebooks phục vụ EDA và thử nghiệm thuật toán
│   ├── 01_dataset_exploration.ipynb    # Khám phá cấu trúc và phân phối nhãn của dataset D3S
│   ├── 02_eye_analysis.ipynb           # Thử nghiệm trích xuất mốc mắt và tính EAR
│   ├── 03_yawning_analysis.ipynb       # Thử nghiệm tính MAR và phát hiện ngáp
│   ├── 04_head_pose_analysis.ipynb     # Thử nghiệm mở rộng tính góc nghiêng đầu
│   └── 05_experiments.ipynb            # Chạy thử và so sánh các thực nghiệm
├── reports/                            # Báo cáo kết quả môn học
│   ├── figures/                        # Biểu đồ và hình ảnh minh họa
│   ├── final_report/                   # Báo cáo môn học cuối kỳ hoàn chỉnh
│   ├── presentation/                   # Slide thuyết trình
│   └── tables/                         # Bảng số liệu thống kê
├── scripts/                            # Các kịch bản tự động hóa
│   ├── download_dataset.py             # Script hỗ trợ tải dataset D3S
│   ├── evaluate.py                     # Script chạy đo lường và đánh giá tự động
│   ├── preprocess_data.py              # Script tiền xử lý dữ liệu hàng loạt
│   └── run_demo.py                     # Script khởi chạy demo nhanh
├── src/                                # Mã nguồn chính của hệ thống
│   ├── acquisition/                    # Tầng nhận dữ liệu đầu vào
│   │   ├── camera.py                   # Quản lý luồng webcam thời gian thực
│   │   └── video_reader.py             # Đọc và duyệt frame từ file video dataset
│   ├── alert/                          # Quản lý cơ chế cảnh báo
│   │   ├── alarm.py                    # Logic kích hoạt âm thanh/tín hiệu cảnh báo
│   │   └── alert_manager.py            # Quản lý trạng thái cảnh báo, cooldown và recovery
│   ├── detection/                      # Tầng phát hiện khuôn mặt và điểm mốc
│   │   ├── face_detector.py            # Phát hiện vùng khuôn mặt của tài xế
│   │   └── landmark_detector.py        # Trích xuất 468/478 điểm mốc khuôn mặt bằng Face Mesh
│   ├── drowsiness/                     # Bộ suy luận trạng thái buồn ngủ
│   │   ├── classifier.py               # Mô hình phân loại học máy nếu mở rộng
│   │   ├── decision_engine.py          # Kết hợp tín hiệu thời gian từ mắt và miệng để kết luận
│   │   └── rule_based.py               # Logic quy tắc ngưỡng cơ bản
│   ├── features/                       # Trích xuất đặc trưng hình học frame-level
│   │   ├── eye.py                      # Tính Eye Aspect Ratio (EAR)
│   │   ├── feature_extractor.py        # Tổng hợp các đặc trưng khuôn mặt trong frame
│   │   ├── head_pose.py                # Ước lượng góc nghiêng đầu, module mở rộng
│   │   └── mouth.py                    # Tính Mouth Aspect Ratio (MAR)
│   ├── pipeline/                       # Điều phối luồng xử lý tổng thể
│   │   ├── offline_pipeline.py         # Pipeline đánh giá hàng loạt trên video file dataset
│   │   └── realtime_pipeline.py        # Pipeline xử lý trực tiếp từ webcam/camera
│   ├── temporal/                       # Phân tích hành vi theo chuỗi thời gian
│   │   ├── eye_closure.py              # Theo dõi thời gian nhắm mắt liên tục
│   │   ├── head_tracker.py             # Theo dõi chuyển động đầu theo thời gian
│   │   ├── temporal_analyzer.py        # Điều phối phân tích hành vi theo thời gian
│   │   └── yawning_tracker.py          # Theo dõi thời lượng và tần suất ngáp
│   ├── utils/                          # Tiện ích dùng chung
│   │   ├── config.py                   # Đọc và nạp cấu hình từ file YAML
│   │   ├── geometry.py                 # Tính khoảng cách Euclidean, vector và tỷ lệ hình học
│   │   ├── logger.py                   # Ghi log quá trình chạy của hệ thống
│   │   └── timer.py                    # Đo FPS và latency xử lý
│   ├── visualization/                  # Trực quan hóa kết quả lên màn hình
│   │   ├── dashboard.py                # Dashboard theo dõi chỉ số mở rộng
│   │   └── overlay.py                  # Vẽ landmark, bounding box, EAR/MAR và cảnh báo lên frame
│   └── main.py                         # Entry point chính của hệ thống
├── tests/                              # Bộ kiểm thử mã nguồn
│   ├── integration/                    # Kiểm thử tích hợp luồng pipeline
│   ├── test_data/                      # Dữ liệu mẫu dùng cho unit test
│   └── unit/                           # Kiểm thử đơn vị cho từng hàm trích xuất và tính toán
├── .env.example                        # Mẫu cấu hình biến môi trường
├── .gitignore                          # Cấu hình bỏ qua các file không đưa lên Git
├── requirements.txt                    # Danh sách các thư viện Python cần cài đặt
└── README.md                           # Tài liệu tổng quan và hướng dẫn sử dụng dự án
```

## Dataset

Primary dataset của dự án là **Driver Drowsiness Dataset (D3S)**. Dataset hỗ trợ các nhóm nhãn quan trọng cho bài toán phát hiện buồn ngủ, bao gồm mắt nhắm, ngáp, trạng thái bình thường và biểu cảm vui vẻ; dữ liệu có thể được sử dụng ở dạng video và frames trích xuất.

| Thành phần | Mô tả |
| --- | --- |
| Primary dataset | Driver Drowsiness Dataset (D3S) |
| Tín hiệu cần khai thác | Eye state, mouth state, yawning behavior, normal behavior |
| Đơn vị xử lý | Video-level và frame-level |
| Nguyên tắc split | Chia theo subject/video để tránh data leakage |
| Đầu ra đánh giá | So sánh Mode A và Mode B trên cùng tập kiểm thử |

**Lưu ý về data leakage:** không chia ngẫu nhiên từng frame nếu các frame đó đến từ cùng một subject hoặc cùng một video. Việc tách train/validation/test cần được thực hiện theo subject/video để đảm bảo đánh giá công bằng.

## Installation & Usage

### 1. Clone repository

```bash
git clone https://github.com/Duy-DS/PY_ADV_PJ_GR.git
cd PY_ADV_PJ_GR
```

### 2. Tạo môi trường ảo

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Chạy offline pipeline trên video

```bash
# Chế độ A (eye_only — Chỉ xét dấu hiệu Mắt nhắm kéo dài):
python src/main.py --source data/raw/d3s/d3s_video_dataset.mp4 --mode eye_only --config configs/default.yaml
```


```bash
# Chế độ B (eye_yawn — Kết hợp cả Mắt nhắm kéo dài + Ngáp):
python src/main.py --source data/raw/d3s/d3s_video_dataset.mp4 --mode eye_yawn --config configs/default.yaml
```

### 5. Chạy realtime webcam

```bash
python src/main.py --source webcam --mode eye_yawn --config configs/default.yaml
```

### 6. Chạy thực nghiệm đánh giá

```bash
python scripts/evaluate.py --config configs/experiment.yaml
```

## Technology Stack

| Nhóm công nghệ | Công cụ |
| --- | --- |
| Ngôn ngữ | Python 3.9+ |
| Computer Vision | OpenCV, MediaPipe Face Mesh |
| Data Science | NumPy, Pandas, Scikit-learn, Matplotlib |
| Configuration | PyYAML |
| Evaluation | Accuracy, Precision, Recall, F1-score, FPS, Latency |

## Evaluation Criteria

| Tiêu chí | Ý nghĩa |
| --- | --- |
| Accuracy | Tỷ lệ dự đoán đúng trên tập kiểm thử. |
| Precision | Độ tin cậy của các cảnh báo `DROWSY`. |
| Recall | Khả năng phát hiện đúng các trường hợp buồn ngủ. |
| F1-score | Cân bằng giữa Precision và Recall. |
| FPS | Tốc độ xử lý, phù hợp cho realtime hay không. |
| Latency | Độ trễ từ lúc nhận frame đến lúc đưa ra quyết định/cảnh báo. |
| False Positive Rate | Tỷ lệ cảnh báo sai trong hành vi bình thường. |

## Milestones

| Tuần | Mục tiêu |
| --- | --- |
| Week 1 | Chốt problem statement, nghiên cứu dataset D3S, thiết lập repository và baseline pipeline. |
| Week 2 | Hoàn thiện face detection, landmark detection và trích xuất EAR/MAR frame-level. |
| Week 3 | Xây dựng temporal analysis cho eye closure và yawning tracking. |
| Week 4 | Hoàn thiện decision engine cho Experiment A và Experiment B. |
| Week 5 | Đánh giá, tính metrics, kiểm thử robustness và tối ưu ngưỡng. |
| Week 6 | Hoàn thiện report, slide, video demo và bàn giao source code. |

## Deliverables

| Hạng mục | Kết quả bàn giao |
| --- | --- |
| Source code | Mã nguồn Python được tổ chức theo module rõ ràng, dễ mở rộng và kiểm thử. |
| Experiment report | Báo cáo so sánh Mode Eye-only và Eye + Yawn dựa trên metrics định lượng. |
| Demo video | Video minh họa pipeline hoạt động trên video/webcam. |
| Documentation | README, tài liệu kiến trúc, dataset, testing và hướng dẫn chạy hệ thống. |

## Quy Ước Trạng Thái

| Trạng thái | Ý nghĩa |
| --- | --- |
| `NORMAL` | Tài xế ở trạng thái bình thường, không có dấu hiệu buồn ngủ vượt ngưỡng. |
| `DROWSY` | Phát hiện dấu hiệu buồn ngủ dựa trên closure/yawning theo chuỗi thời gian. |
| `UNKNOWN` | Không đủ dữ liệu tin cậy do mất khuôn mặt, landmarks lỗi hoặc frame không hợp lệ. |

Hệ thống chỉ kích hoạt cảnh báo khi tín hiệu drowsiness được xác nhận qua temporal analysis. Các frame `UNKNOWN` cần được ghi log để phục vụ đánh giá chất lượng nhận diện và không được tính như bằng chứng trực tiếp cho trạng thái `DROWSY`.
