"""Script tích hợp và demo detection lên Video/Camera (Task T1001)."""

import cv2
import sys
from pathlib import Path

# Đảm bảo import đúng đường dẫn từ thư mục gốc
REPO_ROOT = (
    Path(__file__).resolve().parents[2]
)  # Điều chỉnh số 2 tùy vị trí đặt file
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.camera import CameraStream  # noqa: E402
from src.acquisition.video_reader import VideoReader  # noqa: E402
from src.detection.face_detector import FaceDetector  # noqa: E402
from src.detection.landmark_detector import LandmarkDetector  # noqa: E402


def run_demo(use_camera: bool = True, video_path: str = ""):
    # 1. Khởi tạo các module Detection
    face_detector = FaceDetector(min_detection_confidence=0.5)
    landmark_detector = LandmarkDetector(min_detection_confidence=0.5)

    # 2. Khởi tạo nguồn Input (Webcam hoặc Video file)
    try:
        stream = (
            CameraStream(device_id=0)
            if use_camera
            else VideoReader(video_path)
        )
    except Exception as e:
        print(f"[ERROR] Không thể mở input: {e}")
        return

    print("[INFO] Bấm 'q' hoặc 'ESC' trên cửa sổ video để thoát.")

    # 3. Chạy vòng lặp xử lý từng frame
    with stream as source:
        for frame_id, timestamp, frame in source:
            # Phát hiện khuôn mặt
            face_result = face_detector.process(frame)

            if face_result["face_detected"]:
                # Vẽ Bounding Box khuôn mặt (Màu xanh lá)
                x, y, w, h = face_result["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Trích xuất Landmark
                landmark_result = landmark_detector.process(frame)

                if landmark_result["landmark_valid"]:
                    # Vẽ mắt trái (Đỏ)
                    for pt in landmark_result["left_eye_points"]:
                        cv2.circle(frame, pt, 2, (0, 0, 255), -1)
                    # Vẽ mắt phải (Xanh dương)
                    for pt in landmark_result["right_eye_points"]:
                        cv2.circle(frame, pt, 2, (255, 0, 0), -1)
                    # Vẽ viền môi (Vàng)
                    for pt in landmark_result["mouth_points"]:
                        cv2.circle(frame, pt, 2, (0, 255, 255), -1)
            else:
                # Báo trạng thái UNKNOWN khi mất dấu khuôn mặt (FReq004)
                cv2.putText(
                    frame,
                    "UNKNOWN - No Face",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Hiển thị FPS và Frame ID
            cv2.putText(
                frame,
                f"Frame: {frame_id} | Time: {timestamp:.1f}s",
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # 4. Hiển thị đè lên màn hình
            cv2.imshow("Driver Drowsiness - Detection PoC", frame)

            # Delay phù hợp để xem video, 1ms cho camera
            delay = (
                1
                if use_camera
                else max(1, int(1000 / getattr(stream, "fps", 30)))
            )
            key = cv2.waitKey(delay) & 0xFF
            if key in [ord("q"), 27]:
                break

    # 5. Giải phóng tài nguyên
    face_detector.release()
    landmark_detector.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng pipeline.")


if __name__ == "__main__":
    # Để test camera laptop
    run_demo(use_camera=True)

    # Để test video từ D3S Dataset, comment dòng trên và uncomment dòng dưới:
    # run_demo(use_camera=False,
    # video_path="data/raw/d3s/d3s_video_dataset.mp4")
