"""Script tích hợp và demo detection lên Video/Camera có giao diện nút bấm (Task T1001)."""

import cv2
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# Đảm bảo import đúng đường dẫn từ thư mục gốc
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.camera import CameraStream
from src.acquisition.video_reader import VideoReader
from src.detection.face_detector import FaceDetector
from src.detection.landmark_detector import LandmarkDetector


def run_pipeline(use_camera: bool, video_path: str = ""):
    """Chạy pipeline xử lý trên nguồn dữ liệu được chọn."""
    print("[INFO] Đang khởi tạo mô hình AI...")
    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()

    try:
        if use_camera:
            print("[INFO] Đang kết nối Webcam...")
            stream = CameraStream(device_id=0)
        else:
            print(f"[INFO] Đang mở video: {video_path}")
            stream = VideoReader(video_path)
    except Exception as e:
        print(f"[ERROR] Không thể mở dữ liệu đầu vào: {e}")
        return

    print("[INFO] Bấm phím 'q' hoặc 'ESC' trên cửa sổ video để thoát.")

    with stream as source:
        for frame_id, timestamp, frame in source:
            # 1. Phát hiện khuôn mặt
            face_result = face_detector.process(frame)

            if face_result.get("face_detected", False):
                # Vẽ Bounding Box khuôn mặt
                x, y, w, h = face_result["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 2. Trích xuất Landmark
                lm_result = landmark_detector.process(frame)
                if lm_result.get("landmark_valid", False):
                    for pt in lm_result["left_eye_points"]:
                        cv2.circle(frame, pt, 2, (0, 0, 255), -1)  # Mắt trái: Đỏ
                    for pt in lm_result["right_eye_points"]:
                        cv2.circle(frame, pt, 2, (255, 0, 0), -1)  # Mắt phải: Xanh
                    for pt in lm_result["mouth_points"]:
                        cv2.circle(frame, pt, 2, (0, 255, 255), -1)  # Miệng: Vàng
            else:
                cv2.putText(
                    frame,
                    "UNKNOWN - No Face",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Hiển thị thông số
            cv2.putText(
                frame,
                f"Frame: {frame_id} | Time: {timestamp:.1f}s",
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Driver Drowsiness - Pipeline Test", frame)

            # Tính toán thời gian delay để video chạy đúng tốc độ thực
            fps = getattr(stream, "fps", 30)
            if fps <= 0:
                fps = 30
            delay = 1 if use_camera else max(1, int(1000 / fps))

            key = cv2.waitKey(delay) & 0xFF
            if key in [ord("q"), 27]:
                break

    face_detector.release()
    landmark_detector.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng pipeline an toàn.")


def show_gui():
    """Hiển thị giao diện nút bấm để chọn đầu vào."""
    root = tk.Tk()
    root.title("Chọn Nguồn Dữ Liệu")

    # Canh giữa màn hình
    window_width, window_height = 320, 160
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

    def on_webcam_click():
        root.destroy()
        run_pipeline(use_camera=True)

    def on_video_click():
        # Mở hộp thoại chọn file video
        file_path = filedialog.askopenfilename(
            title="Chọn file Video",
            initialdir=str(REPO_ROOT / "data" / "raw" / "d3s"),
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")],
        )
        if file_path:
            root.destroy()
            run_pipeline(use_camera=False, video_path=file_path)

    # Thêm text và các nút bấm
    tk.Label(
        root, text="Vui lòng chọn nguồn dữ liệu:", font=("Arial", 11, "bold")
    ).pack(pady=15)

    tk.Button(
        root,
        text="Sử dụng Webcam",
        font=("Arial", 10),
        bg="#2ecc71",
        fg="white",
        width=25,
        command=on_webcam_click,
    ).pack(pady=5)

    tk.Button(
        root,
        text="Chọn Video từ máy tính",
        font=("Arial", 10),
        bg="#3498db",
        fg="white",
        width=25,
        command=on_video_click,
    ).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    show_gui()
