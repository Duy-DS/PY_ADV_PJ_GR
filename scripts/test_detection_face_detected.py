"""Demo phát hiện khuôn mặt và hành vi typed T0704, không kết luận buồn ngủ."""

import sys
import tkinter as tk
from contextlib import ExitStack
from pathlib import Path
from tkinter import filedialog

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.camera import CameraStream  # noqa: E402
from src.acquisition.video_reader import VideoReader  # noqa: E402
from src.detection.face_detector import FaceDetector  # noqa: E402
from src.detection.landmark_detector import LandmarkDetector  # noqa: E402
from src.features.measurement_adapter import (  # noqa: E402
    measure_eye,
    measure_mouth,
)
from src.temporal.temporal_analyzer import TemporalAnalyzer  # noqa: E402
from src.utils.contracts import unknown_eye, unknown_mouth  # noqa: E402


def run_pipeline(use_camera: bool, video_path: str = ""):
    """Chạy demo, cleanup ngay cả khi khởi tạo thất bại hoặc EOF.

    Parameters: use_camera chọn camera; video_path là tệp video.
    Returns: Không có.
    Raises: OSError, ValueError hoặc lỗi detector; cleanup luôn được gọi.
    """
    with ExitStack() as cleanup:
        cleanup.callback(cv2.destroyAllWindows)
        analyzer = TemporalAnalyzer()
        cleanup.callback(analyzer.reset)
        detector = FaceDetector(
            model_path=str(
                REPO_ROOT / "models/pretrained/blaze_face_short_range.tflite"
            )
        )
        cleanup.callback(detector.release)
        landmarks = LandmarkDetector(
            model_path=str(
                REPO_ROOT / "models/pretrained/face_landmarker.task"
            )
        )
        cleanup.callback(landmarks.release)
        stream = (
            CameraStream(device_id=0)
            if use_camera
            else VideoReader(video_path)
        )
        source = cleanup.enter_context(stream)
        for frame_id, timestamp, image in source:
            detection = detector.process(image)
            face = bool(detection.get("face_detected", False))
            eye, mouth = unknown_eye(), unknown_mouth()
            if face:
                mesh = landmarks.process(image)
                if mesh.get("landmark_valid", False):
                    eye = measure_eye(
                        mesh["left_eye_points"],
                        mesh["right_eye_points"],
                        analyzer.settings,
                    )
                    mouth = measure_mouth(
                        mesh["mouth_points"], analyzer.settings
                    )
            result = analyzer.update(face, timestamp, eye, mouth)
            text = (
                f"Frame {frame_id} | {timestamp:.2f}s | {result.action} | "
                f"Eye {result.eye.eye_state} | "
                f"Mouth {result.mouth.mouth_state}"
            )
            cv2.putText(
                image,
                text,
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Temporal behavior T0704", image)
            fps = getattr(stream, "fps", 30)
            delay = 1 if use_camera else max(1, int(1000 / max(fps, 1)))
            if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                break


def show_gui():
    """Hiển thị lựa chọn webcam/video.

    Parameters: Không có.
    Returns: Không có.
    Raises: tkinter.TclError nếu môi trường không hỗ trợ GUI.
    """
    root = tk.Tk()
    root.title("Temporal behavior T0704")
    root.geometry("360x160")

    def on_webcam_click():
        """Đóng GUI và chạy camera.

        Parameters: Không có.
        Returns: Không có.
        Raises: Lỗi pipeline được truyền cho caller.
        """
        root.destroy()
        run_pipeline(True)

    def on_video_click():
        """Chọn tệp và chạy video.

        Parameters: Không có.
        Returns: Không có.
        Raises: Lỗi pipeline được truyền cho caller.
        """
        path = filedialog.askopenfilename(
            title="Chọn video",
            initialdir=str(REPO_ROOT / "data/raw/d3s"),
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov")],
        )
        if path:
            root.destroy()
            run_pipeline(False, path)

    tk.Button(root, text="Webcam", command=on_webcam_click).pack(pady=15)
    tk.Button(root, text="Chọn video", command=on_video_click).pack(pady=5)
    root.mainloop()


if __name__ == "__main__":
    show_gui()
