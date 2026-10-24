import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# Đảm bảo import đúng đường dẫn từ thư mục gốc
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.camera import CameraStream
from src.detection.face_detector import FaceDetector
from src.detection.landmark_detector import LandmarkDetector

WINDOW_TITLE = "Detection Test - Camera"
EXIT_KEYS = {ord("q"), ord("Q"), 27}
EXIT_PROMPT = "Press 'q' / 'ESC' to exit"


def draw_hud(
    frame: np.ndarray, frame_id: int, live_fps: float, face_state: str
) -> None:
    """Vẽ thông số HUD lên màn hình."""
    telemetry = f"Frame: {frame_id} | Live FPS: {live_fps:.1f} | Face: {face_state}"
    color = (0, 255, 0) if face_state == "DETECTED" else (0, 0, 255)

    cv2.putText(frame, telemetry, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(
        frame,
        EXIT_PROMPT,
        (20, max(30, frame.shape[0] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )


def run_camera_detection(device_id: int) -> None:
    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    camera = None

    try:
        camera = CameraStream(device_id=device_id)
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        print(f" Camera stream opened on device: {device_id}")

        frame_count = 0
        for frame_id, timestamp, frame in camera:
            frame_count += 1
            live_fps = frame_count / timestamp if timestamp > 0.0 else 0.0

            # Xử lý Face Detection
            face_result = face_detector.process(frame)
            state = face_result["state"]

            if face_result["face_detected"]:
                x, y, w, h = face_result["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Xử lý Landmark
                lm_result = landmark_detector.process(frame)
                if lm_result["landmark_valid"]:
                    for pt in lm_result["left_eye_points"]:
                        cv2.circle(frame, pt, 2, (0, 0, 255), -1)
                    for pt in lm_result["right_eye_points"]:
                        cv2.circle(frame, pt, 2, (255, 0, 0), -1)
                    for pt in lm_result["mouth_points"]:
                        cv2.circle(frame, pt, 2, (0, 255, 255), -1)

            draw_hud(frame, frame_id, live_fps, state)
            cv2.imshow(WINDOW_TITLE, frame)

            if (cv2.waitKey(1) & 0xFF) in EXIT_KEYS or cv2.getWindowProperty(
                WINDOW_TITLE, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        if camera:
            camera.release()
        face_detector.release()
        landmark_detector.release()
        cv2.destroyAllWindows()
        print(" Cleanly released resources.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test Face & Landmark Detection on Webcam."
    )
    parser.add_argument("--device", type=int, default=0, help="Camera device index.")
    args = parser.parse_args(argv)

    try:
        run_camera_detection(args.device)
    except Exception as e:
        print(f" {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
