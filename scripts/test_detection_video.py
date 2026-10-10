"""Interactively verify T0401 & T0402 face/landmark detection on a video file."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.video_reader import VideoReader
from src.detection.face_detector import FaceDetector
from src.detection.landmark_detector import LandmarkDetector

DATASET_DIR = REPO_ROOT / "data" / "raw" / "d3s"
DEFAULT_VIDEO = DATASET_DIR / "d3s_video_dataset.mp4"
WINDOW_TITLE = "Detection Test - Video"
EXIT_KEYS = {ord("q"), ord("Q"), 27}


def draw_hud(frame: np.ndarray, frame_id: int, fps: float, face_state: str) -> None:
    """Vẽ thông số HUD lên màn hình."""
    telemetry = f"Frame: {frame_id} | Video FPS: {fps:.1f} | Face: {face_state}"
    color = (0, 255, 0) if face_state == "DETECTED" else (0, 0, 255)

    cv2.putText(frame, telemetry, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_video_detection(video_path: Path) -> None:
    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    reader = None

    try:
        reader = VideoReader(video_path)
        delay_ms = max(1, round(1000.0 / reader.fps))
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        print(f"[INFO] Playing video: {reader.video_path.name}")

        for frame_id, timestamp, frame in reader:
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

            draw_hud(frame, frame_id, reader.fps, state)
            cv2.imshow(WINDOW_TITLE, frame)

            if (cv2.waitKey(delay_ms) & 0xFF) in EXIT_KEYS or cv2.getWindowProperty(
                WINDOW_TITLE, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        if reader:
            reader.release()
        face_detector.release()
        landmark_detector.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanly released resources.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test Face & Landmark Detection on Video."
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_VIDEO, help="Path to video file."
    )
    args = parser.parse_args(argv)

    try:
        run_video_detection(args.source)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
