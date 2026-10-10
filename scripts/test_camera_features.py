"""Script ki?m tra tr?ch xu?t ??c tr?ng m?t (EAR) v? mi?ng (MAR) t? webcam th?i gian th?c.

Fulfills Task: T1001 PoC Webcam Verification & FReq002/FReq032.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

# C?u h?nh UTF-8 cho stdout/stderr ?? tr?nh l?i m? h?a k? t? tr?n Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

# Thi?t l?p ???ng d?n g?c d? ?n ?? ??m b?o import ??ng
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.camera import CameraStream
from src.detection.landmark_detector import LandmarkDetector
from src.features.eye import EyeFeatureExtractor
from src.features.mouth import MouthFeatureExtractor
from src.utils.config import load_config

CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
WINDOW_TITLE = "Webcam Facial Features PoC (Task T1001)"
EXIT_KEYS = {ord("q"), ord("Q"), 27}


def window_was_closed(window_title: str) -> bool:
    """Ki?m tra xem ng??i d?ng ?? b?m n?t X ?? ??ng c?a s? OpenCV ch?a."""
    try:
        return cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main(device_id: int = 0, max_frames: Optional[int] = None) -> None:
    cfg = load_config(str(CONFIG_PATH))

    try:
        cam = CameraStream(device_id=device_id)
    except Exception as e:
        print(f"[ERROR] Kh?ng th? kh?i ??ng Camera device {device_id}: {e}")
        return

    detector = LandmarkDetector()
    eye_ext = EyeFeatureExtractor(ear_threshold=cfg["eye"]["ear_threshold"])
    mouth_ext = MouthFeatureExtractor(mar_threshold=cfg["mouth"]["mar_threshold"])

    print(f">> [INFO] ?ang b?t Webcam thi?t b? {device_id}... H?y nh?n v?o camera.")
    print(">> [INFO] Nh?n ph?m 'q', 'Q', 'ESC' ho?c b?m [X] tr?n c?a s? ?? tho?t (ho?c Ctrl+C t?i Terminal).")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    prev_time = time.perf_counter()
    frame_count = 0

    try:
        for frame_id, ts, frame in cam:
            frame_count += 1
            curr_time = time.perf_counter()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # 1. Tr?ch xu?t landmark t? LandmarkDetector
            lm = detector.extract_landmarks(frame)

            if lm.get("face_detected", False):
                eye_res = eye_ext.compute(
                    lm["left_eye_landmarks"], lm["right_eye_landmarks"]
                )
                mouth_res = mouth_ext.compute(lm["mouth_landmarks"])

                ear_val = eye_res.get("ear", 0.0)
                eye_st = eye_res.get("eye_state", "UNKNOWN")
                mar_val = mouth_res.get("mar", 0.0)
                mouth_st = mouth_res.get("mouth_state", "UNKNOWN")

                hud_info = f"EAR: {ear_val:.2f} [{eye_st}] - MAR: {mar_val:.2f} [{mouth_st}]"
                color = (0, 255, 0) if eye_st == "OPEN" else (0, 165, 255)
                cv2.putText(
                    frame,
                    hud_info,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            else:
                # FReq004/FReq030: M?t khu?n m?t -> Hi?n th? c?nh b?o UNKNOWN
                cv2.putText(
                    frame,
                    "FACE NOT DETECTED (UNKNOWN)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # V? th?ng s? FPS v? Frame ID
            cv2.putText(
                frame,
                f"Frame: {frame_id} | FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow(WINDOW_TITLE, frame)

            # B?t s? ki?n tho?t: q, Q, ESC (27) ho?c b?m ??ng c?a s?
            key = cv2.waitKey(1) & 0xFF
            if key in EXIT_KEYS or window_was_closed(WINDOW_TITLE):
                print("[INFO] Nh?n t?n hi?u d?ng t? ng??i d?ng.")
                break

            if max_frames and frame_count >= max_frames:
                print(f"[INFO] ?? ??t gi?i h?n {max_frames} frames.")
                break

    except KeyboardInterrupt:
        print("[INFO] ?? nh?n t?n hi?u d?ng (Ctrl+C) t? Terminal.")
    finally:
        # FReq032: ??m b?o gi?i ph?ng to?n b? t?i nguy?n an to?n
        cam.release()
        detector.release()
        cv2.destroyAllWindows()
        print(f"[SUCCESS] ?? x? l? {frame_count} frames v? gi?i ph?ng an to?n Camera, MediaPipe v? c?a s? OpenCV.")


if __name__ == "__main__":
    main()


