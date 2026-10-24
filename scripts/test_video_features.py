"""Script ki?m tra tr?ch xu?t ??c tr?ng m?t (EAR) v? mi?ng (MAR) c?p frame.

Fulfills Task: T1001 PoC Verification.
"""

import argparse
from pathlib import Path
import sys
import cv2

# 1. ???ng d?n an to?n relative t? project root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.video_reader import VideoReader
from src.detection.landmark_detector import LandmarkDetector
from src.features.eye import EyeFeatureExtractor
from src.features.mouth import MouthFeatureExtractor
from src.utils.config import load_config

CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_VIDEO = REPO_ROOT / "data" / "raw" / "d3s" / "d3s_video_dataset.mp4"


def main():
    parser = argparse.ArgumentParser(
        description="Ki?m tra tr?ch xu?t ??c tr?ng m?t (EAR) v? mi?ng (MAR)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_VIDEO),
        help="???ng d?n video ho?c webcam ID.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="S? khung h?nh t?i ?a x? l?."
    )
    parser.add_argument(
        "--no-visual",
        dest="visual",
        action="store_false",
        default=True,
        help="T?t c?a s? ?? h?a.",
    )
    args = parser.parse_args()

    video_path = Path(args.source)
    if not video_path.exists():
        print(f" Kh?ng t?m th?y video t?i: {video_path}")
        return

    cfg = load_config(str(CONFIG_PATH))
    reader = VideoReader(str(video_path))
    detector = LandmarkDetector()
    eye_ext = EyeFeatureExtractor(ear_threshold=cfg["eye"]["ear_threshold"])
    mouth_ext = MouthFeatureExtractor(mar_threshold=cfg["mouth"]["mar_threshold"])

    print(">> ?ang ch?y ki?m th? Pipeline...")
    if args.visual:
        print(
            ">> Nh?n Q ho?c ESC tr?n c?a s? hi?n th? ?? d?ng (ho?c Ctrl+C t?i Terminal)."
        )
    else:
        print(">> Ch? ?? kh?ng ?? h?a (--no-visual)... Nh?n Ctrl+C ?? d?ng.")

    frame_count = 0
    try:
        for frame_id, ts, frame in reader:
            if args.max_frames and frame_count >= args.max_frames:
                print(f" ?? ??t gi?i h?n {args.max_frames} frames.")
                break

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

                hud_text = f"Frame {frame_id:04d} | EAR: {ear_val:.2f} ({eye_st}) - MAR: {mar_val:.2f} ({mouth_st})"
                if not args.visual and frame_count % 30 == 0:
                    print(f" {hud_text}")

                if args.visual:
                    cv2.putText(
                        frame,
                        hud_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            else:
                if not args.visual and frame_count % 30 == 0:
                    print(f" Frame {frame_id:04d} | FACE NOT DETECTED (UNKNOWN)")
                if args.visual:
                    cv2.putText(
                        frame,
                        "FACE NOT DETECTED (UNKNOWN)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            frame_count += 1

            if args.visual:
                cv2.imshow("Test Features Frame-Level (T1001 PoC)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print(" Nh?n t?n hi?u d?ng t? c?a s? giao di?n.")
                    break

    except KeyboardInterrupt:
        print(" ?? d?ng t? Terminal.")
    finally:
        reader.release()
        detector.release()
        if args.visual:
            cv2.destroyAllWindows()
        print(f" ?? x? l? {frame_count} frames v? gi?i ph?ng t?i nguy?n an to?n.")


if __name__ == "__main__":
    main()
