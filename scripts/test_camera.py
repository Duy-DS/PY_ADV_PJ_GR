"""Interactively verify T0301 live-camera acquisition and display."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.acquisition.camera import CameraStream  # noqa: E402

WINDOW_TITLE = "Webcam Stream (Task T0301)"
EXIT_KEYS = {ord("q"), ord("Q"), 27}
EXIT_PROMPT = "Press 'q' / 'ESC' on window or Ctrl+C in terminal to exit"


def draw_camera_hud(
    frame: np.ndarray,
    frame_id: int,
    elapsed: float,
    live_fps: float,
) -> None:
    """Draw live camera telemetry and exit controls on a frame.

    Parameters:
        frame: Mutable BGR frame displayed by OpenCV.
        frame_id: Zero-based camera frame index.
        elapsed: Monotonic elapsed stream time in seconds.
        live_fps: Average observed frame rate for this stream.

    Returns:
        None.

    Raises:
        cv2.error: If OpenCV cannot render text on the supplied frame.
    """

    telemetry = (
        f"Frame ID: {frame_id} | Elapsed: {elapsed:.2f} s | "
        f"Live FPS: {live_fps:.2f}"
    )
    cv2.putText(
        frame,
        telemetry,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        EXIT_PROMPT,
        (20, max(30, frame.shape[0] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def window_was_closed(window_title: str) -> bool:
    """Return whether a user closed an OpenCV display window.

    Parameters:
        window_title: Name passed to ``cv2.namedWindow``.

    Returns:
        True when the window is no longer visible, otherwise False.

    Raises:
        None.
    """

    try:
        return cv2.getWindowProperty(
            window_title,
            cv2.WND_PROP_VISIBLE,
        ) < 1
    except cv2.error:
        return True


def run_camera_test(device_id: int) -> None:
    """Open and display a live camera stream with safe termination.

    Parameters:
        device_id: OpenCV camera device index.

    Returns:
        None.

    Raises:
        ValueError: If OpenCV cannot open the requested camera device.
    """

    camera: CameraStream | None = None
    try:
        camera = CameraStream(device_id=device_id)
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        print(f"[INFO] Camera stream opened on device index: {device_id}")
        print(f"[INFO] Controls: {EXIT_PROMPT}.")

        frame_count = 0
        for frame_id, timestamp, frame in camera:
            frame_count += 1
            live_fps = frame_count / timestamp if timestamp > 0.0 else 0.0
            draw_camera_hud(
                frame,
                frame_id,
                timestamp,
                live_fps,
            )
            cv2.imshow(WINDOW_TITLE, frame)

            key_code = cv2.waitKey(1) & 0xFF
            if key_code in EXIT_KEYS:
                print("[INFO] Exit key detected from OpenCV window.")
                break
            if window_was_closed(WINDOW_TITLE):
                print("[INFO] OpenCV window was closed.")
                break
        else:
            print("[INFO] Camera stream ended or disconnected.")
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received from terminal.")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("[SUCCESS] Camera resources released cleanly (FReq032).")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for interactive camera capture.

    Parameters:
        None.

    Returns:
        Configured argument parser.

    Raises:
        None.
    """

    parser = argparse.ArgumentParser(
        description="Interactively test the T0301 live camera stream."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the camera verification command-line interface.

    Parameters:
        argv: Optional argument sequence used instead of ``sys.argv``.

    Returns:
        Process exit code: zero for success, one when the camera cannot open.

    Raises:
        None.
    """

    args = build_parser().parse_args(argv)
    try:
        run_camera_test(args.device)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        print(
            "[HINT] Check camera permissions or try another --device index.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
