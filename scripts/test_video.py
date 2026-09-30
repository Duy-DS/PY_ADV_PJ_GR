"""Interactively verify T0301 video-file acquisition and display."""

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

from src.acquisition.video_reader import VideoReader  # noqa: E402

DATASET_DIR = REPO_ROOT / "data" / "raw" / "d3s"
DEFAULT_VIDEO = DATASET_DIR / "d3s_video_dataset.mp4"
WINDOW_TITLE = "D3S Video Player (Task T0301)"
EXIT_KEYS = {ord("q"), ord("Q"), 27}
EXIT_PROMPT = "Press 'q' / 'ESC' on window or Ctrl+C in terminal to exit"


def resolve_video_source(requested_path: Path) -> Path:
    """Resolve a requested video and fall back to another D3S video.

    Parameters:
        requested_path: CLI-provided video path. Relative paths are resolved
            from the current working directory.

    Returns:
        The resolved path to an existing MP4 or AVI video.

    Raises:
        FileNotFoundError: If neither the requested path nor a fallback D3S
            video exists.
    """

    candidate = requested_path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()

    if candidate.is_file():
        return candidate

    fallback_videos = sorted(
        path.resolve()
        for pattern in ("*.mp4", "*.avi")
        for path in DATASET_DIR.rglob(pattern)
        if path.is_file()
    )
    if fallback_videos:
        fallback = fallback_videos[0]
        print(
            f"[WARNING] Video file not found: {candidate}\n"
            f"[INFO] Using fallback D3S video: {fallback}",
            file=sys.stderr,
        )
        return fallback

    raise FileNotFoundError(
        f"Video file not found: {candidate}. No fallback MP4 or AVI files "
        f"were found in {DATASET_DIR}. Run Task T0201 or provide --source."
    )


def draw_video_hud(
    frame: np.ndarray,
    frame_id: int,
    timestamp: float,
    fps: float,
    total_frames: int,
) -> None:
    """Draw video playback telemetry and exit controls on a frame.

    Parameters:
        frame: Mutable BGR frame displayed by OpenCV.
        frame_id: Zero-based frame index.
        timestamp: Relative video timestamp in seconds.
        fps: Source frame rate.
        total_frames: Total source frame count reported by OpenCV.

    Returns:
        None.

    Raises:
        cv2.error: If OpenCV cannot render text on the supplied frame.
    """

    telemetry = (
        f"Frame ID: {frame_id} | Timestamp: {timestamp:.2f} s | "
        f"FPS: {fps:.2f} | Total Frames: {total_frames}"
    )
    cv2.putText(
        frame,
        telemetry,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        EXIT_PROMPT,
        (20, max(30, frame.shape[0] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
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


def run_video_test(video_path: Path) -> None:
    """Open and stream video frames with responsive termination.

    Parameters:
        video_path: Path to the target video file.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the specified video does not exist.
        ValueError: If OpenCV cannot open the video or its FPS is invalid.
    """

    reader: VideoReader | None = None
    try:
        reader = VideoReader(video_path)
        delay_ms = max(1, round(1000.0 / reader.fps))
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

        print(
            f"[INFO] Streaming: {reader.video_path.name} "
            f"({reader.fps:.2f} FPS, {reader.total_frames} frames)"
        )
        print(f"[INFO] Controls: {EXIT_PROMPT}.")

        for frame_id, timestamp, frame in reader:
            draw_video_hud(
                frame,
                frame_id,
                timestamp,
                reader.fps,
                reader.total_frames,
            )
            cv2.imshow(WINDOW_TITLE, frame)

            key_code = cv2.waitKey(delay_ms) & 0xFF
            if key_code in EXIT_KEYS:
                print("[INFO] Exit key detected from OpenCV window.")
                break
            if window_was_closed(WINDOW_TITLE):
                print("[INFO] OpenCV window was closed.")
                break
        else:
            print("[INFO] End of video reached.")
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received from terminal.")
    finally:
        if reader is not None:
            reader.release()
        cv2.destroyAllWindows()
        print("[SUCCESS] Video resources released cleanly (FReq032).")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for interactive video playback.

    Parameters:
        None.

    Returns:
        Configured argument parser.

    Raises:
        None.
    """

    parser = argparse.ArgumentParser(
        description="Interactively test the T0301 D3S video reader."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_VIDEO,
        help=(
            "Video path (default: data/raw/d3s/"
            "d3s_video_dataset.mp4)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the video verification command-line interface.

    Parameters:
        argv: Optional argument sequence used instead of ``sys.argv``.

    Returns:
        Process exit code: zero for success, one for an input/opening error.

    Raises:
        None.
    """

    args = build_parser().parse_args(argv)
    try:
        video_path = resolve_video_source(args.source)
        run_video_test(video_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
