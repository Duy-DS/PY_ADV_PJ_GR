"""Xác minh tương tác việc đọc và hiển thị file video cho T0301."""

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
WINDOW_TITLE = "Trình phát Video D3S (Task T0301)"
EXIT_KEYS = {ord("q"), ord("Q"), 27}
EXIT_PROMPT = "Nhấn 'q' / 'ESC' trên cửa sổ hoặc Ctrl+C ở terminal để thoát"


def resolve_video_source(requested_path: Path) -> Path:
    """Xác định file video yêu cầu và dự phòng sang video D3S khác nếu cần.

    Tham số:
        requested_path: Đường dẫn video do CLI cung cấp. Đường dẫn tương đối được
            phân giải từ thư mục làm việc hiện tại.

    Trả về:
        Đường dẫn đã được phân giải tới file video MP4 hoặc AVI hiện có.

    Ngoại lệ:
        FileNotFoundError: Nếu cả đường dẫn yêu cầu lẫn video D3S dự phòng đều không tồn tại.
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
            f" Không tìm thấy file video: {candidate}\n"
            f" Sử dụng video D3S dự phòng: {fallback}",
            file=sys.stderr,
        )
        return fallback

    raise FileNotFoundError(
        f"Không tìm thấy file video: {candidate}. Không có file MP4 hay AVI dự phòng "
        f"nào được tìm thấy trong {DATASET_DIR}. Hãy chạy Task T0201 hoặc cung cấp --source."
    )


def draw_video_hud(
    frame: np.ndarray,
    frame_id: int,
    timestamp: float,
    fps: float,
    total_frames: int,
) -> None:
    """Vẽ thông số (telemetry) của trình phát video và điều khiển thoát lên khung hình.

    Tham số:
        frame: Khung hình BGR có thể thay đổi được hiển thị bởi OpenCV.
        frame_id: Chỉ số khung hình bắt đầu từ 0.
        timestamp: Thời gian video tương đối tính bằng giây.
        fps: Tốc độ khung hình (frame rate) nguồn.
        total_frames: Tổng số khung hình nguồn được báo cáo bởi OpenCV.
    """

    telemetry = (
        f"Frame ID: {frame_id} | Thoi gian: {timestamp:.2f} s | "
        f"FPS: {fps:.2f} | Tong khung hinh: {total_frames}"
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
    """Trả về True nếu người dùng đã đóng cửa sổ hiển thị OpenCV."""

    try:
        return (
            cv2.getWindowProperty(
                window_title,
                cv2.WND_PROP_VISIBLE,
            )
            < 1
        )
    except cv2.error:
        return True


def run_video_test(video_path: Path) -> None:
    """Mở và phát (stream) các khung hình video với khả năng chấm dứt phản hồi nhanh."""

    reader: VideoReader | None = None
    try:
        reader = VideoReader(video_path)
        delay_ms = max(1, round(1000.0 / reader.fps))
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

        print(
            f" Đang phát: {reader.video_path.name} "
            f"({reader.fps:.2f} FPS, {reader.total_frames} frames)"
        )
        print(f" Điều khiển: {EXIT_PROMPT}.")

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
                print(" Đã phát hiện phím thoát từ cửa sổ OpenCV.")
                break
            if window_was_closed(WINDOW_TITLE):
                print(" Cửa sổ OpenCV đã bị đóng.")
                break
        else:
            print(" Đã phát đến cuối video.")
    except KeyboardInterrupt:
        print("\n Đã nhận được lệnh Ctrl+C từ terminal.")
    finally:
        if reader is not None:
            reader.release()
        cv2.destroyAllWindows()
        print(" Đã giải phóng sạch sẽ tài nguyên video (FReq032).")


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng bộ phân tích tham số dòng lệnh cho trình phát video tương tác."""

    parser = argparse.ArgumentParser(
        description="Kiểm thử tương tác trình đọc video D3S (Task T0301)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_VIDEO,
        help=("Đường dẫn video (mặc định: data/raw/d3s/d3s_video_dataset.mp4)."),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy giao diện dòng lệnh xác minh video."""

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
