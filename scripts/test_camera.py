"""Xác minh tương tác việc thu thập và hiển thị luồng camera trực tiếp cho T0301."""

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

WINDOW_TITLE = "Luồng Webcam (Task T0301)"
EXIT_KEYS = {ord("q"), ord("Q"), 27}
EXIT_PROMPT = "Nhấn 'q' / 'ESC' trên cửa sổ hoặc Ctrl+C ở terminal để thoát"


def draw_camera_hud(
    frame: np.ndarray,
    frame_id: int,
    elapsed: float,
    live_fps: float,
) -> None:
    """Vẽ thông số từ xa (telemetry) của camera trực tiếp và điều khiển thoát lên khung hình.

    Tham số:
        frame: Khung hình BGR có thể thay đổi được hiển thị bởi OpenCV.
        frame_id: Chỉ số khung hình camera bắt đầu từ 0.
        elapsed: Thời gian luồng đã trôi qua tính bằng giây.
        live_fps: Tốc độ khung hình trung bình quan sát được cho luồng này.

    Trả về:
        Không có.

    Ngoại lệ:
        cv2.error: Nếu OpenCV không thể vẽ chữ lên khung hình cung cấp.
    """

    telemetry = (
        f"Frame ID: {frame_id} | Thoi gian: {elapsed:.2f} s | "
        f"FPS Truc tiep: {live_fps:.2f}"
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
    """Trả về True nếu người dùng đã đóng cửa sổ hiển thị OpenCV.

    Tham số:
        window_title: Tên được truyền vào ``cv2.namedWindow``.

    Trả về:
        True khi cửa sổ không còn hiển thị nữa, ngược lại là False.
    """

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


def run_camera_test(device_id: int) -> None:
    """Mở và hiển thị luồng camera trực tiếp với khả năng chấm dứt an toàn.

    Tham số:
        device_id: Chỉ số thiết bị camera OpenCV.

    Ngoại lệ:
        ValueError: Nếu OpenCV không thể mở thiết bị camera được yêu cầu.
    """

    camera: CameraStream | None = None
    try:
        camera = CameraStream(device_id=device_id)
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        print(f" Luồng camera đã được mở trên chỉ số thiết bị: {device_id}")
        print(f" Điều khiển: {EXIT_PROMPT}.")

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
                print(" Đã phát hiện phím thoát từ cửa sổ OpenCV.")
                break
            if window_was_closed(WINDOW_TITLE):
                print(" Cửa sổ OpenCV đã bị đóng.")
                break
        else:
            print(" Luồng camera kết thúc hoặc bị ngắt kết nối.")
    except KeyboardInterrupt:
        print("\n Đã nhận được lệnh Ctrl+C từ terminal.")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print(" Đã giải phóng sạch sẽ tài nguyên camera (FReq032).")


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng bộ phân tích tham số dòng lệnh để chụp ảnh camera tương tác."""

    parser = argparse.ArgumentParser(
        description="Kiểm thử tương tác luồng camera trực tiếp (Task T0301)."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Chỉ số thiết bị camera (mặc định: 0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy giao diện dòng lệnh xác minh camera."""

    args = build_parser().parse_args(argv)
    try:
        run_camera_test(args.device)
    except ValueError as error:
        print(f" {error}", file=sys.stderr)
        print(
            " Vui lòng kiểm tra quyền truy cập camera hoặc thử chỉ số --device khác.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
