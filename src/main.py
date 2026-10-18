"""CLI T1002 cho video tuần tự và webcam tùy chọn."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def positive_integer(value: str) -> int:
    """Kiểm tra giới hạn frame.

    Parameters: value từ dòng lệnh.
    Returns: Số nguyên dương.
    Raises: argparse.ArgumentTypeError nếu không hợp lệ.
    """
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser không khởi tạo CV/model.

    Parameters: Không có.
    Returns: ArgumentParser với CLI chuẩn.
    Raises: Không có.
    """
    parser = argparse.ArgumentParser(
        description="Driver drowsiness detection: sequential video or webcam."
    )
    parser.add_argument(
        "--source", required=True, help="webcam, index or video"
    )
    parser.add_argument("--mode", choices=["eye_only", "eye_yawn"])
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/default.yaml"
    )
    visual = parser.add_mutually_exclusive_group()
    visual.add_argument("--visual", dest="display", action="store_true")
    visual.add_argument(
        "--no-visual", "--headless", dest="display", action="store_false"
    )
    parser.set_defaults(display=True)
    parser.add_argument("--max-frames", type=positive_integer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI và in tổng kết JSON, lỗi có hướng dẫn trên stderr.

    Parameters: argv tùy chọn cho kiểm thử.
    Returns: 0 thành công, 1 lỗi chạy; argparse dùng mã 2 cho tham số sai.
    Raises: SystemExit khi --help hoặc tham số không hợp lệ.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        from src.pipeline.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline(args.config, args.mode)
        summary = pipeline.run(
            args.source, display=args.display, max_frames=args.max_frames
        )
    except Exception as exc:
        logging.error(
            "Pipeline failed: %s. Check --source, --config and model assets.",
            exc,
        )
        return 1
    print(json.dumps(asdict(summary), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
