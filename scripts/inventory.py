"""
Công cụ tổng hợp D3S Dataset: Tạo metadata cho Video và trích xuất Nhãn từ Ảnh (.jpg).
Hợp nhất từ `generate_labels.py` và `inventory_d3s.py`.
"""

import argparse
import csv
import re
import sys
import pandas as pd
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
UNKNOWN_VALUE = "UNKNOWN"

LABEL_ALIASES = {
    "eyeclose": "eyeclose",
    "eyeclosed": "eyeclose",
    "eye_close": "eyeclose",
    "eyesclosed": "eyeclose",
    "eyes_closed": "eyeclose",
    "closedeyes": "eyeclose",
    "closed_eyes": "eyeclose",
    "yawn": "yawning",
    "yawning": "yawning",
    "neutral": "neutral",
    "normal": "neutral",
    "happy": "happy",
    "smile": "happy",
    "smiling": "happy",
}


@dataclass(frozen=True)
class D3SVideoRecord:
    video_id: str
    relative_path: str
    file_name: str
    subject_id: str
    label: str
    label_source: str
    resolution: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    file_size_bytes: int


def _parse_subject_id(relative_path: str) -> str:
    normalized = relative_path.lower().replace("\\", "/")
    patterns = (
        r"(?:^|[/_\-\s])sub(?:ject)?[_\-\s]?0?([1-3])(?:$|[/_\-\s.])",
        r"(?:^|[/_\-\s])driver[_\-\s]?0?([1-3])(?:$|[/_\-\s.])",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return f"sub{match.group(1)}"
    return UNKNOWN_VALUE


def _parse_label(relative_path: str) -> str:
    normalized = relative_path.lower().replace("\\", "/")
    tokens = [t for t in re.split(r"[^a-z0-9]+", normalized) if t and t != "d3s"]
    compact = "_".join(tokens)
    for alias, label in LABEL_ALIASES.items():
        if alias in tokens or (len(alias.split("_")) > 1 and alias in compact):
            return label
    return UNKNOWN_VALUE


def process_videos(input_dir: Path, output_csv: Path):
    video_files = [
        p
        for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and "_downloads" not in p.parts
    ]

    if not video_files:
        print(f" Không tìm thấy file video nào trong {input_dir}.")
        return

    records = []
    for index, video_path in enumerate(sorted(video_files), 1):
        try:
            capture = cv2.VideoCapture(str(video_path))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()

            if fps <= 0 or total_frames <= 0:
                continue

            rel_path = video_path.resolve().relative_to(input_dir).as_posix()
            sub_id, lbl = _parse_subject_id(rel_path), _parse_label(rel_path)

            records.append(
                D3SVideoRecord(
                    video_id=f"d3s_{index:04d}",
                    relative_path=rel_path,
                    file_name=video_path.name,
                    subject_id=sub_id,
                    label=lbl,
                    label_source="path"
                    if UNKNOWN_VALUE not in (sub_id, lbl)
                    else "unparsed",
                    resolution=f"{width}x{height}",
                    width=width,
                    height=height,
                    fps=round(fps, 6),
                    total_frames=total_frames,
                    duration_sec=round(total_frames / fps, 6),
                    file_size_bytes=video_path.stat().st_size,
                )
            )
        except Exception as e:
            print(f" Lỗi khi xử lý video {video_path.name}: {e}")

    if records:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(r) for r in records]).to_csv(output_csv, index=False)
        print(f"Đã tạo Video Inventory: '{output_csv}' ({len(records)} videos)")


def process_images(input_dir: Path, output_csv: Path):
    data = []
    drowsy_labels = ["Eyeclose", "Yawn"]

    # Tìm các thư mục bắt đầu bằng "Sub"
    subject_folders = [
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("Sub")
    ]

    if not subject_folders:
        print(
            f" Không tìm thấy cấu trúc thư mục dạng ảnh (Sub1, Sub2...) trong {input_dir}."
        )
        return

    for subject_folder in subject_folders:
        subject_id = subject_folder.name
        for label_folder in [d for d in subject_folder.iterdir() if d.is_dir()]:
            label = label_folder.name
            is_drowsy = 1 if label in drowsy_labels else 0

            for img_file in label_folder.glob("*.jpg"):
                data.append(
                    {
                        "image_path": str(img_file),
                        "subject_id": subject_id,
                        "label_name": label,
                        "is_drowsy": is_drowsy,
                    }
                )

    df = pd.DataFrame(data)
    if not df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f" Đã tạo Image Labels: '{output_csv}' ({len(df)} ảnh)")


def main():
    parser = argparse.ArgumentParser(
        description="Tạo Inventory Video & Labels từ Ảnh JPG"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Thư mục chứa dữ liệu thô (Video/Ảnh).",
    )
    parser.add_argument(
        "--out-inventory",
        type=Path,
        default=REPO_ROOT / "data/metadata/d3s_inventory.csv",
        help="Đường dẫn lưu file inventory.",
    )
    parser.add_argument(
        "--out-labels",
        type=Path,
        default=REPO_ROOT / "data/raw/labels.csv",
        help="Đường dẫn lưu file labels.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        print(f" Không tìm thấy thư mục: {input_dir}")
        sys.exit(1)

    # Thực thi cả 2 quy trình song song
    process_videos(input_dir, args.out_inventory)
    process_images(input_dir, args.out_labels)


if __name__ == "__main__":
    main()
