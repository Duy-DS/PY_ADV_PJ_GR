"""Create an OpenCV-backed inventory for local D3S video files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw" / "d3s"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "metadata" / "d3s_inventory.csv"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
UNKNOWN_VALUE = "UNKNOWN"

CSV_COLUMNS = (
    "video_id",
    "relative_path",
    "file_name",
    "subject_id",
    "label",
    "label_source",
    "resolution",
    "width",
    "height",
    "fps",
    "total_frames",
    "duration_sec",
    "file_size_bytes",
)

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
    """Metadata for one video file discovered under the D3S root."""

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for inventory generation."""

    parser = argparse.ArgumentParser(
        description="Scan data/raw/d3s videos and write d3s_inventory.csv."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="D3S video root to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path for the generated inventory.",
    )
    return parser.parse_args(argv)


def discover_video_files(input_dir: Path) -> list[Path]:
    """Return candidate videos, excluding download cache copies."""

    input_dir = input_dir.resolve()
    if not input_dir.exists():
        return []

    video_files: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "_downloads" in path.relative_to(input_dir).parts:
            continue
        video_files.append(path)

    return sorted(video_files)


def build_inventory(input_dir: Path) -> list[D3SVideoRecord]:
    """Collect metadata for all videos below ``input_dir``."""

    input_dir = input_dir.resolve()
    records: list[D3SVideoRecord] = []
    for index, video_path in enumerate(discover_video_files(input_dir), 1):
        records.append(_inspect_video(video_path, input_dir, index))
    return records


def write_inventory(
    records: Sequence[D3SVideoRecord],
    output_path: Path,
) -> None:
    """Write inventory records as a deterministic CSV file."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def generate_inventory(
    input_dir: Path,
    output_path: Path,
) -> list[D3SVideoRecord]:
    """Build and persist a D3S inventory CSV."""

    records = build_inventory(input_dir)
    write_inventory(records, output_path)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inventory command."""

    args = parse_args(argv)
    records = generate_inventory(args.input_dir, args.output)
    print(f"Wrote {len(records)} D3S video record(s) to {args.output}")
    return 0


def _inspect_video(
    video_path: Path,
    input_dir: Path,
    index: int,
) -> D3SVideoRecord:
    """Read one video's OpenCV metadata and parsed path labels."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if fps <= 0:
        raise RuntimeError(f"Video has invalid FPS {fps}: {video_path}")
    if total_frames <= 0:
        raise RuntimeError(
            f"Video has invalid frame count {total_frames}: {video_path}"
        )

    relative_path = video_path.resolve().relative_to(input_dir).as_posix()
    subject_id = _parse_subject_id(relative_path)
    label = _parse_label(relative_path)
    if UNKNOWN_VALUE in (subject_id, label):
        label_source = "unparsed"
    else:
        label_source = "path"

    return D3SVideoRecord(
        video_id=f"d3s_{index:04d}",
        relative_path=relative_path,
        file_name=video_path.name,
        subject_id=subject_id,
        label=label,
        label_source=label_source,
        resolution=f"{width}x{height}",
        width=width,
        height=height,
        fps=round(fps, 6),
        total_frames=total_frames,
        duration_sec=round(total_frames / fps, 6),
        file_size_bytes=video_path.stat().st_size,
    )


def _parse_subject_id(relative_path: str) -> str:
    """Parse D3S subject ids from path tokens such as sub1 or subject_01."""

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
    """Parse one of the project-approved D3S labels from path tokens."""

    normalized = relative_path.lower().replace("\\", "/")
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", normalized)
        if token and token != "d3s"
    ]
    compact = "_".join(tokens)
    for alias, label in LABEL_ALIASES.items():
        alias_tokens = alias.split("_")
        if alias in tokens:
            return label
        if len(alias_tokens) > 1 and alias in compact:
            return label
    return UNKNOWN_VALUE


if __name__ == "__main__":
    sys.exit(main())
