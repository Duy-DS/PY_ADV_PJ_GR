"""Tạo notebook khám phá tập dữ liệu D3S đã thực thi từ file CSV inventory."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY_PATH = REPO_ROOT / "data" / "metadata" / "d3s_inventory.csv"
NOTEBOOK_FILENAME = "01_dataset_exploration.ipynb"
DEFAULT_NOTEBOOK_PATH = REPO_ROOT / "notebooks" / NOTEBOOK_FILENAME


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Tạo bộ phân tích tham số dòng lệnh."""

    parser = argparse.ArgumentParser(
        description="Xây dựng notebook khám phá tập dữ liệu D3S đã thực thi."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help="Đường dẫn đầu vào của file CSV inventory D3S.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NOTEBOOK_PATH,
        help="Đường dẫn đầu ra cho notebook.",
    )
    return parser.parse_args(argv)


def read_inventory(inventory_path: Path) -> list[dict[str, str]]:
    """Đọc file CSV inventory thành danh sách các từ điển thông thường."""

    with inventory_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_notebook(
    records: Sequence[dict[str, str]],
    inventory_path: Path,
    output_path: Path,
) -> None:
    """Ghi một notebook hợp lệ đã được thực thi kèm theo kết quả biểu đồ."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = _build_notebook(records, inventory_path)
    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy quy trình tạo notebook."""

    args = parse_args(argv)
    records = read_inventory(args.inventory)
    write_notebook(records, args.inventory, args.output)
    print(f"Đã ghi notebook tập dữ liệu đã thực thi tại {args.output}")
    return 0


def _build_notebook(
    records: Sequence[dict[str, str]],
    inventory_path: Path,
) -> dict[str, Any]:
    """Xây dựng cấu trúc JSON của notebook với các kết quả (outputs) được tính toán trước."""

    cells: list[dict[str, Any]] = [
        _markdown_cell(
            "# Khám phá tập dữ liệu D3S\n\n"
            "Task T0202 đánh giá inventory cho Hệ thống Phát hiện Buồn ngủ "
            "của Tài xế. Notebook này đọc file `data/metadata/d3s_inventory.csv`, "
            "tóm tắt siêu dữ liệu (metadata) ở cấp độ video, và ghi chép lại hướng dẫn "
            "chia tách dữ liệu sao cho giữ nguyên được ranh giới của đối tượng/video."
        ),
        _code_cell(
            "\n".join(
                [
                    "from collections import Counter",
                    "import csv",
                    "from pathlib import Path",
                    "import matplotlib.pyplot as plt",
                    "",
                    "repo_root = Path.cwd().resolve()",
                    "inventory_path = repo_root / "
                    "'data' / 'metadata' / 'd3s_inventory.csv'",
                    "with inventory_path.open(newline='', encoding='utf-8') "
                    "as csv_file:",
                    "    records = list(csv.DictReader(csv_file))",
                    "",
                    "print(f'Đã tải {len(records)} bản ghi video từ {inventory_path}')",
                ]
            ),
            execution_count=1,
            text_output=(
                "Đã tải "
                f"{len(records)} bản ghi video từ "
                f"{_display_path(inventory_path)}\n"
            ),
        ),
        _code_cell(
            "\n".join(
                [
                    "for record in records:",
                    "    print(",
                    "        record['video_id'],",
                    "        record['relative_path'],",
                    "        record['subject_id'],",
                    "        record['label'],",
                    "        record['resolution'],",
                    "        record['fps'],",
                    "        record['total_frames'],",
                    "        record['duration_sec'],",
                    "    )",
                ]
            ),
            execution_count=2,
            text_output=_inventory_table_text(records),
        ),
        _markdown_cell(
            "## Nguồn gốc của Siêu dữ liệu (Metadata Provenance)\n\n"
            "Các nhãn về đối tượng và phân lớp chỉ được phân tích cú pháp từ các thành phần "
            "trong đường dẫn hoặc tên file, ví dụ như `subject_02/yawning/sample.mp4`. Cấu trúc "
            "D3S nội bộ hiện tại là một file MP4 nguyên khối duy nhất có tên là "
            "`d3s_video_dataset.mp4`, vì vậy các trường đối tượng và nhãn của nó "
            "sẽ duy trì trạng thái `UNKNOWN` (Không xác định) cho đến khi nhóm dự án nhập vào một "
            "cấu trúc phát hành khung hình/video bài bản hoặc bổ sung thêm các chú thích phân đoạn đã được kiểm chứng."
        ),
        _code_cell(
            "\n".join(
                [
                    "label_counts = Counter(",
                    "    record['label'] for record in records",
                    ")",
                    "fig, ax = plt.subplots(figsize=(6, 4))",
                    "ax.bar(label_counts.keys(), label_counts.values(), "
                    "color='#2f80ed')",
                    "ax.set_title('Tần suất lớp D3S theo bản ghi video')",
                    "ax.set_xlabel('Nhãn')",
                    "ax.set_ylabel('Số lượng video')",
                    "ax.grid(axis='y', alpha=0.25)",
                    "plt.tight_layout()",
                ]
            ),
            execution_count=3,
            image_png=_bar_chart_png(
                Counter(record["label"] for record in records),
                "Tần suất lớp D3S theo bản ghi video",
                "Nhãn",
                "Số lượng video",
                "#2f80ed",
            ),
        ),
        _code_cell(
            "\n".join(
                [
                    "durations = [float(record['duration_sec']) for record in records]",
                    "fig, ax = plt.subplots(figsize=(6, 4))",
                    "ax.hist(durations, bins=max(1, min(8, len(durations))), "
                    "color='#27ae60', edgecolor='white')",
                    "ax.set_title('Phân phối thời lượng video D3S')",
                    "ax.set_xlabel('Thời lượng (giây)')",
                    "ax.set_ylabel('Số lượng video')",
                    "ax.grid(axis='y', alpha=0.25)",
                    "plt.tight_layout()",
                ]
            ),
            execution_count=4,
            image_png=_histogram_png(
                [float(record["duration_sec"]) for record in records],
                "Phân phối thời lượng video D3S",
                "Thời lượng (giây)",
                "Số lượng video",
            ),
        ),
        _code_cell(
            "\n".join(
                [
                    "subject_counts = Counter(",
                    "    record['subject_id'] for record in records",
                    ")",
                    "fig, ax = plt.subplots(figsize=(6, 4))",
                    "ax.bar(subject_counts.keys(), subject_counts.values(), "
                    "color='#f2994a')",
                    "ax.set_title('Sự cân bằng đối tượng D3S theo bản ghi video')",
                    "ax.set_xlabel('Đối tượng (Subject)')",
                    "ax.set_ylabel('Số lượng video')",
                    "ax.grid(axis='y', alpha=0.25)",
                    "plt.tight_layout()",
                ]
            ),
            execution_count=5,
            image_png=_bar_chart_png(
                Counter(record["subject_id"] for record in records),
                "Sự cân bằng đối tượng D3S theo bản ghi video",
                "Đối tượng (Subject)",
                "Số lượng video",
                "#f2994a",
            ),
        ),
        _markdown_cell(
            "## Hướng dẫn Chia tách Cấp độ Đối tượng\n\n"
            "- Chia tách theo đối tượng khi có sẵn ID đối tượng: ví dụ, "
            "huấn luyện trên hai đối tượng và giữ lại đối tượng thứ ba để "
            "kiểm tra (validation) hoặc thử nghiệm (testing).\n"
            "- Nếu một đối tượng không thể tách rời, hãy chia tách theo toàn bộ "
            "video nguồn hoặc phiên làm việc liền kề đã được xác minh, tuyệt đối không "
            "chia tách bằng cách xáo trộn ngẫu nhiên các khung hình.\n"
            "- Giữ các khung hình lân cận từ cùng một tài xế/phiên làm việc nằm trong "
            "cùng một phân vùng để các tín hiệu chuỗi thời gian không bị rò rỉ sang "
            "tập dữ liệu đánh giá.\n"
            "- Báo cáo các bản ghi `UNKNOWN` và không hợp lệ một cách riêng biệt trong quá trình "
            "đánh giá thay vì âm thầm loại bỏ chúng."
        ),
        _code_cell(
            "\n".join(
                [
                    "unparsed = [",
                    "    record for record in records",
                    "    if record['subject_id'] == 'UNKNOWN'",
                    "    or record['label'] == 'UNKNOWN'",
                    "]",
                    "print(f'Các hàng cần nhãn xác minh: {len(unparsed)}')",
                    "for record in unparsed:",
                    "    print(record['relative_path'])",
                ]
            ),
            execution_count=6,
            text_output=_unparsed_text(records),
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": _python_version_text(),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _markdown_cell(source: str) -> dict[str, Any]:
    """Tạo một ô markdown cho notebook."""

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _display_path(path: Path) -> str:
    """Trả về đường dẫn tương đối ổn định của kho lưu trữ khi có thể."""

    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _python_version_text() -> str:
    """Trả về phiên bản Python dạng ngắn cho siêu dữ liệu notebook."""

    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _code_cell(
    source: str,
    execution_count: int,
    text_output: str | None = None,
    image_png: str | None = None,
) -> dict[str, Any]:
    """Tạo một ô mã đã thực thi kèm theo kết quả dạng văn bản hoặc PNG."""

    outputs: list[dict[str, Any]] = []
    if text_output is not None:
        outputs.append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": text_output.splitlines(keepends=True),
            }
        )
    if image_png is not None:
        outputs.append(
            {
                "data": {"image/png": image_png, "text/plain": ["<Figure>"]},
                "metadata": {},
                "output_type": "display_data",
            }
        )

    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    }


def _inventory_table_text(records: Sequence[dict[str, str]]) -> str:
    """Hiển thị các hàng inventory dạng gọn gàng để xuất ra stdout của notebook."""

    lines: list[str] = []
    for record in records:
        lines.append(
            " ".join(
                [
                    record["video_id"],
                    record["relative_path"],
                    record["subject_id"],
                    record["label"],
                    record["resolution"],
                    record["fps"],
                    record["total_frames"],
                    record["duration_sec"],
                ]
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _unparsed_text(records: Sequence[dict[str, str]]) -> str:
    """Hiển thị các hàng vẫn cần được xác minh nhãn."""

    unparsed = [
        record
        for record in records
        if record["subject_id"] == "UNKNOWN" or record["label"] == "UNKNOWN"
    ]
    lines = [f"Các hàng cần nhãn xác minh: {len(unparsed)}"]
    lines.extend(record["relative_path"] for record in unparsed)
    return "\n".join(lines) + "\n"


def _bar_chart_png(
    counts: Counter[str],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> str:
    """Biểu diễn biểu đồ cột thành chuỗi base64 PNG."""

    labels = list(counts.keys()) or ["không có"]
    values = list(counts.values()) or [0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _figure_to_base64(fig)


def _histogram_png(
    values: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> str:
    """Biểu diễn biểu đồ phân phối (histogram) thành chuỗi base64 PNG."""

    bins = max(1, min(8, len(values)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color="#27ae60", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _figure_to_base64(fig)


def _figure_to_base64(fig: plt.Figure) -> str:
    """Tuần tự hóa (Serialize) biểu đồ matplotlib thành chuỗi base64 PNG."""

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    sys.exit(main())
