"""Create an executed D3S exploration notebook from the inventory CSV."""

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
    """Parse command-line arguments for notebook generation."""

    parser = argparse.ArgumentParser(
        description="Build an executed D3S dataset exploration notebook."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help="Input D3S inventory CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NOTEBOOK_PATH,
        help="Output notebook path.",
    )
    return parser.parse_args(argv)


def read_inventory(inventory_path: Path) -> list[dict[str, str]]:
    """Read the inventory CSV into plain dictionaries."""

    with inventory_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_notebook(
    records: Sequence[dict[str, str]],
    inventory_path: Path,
    output_path: Path,
) -> None:
    """Write a valid executed notebook with chart outputs."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = _build_notebook(records, inventory_path)
    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run notebook generation."""

    args = parse_args(argv)
    records = read_inventory(args.inventory)
    write_notebook(records, args.inventory, args.output)
    print(f"Wrote executed dataset notebook to {args.output}")
    return 0


def _build_notebook(
    records: Sequence[dict[str, str]],
    inventory_path: Path,
) -> dict[str, Any]:
    """Build a notebook JSON structure with precomputed outputs."""

    cells: list[dict[str, Any]] = [
        _markdown_cell(
            "# D3S Dataset Exploration\n\n"
            "Task T0202 inventory review for the Driver Drowsiness Detection "
            "System. The notebook reads `data/metadata/d3s_inventory.csv`, "
            "summarizes video-level metadata, and records split guidance that "
            "keeps subject/video boundaries intact."
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
                    "print(f'Loaded {len(records)} video record(s) from "
                    "{inventory_path}')",
                ]
            ),
            execution_count=1,
            text_output=(
                "Loaded "
                f"{len(records)} video record(s) from "
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
            "## Metadata Provenance\n\n"
            "Subject and class labels are parsed only from path or filename "
            "tokens such as `subject_02/yawning/sample.mp4`. The current "
            "local D3S artifact is a single monolithic MP4 named "
            "`d3s_video_dataset.mp4`, so its subject and label fields "
            "remain `UNKNOWN` until the team imports a structured "
            "video/frame release or adds verified segment annotations."
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
                    "ax.set_title('D3S class frequency by video record')",
                    "ax.set_xlabel('Label')",
                    "ax.set_ylabel('Video count')",
                    "ax.grid(axis='y', alpha=0.25)",
                    "plt.tight_layout()",
                ]
            ),
            execution_count=3,
            image_png=_bar_chart_png(
                Counter(record["label"] for record in records),
                "D3S class frequency by video record",
                "Label",
                "Video count",
                "#2f80ed",
            ),
        ),
        _code_cell(
            "\n".join(
                [
                    "durations = [float(record['duration_sec']) "
                    "for record in records]",
                    "fig, ax = plt.subplots(figsize=(6, 4))",
                    "ax.hist(durations, bins=max(1, min(8, len(durations))), "
                    "color='#27ae60', edgecolor='white')",
                    "ax.set_title('D3S video duration distribution')",
                    "ax.set_xlabel('Duration (seconds)')",
                    "ax.set_ylabel('Video count')",
                    "ax.grid(axis='y', alpha=0.25)",
                    "plt.tight_layout()",
                ]
            ),
            execution_count=4,
            image_png=_histogram_png(
                [float(record["duration_sec"]) for record in records],
                "D3S video duration distribution",
                "Duration (seconds)",
                "Video count",
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
                    "ax.set_title('D3S subject balance by video record')",
                    "ax.set_xlabel('Subject')",
                    "ax.set_ylabel('Video count')",
                    "ax.grid(axis='y', alpha=0.25)",
                    "plt.tight_layout()",
                ]
            ),
            execution_count=5,
            image_png=_bar_chart_png(
                Counter(record["subject_id"] for record in records),
                "D3S subject balance by video record",
                "Subject",
                "Video count",
                "#f2994a",
            ),
        ),
        _markdown_cell(
            "## Subject-Level Split Guidance\n\n"
            "- Split by subject when subject IDs are available: for example, "
            "train on two subjects and reserve the third subject for "
            "validation or testing.\n"
            "- If a subject cannot be separated, split by complete source "
            "video or verified contiguous session, never by randomly "
            "shuffled frames.\n"
            "- Keep neighboring frames from the same driver/session in the "
            "same partition so temporal signals do not leak across "
            "evaluation sets.\n"
            "- Report `UNKNOWN` and invalid records separately during "
            "evaluation instead of dropping them silently."
        ),
        _code_cell(
            "\n".join(
                [
                    "unparsed = [",
                    "    record for record in records",
                    "    if record['subject_id'] == 'UNKNOWN'",
                    "    or record['label'] == 'UNKNOWN'",
                    "]",
                    "print(f'Rows needing verified labels: {len(unparsed)}')",
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
    """Create a markdown notebook cell."""

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _display_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _python_version_text() -> str:
    """Return the short Python version for notebook metadata."""

    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _code_cell(
    source: str,
    execution_count: int,
    text_output: str | None = None,
    image_png: str | None = None,
) -> dict[str, Any]:
    """Create an executed code cell with text or PNG output."""

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
    """Render compact inventory rows for notebook stdout."""

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
    """Render rows that still need verified labels."""

    unparsed = [
        record
        for record in records
        if record["subject_id"] == "UNKNOWN" or record["label"] == "UNKNOWN"
    ]
    lines = [f"Rows needing verified labels: {len(unparsed)}"]
    lines.extend(record["relative_path"] for record in unparsed)
    return "\n".join(lines) + "\n"


def _bar_chart_png(
    counts: Counter[str],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> str:
    """Render a bar chart to base64 PNG."""

    labels = list(counts.keys()) or ["none"]
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
    """Render a histogram to base64 PNG."""

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
    """Serialize a matplotlib figure to base64 PNG."""

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    sys.exit(main())
