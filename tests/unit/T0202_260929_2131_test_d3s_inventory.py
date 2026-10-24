"""Unit tests for the T0202 D3S inventory catalog."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "data" / "metadata" / "d3s_inventory.csv"
SCRIPT_PATH = REPO_ROOT / "scripts" / "inventory_d3s.py"

SPEC = importlib.util.spec_from_file_location("inventory_d3s", SCRIPT_PATH)
inventory_d3s = importlib.util.module_from_spec(SPEC)
sys.modules["inventory_d3s"] = inventory_d3s
assert SPEC.loader is not None
SPEC.loader.exec_module(inventory_d3s)


def test_inventory_csv_exists_with_expected_schema() -> None:
    """The generated inventory must expose the agreed catalog columns."""

    assert INVENTORY_PATH.exists()

    with INVENTORY_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        assert reader.fieldnames == list(inventory_d3s.CSV_COLUMNS)


def test_inventory_csv_contains_non_empty_records() -> None:
    """The local D3S catalog should contain at least one video record."""

    rows = _read_inventory_rows()

    assert rows
    assert all(row["video_id"] for row in rows)
    assert all(row["relative_path"] for row in rows)
    assert all(row["file_name"] for row in rows)
    assert all(row["subject_id"] for row in rows)
    assert all(row["label"] for row in rows)
    assert all("_downloads" not in row["relative_path"] for row in rows)


def test_inventory_csv_has_valid_video_measurements() -> None:
    """OpenCV-derived measurements must be usable."""

    rows = _read_inventory_rows()

    for row in rows:
        fps = float(row["fps"])
        total_frames = int(row["total_frames"])
        duration_sec = float(row["duration_sec"])
        width = int(row["width"])
        height = int(row["height"])

        assert fps > 0.0
        assert total_frames > 0
        assert duration_sec > 0.0
        assert width > 0
        assert height > 0
        assert row["resolution"] == f"{width}x{height}"


def test_subject_and_label_parser_supports_structured_d3s_paths() -> None:
    """Future extracted D3S folders should parse subject and class labels."""

    relative_path = "subject_02/yawning/subject_02_yawning_001.mp4"

    assert inventory_d3s._parse_subject_id(relative_path) == "sub2"
    assert inventory_d3s._parse_label(relative_path) == "yawning"


def _read_inventory_rows() -> list[dict[str, str]]:
    """Read the generated inventory rows from disk."""

    with INVENTORY_PATH.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
