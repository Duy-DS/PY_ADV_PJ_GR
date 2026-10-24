"""Unit tests for the T0201 D3S dataset preparation utility."""

from __future__ import annotations

import io
import importlib.util
import sys
import urllib.error
import urllib.parse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "download_dataset.py"

SPEC = importlib.util.spec_from_file_location(
    "download_dataset",
    SCRIPT_PATH,
)
download_dataset = importlib.util.module_from_spec(SPEC)
sys.modules["download_dataset"] = download_dataset
assert SPEC.loader is not None
SPEC.loader.exec_module(download_dataset)


class _FakeDownloadResponse(io.BytesIO):
    """Minimal context-managed HTTP response used by download tests."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {
            "Content-Type": "application/zip",
            "Content-Length": str(len(payload)),
        }

    def geturl(self) -> str:
        """Return a stable test response URL."""

        return "https://example.test/d3s.zip"


def test_default_download_name_matches_published_video() -> None:
    """The official D3S link is stored as the MP4 it actually serves."""

    assert download_dataset.DEFAULT_ARCHIVE_NAME == "d3s_video_dataset.mp4"


def test_prepare_existing_zip_preserves_layout_and_verifies_video(
    tmp_path: Path,
) -> None:
    """A local D3S archive is extracted without flattening subject folders."""

    video_name = "subject_01_yawning.mp4"
    source_video = tmp_path / "source" / "subject_01" / "yawning" / video_name
    _write_test_video(source_video)

    archive_path = tmp_path / "d3s_video_dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(
            source_video,
            "subject_01/yawning/subject_01_yawning.mp4",
        )

    output_dir = tmp_path / "data" / "raw" / "d3s"
    summary = download_dataset.prepare_d3s_dataset(
        output_dir=output_dir,
        archive_path=archive_path,
        min_videos=1,
    )

    extracted_video = output_dir / "subject_01" / "yawning" / video_name
    assert extracted_video.exists()
    assert summary.video_count == 1
    assert summary.verified_videos[0].width == 32
    assert summary.verified_videos[0].height == 24
    assert "subject_01" in summary.verified_videos[0].path.parts
    assert "_downloads" not in summary.verified_videos[0].path.parts


def test_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    """Archive members cannot escape the configured D3S output directory."""

    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.mp4", b"not a real video")

    output_dir = tmp_path / "data" / "raw" / "d3s"
    with pytest.raises(download_dataset.DatasetExtractionError):
        download_dataset.extract_archive(archive_path, output_dir)

    assert not (tmp_path / "outside.mp4").exists()


def test_verify_dataset_videos_fails_for_unreadable_video(
    tmp_path: Path,
) -> None:
    """Invalid videos produce a clear verification exception."""

    output_dir = tmp_path / "data" / "raw" / "d3s" / "subject_01"
    output_dir.mkdir(parents=True)
    bad_video = output_dir / "corrupt.mp4"
    bad_video.write_bytes(b"not a valid mp4")

    with pytest.raises(
        download_dataset.DatasetVerificationError,
        match="failed verification",
    ):
        download_dataset.verify_dataset_videos(output_dir.parent, min_videos=1)


def test_skip_download_verifies_existing_dataset(tmp_path: Path) -> None:
    """The command can validate an already prepared D3S directory."""

    output_dir = tmp_path / "data" / "raw" / "d3s"
    existing_video = output_dir / "subject_02" / "neutral" / "sample.mp4"
    _write_test_video(existing_video)

    summary = download_dataset.prepare_d3s_dataset(
        output_dir=output_dir,
        skip_download=True,
        min_videos=1,
    )

    assert summary.video_count == 1
    assert summary.verified_videos[0].frame_count >= 1


def test_google_drive_confirmation_form_uses_modern_download_url() -> None:
    """The current Google Drive warning form is converted to a file URL."""

    file_id = "d3s-file-id"
    html = f"""
        <form id="download-form"
              action="https://drive.usercontent.google.com/download"
              method="get">
          <input type="hidden" name="id" value="{file_id}">
          <input type="hidden" name="export" value="download">
          <input type="hidden" name="confirm" value="t">
          <input type="hidden" name="uuid" value="request-uuid">
        </form>
    """

    confirmation_url = download_dataset._parse_google_drive_confirmation_form(
        html,
        "https://drive.google.com/uc",
        file_id,
    )

    assert confirmation_url is not None
    parsed_url = urllib.parse.urlparse(confirmation_url)
    query = urllib.parse.parse_qs(parsed_url.query)
    assert parsed_url.hostname == "drive.usercontent.google.com"
    assert query == {
        "confirm": ["t"],
        "export": ["download"],
        "id": [file_id],
        "uuid": ["request-uuid"],
    }


def test_download_retries_and_replaces_invalid_archive_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient failure cannot leave a corrupt final archive behind."""

    payload_buffer = io.BytesIO()
    with zipfile.ZipFile(payload_buffer, "w") as archive:
        archive.writestr("subject_01/neutral/readme.txt", "fixture")
    payload = payload_buffer.getvalue()

    attempts = 0

    def fake_open_response(
        url: str,
        timeout: float,
    ) -> _FakeDownloadResponse:
        del url, timeout
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.URLError("temporary connection failure")
        return _FakeDownloadResponse(payload)

    monkeypatch.setattr(
        download_dataset,
        "_open_download_response",
        fake_open_response,
    )
    monkeypatch.setattr(download_dataset.time, "sleep", lambda delay: None)

    destination = tmp_path / "d3s_video_dataset.zip"
    destination.write_bytes(b"incomplete previous download")
    result = download_dataset.download_file(
        "https://example.test/d3s.zip",
        destination,
        retries=2,
        timeout=1.0,
    )

    assert result == destination.resolve()
    assert attempts == 2
    assert zipfile.is_zipfile(destination)
    assert not (tmp_path / "d3s_video_dataset.part.zip").exists()


def _write_test_video(path: Path) -> None:
    """Create a tiny mp4 that OpenCV can read during unit tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot create mp4 fixtures in this environment.")

    try:
        for index in range(3):
            frame = np.full(
                (24, 32, 3),
                fill_value=index * 60,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()
