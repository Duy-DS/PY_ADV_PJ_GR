"""Download, extract, and verify the D3S driver drowsiness dataset.

The script preserves the folder layout provided by the dataset archive so that
subject, session, and behavior/category boundaries remain intact.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
import zipfile
from dataclasses import dataclass
from http.cookiejar import CookieJar
from html.parser import HTMLParser
from pathlib import Path
from typing import Sequence

import cv2

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_D3S_VIDEO_URL = (
    "https://drive.google.com/file/d/"
    "1r27hqFlvznT8f7FyV7ipUtfOJ2nio_LA/view?usp=sharing"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "d3s"
DEFAULT_ARCHIVE_NAME = "d3s_video_dataset.mp4"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_PROGRESS_INTERVAL_SECONDS = 5.0
DEFAULT_DOWNLOAD_RETRIES = 3
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0
RETRY_BACKOFF_SECONDS = 2.0
GOOGLE_DRIVE_HTML_LIMIT = 256 * 1024
GOOGLE_DRIVE_HOSTS = {
    "drive.google.com",
    "drive.usercontent.google.com",
}
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
USER_AGENT = "Mozilla/5.0 (compatible; D3S-Dataset-Downloader/1.0)"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        """Initialize a download failure and whether retrying may help."""

        super().__init__(message)
        self.retryable = retryable


class DatasetExtractionError(RuntimeError):
    """Raised when the dataset archive cannot be safely extracted."""


class DatasetVerificationError(RuntimeError):
    """Raised when extracted dataset videos cannot be read by OpenCV."""


@dataclass(frozen=True)
class VideoVerification:
    """OpenCV readability details for one dataset video."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class DatasetSummary:
    """Summary of a prepared D3S dataset directory."""

    output_dir: Path
    video_count: int
    verified_videos: tuple[VideoVerification, ...]


class _GoogleDriveFormParser(HTMLParser):
    """Collect fields from Google Drive's download confirmation form."""

    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.fields: dict[str, str] = {}
        self._inside_download_form = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Read the confirmation form action and hidden input fields."""

        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "download-form":
            self.action = attributes.get("action")
            self._inside_download_form = True
            return

        if tag != "input" or not self._inside_download_form:
            return

        name = attributes.get("name")
        value = attributes.get("value")
        if name is not None and value is not None:
            self.fields[name] = value

    def handle_endtag(self, tag: str) -> None:
        """Stop collecting fields when the download form closes."""

        if tag == "form" and self._inside_download_form:
            self._inside_download_form = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the dataset preparation utility."""

    parser = argparse.ArgumentParser(
        description="Download, extract, and verify D3S dataset videos."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_D3S_VIDEO_URL,
        help="Dataset archive URL. Defaults to the public D3S video link.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where D3S videos should be extracted.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Use a local archive or video file instead of downloading.",
    )
    parser.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE_NAME,
        help="Filename used for the downloaded dataset archive.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only verify videos already present under --output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload or overwrite an existing local archive/video copy.",
    )
    parser.add_argument(
        "--retries",
        type=_positive_int,
        default=DEFAULT_DOWNLOAD_RETRIES,
        help="Maximum download attempts after transient failures.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        help="Network socket timeout in seconds for each request/read.",
    )
    parser.add_argument(
        "--min-videos",
        type=int,
        default=1,
        help="Minimum number of readable videos required for success.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console logging verbosity.",
    )
    return parser.parse_args(argv)


def download_file(
    url: str,
    destination: Path,
    overwrite: bool = False,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Download a URL atomically with retries and integrity checks."""

    destination = destination.resolve()
    if destination.exists() and not overwrite:
        try:
            _validate_downloaded_file(destination)
        except DatasetDownloadError as exc:
            LOGGER.warning(
                "Existing download is invalid and will be replaced: %s (%s)",
                destination,
                exc,
            )
        else:
            LOGGER.info("Using existing validated archive: %s", destination)
            return destination

    if retries < 1:
        raise ValueError("retries must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_name = f"{destination.stem}.part{destination.suffix}"
    partial_path = destination.with_name(partial_name)
    partial_path.unlink(missing_ok=True)

    last_error: DatasetDownloadError | None = None
    for attempt in range(1, retries + 1):
        LOGGER.info(
            "Download attempt %d/%d: %s",
            attempt,
            retries,
            url,
        )
        try:
            _download_once(url, partial_path, timeout)
            _validate_downloaded_file(partial_path)
            partial_path.replace(destination)
        except DatasetDownloadError as exc:
            last_error = exc
            partial_path.unlink(missing_ok=True)
            if not exc.retryable or attempt == retries:
                break

            delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            LOGGER.warning(
                "Download attempt %d/%d failed: %s. Retrying in %.1fs.",
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)
        else:
            size = destination.stat().st_size
            LOGGER.info(
                "Downloaded and validated %s (%s)",
                destination,
                _format_bytes(size),
            )
            return destination

    if last_error is None:
        raise DatasetDownloadError("Download failed for an unknown reason.")

    failure_message = "Failed to download D3S dataset after "
    failure_message += f"{attempt} attempt(s): {last_error}"
    raise DatasetDownloadError(
        failure_message,
        retryable=last_error.retryable,
    ) from last_error


def extract_archive(archive_path: Path, output_dir: Path) -> list[Path]:
    """Safely extract a zip archive while preserving its internal layout."""

    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(archive_path):
        raise DatasetExtractionError(
            f"Unsupported archive format or corrupt zip: {archive_path}"
        )

    extracted_paths: list[Path] = []
    LOGGER.info("Extracting %s into %s", archive_path, output_dir)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target_path = (output_dir / member.filename).resolve()
                _ensure_child_path(target_path, output_dir)

                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_file:
                    with target_path.open("wb") as target_file:
                        shutil.copyfileobj(source_file, target_file)
                extracted_paths.append(target_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetExtractionError(
            f"Failed to extract {archive_path}: {exc}"
        ) from exc

    LOGGER.info("Extracted %d file(s)", len(extracted_paths))
    return extracted_paths


def discover_video_files(root_dir: Path) -> list[Path]:
    """Find dataset videos, excluding downloader cache files."""

    root_dir = root_dir.resolve()
    if not root_dir.exists():
        return []

    videos: list[Path] = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if "_downloads" in path.relative_to(root_dir).parts:
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return sorted(videos)


def verify_video(video_path: Path) -> VideoVerification:
    """Verify that OpenCV can open a video and read its first frame."""

    video_path = video_path.resolve()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            message = f"OpenCV could not open video: {video_path}"
            raise DatasetVerificationError(message)

        readable, frame = capture.read()
        if not readable or frame is None:
            raise DatasetVerificationError(
                f"OpenCV could not read the first frame: {video_path}"
            )

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        height, width = frame.shape[:2]
    finally:
        capture.release()

    return VideoVerification(
        path=video_path,
        frame_count=max(frame_count, 1),
        fps=max(fps, 0.0),
        width=width,
        height=height,
    )


def verify_dataset_videos(
    output_dir: Path,
    min_videos: int = 1,
) -> tuple[VideoVerification, ...]:
    """Verify all videos under a prepared D3S output directory."""

    video_files = discover_video_files(output_dir)
    if len(video_files) < min_videos:
        raise DatasetVerificationError(
            "Expected at least "
            f"{min_videos} video(s), found {len(video_files)} in {output_dir}"
        )

    checks: list[VideoVerification] = []
    failures: list[str] = []
    for video_path in video_files:
        try:
            checks.append(verify_video(video_path))
        except DatasetVerificationError as exc:
            failures.append(str(exc))

    if failures:
        failure_text = "\n".join(failures)
        raise DatasetVerificationError(
            f"{len(failures)} video(s) failed verification:\n{failure_text}"
        )

    return tuple(checks)


def prepare_d3s_dataset(
    url: str = DEFAULT_D3S_VIDEO_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    archive_path: Path | None = None,
    archive_name: str = DEFAULT_ARCHIVE_NAME,
    skip_download: bool = False,
    overwrite: bool = False,
    min_videos: int = 1,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> DatasetSummary:
    """Prepare the local D3S dataset folder and verify readable videos."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        if archive_path is None:
            archive_path = output_dir / "_downloads" / archive_name
            archive_path = download_file(
                url,
                archive_path,
                overwrite,
                retries,
                timeout,
            )
        else:
            archive_path = archive_path.resolve()

        if zipfile.is_zipfile(archive_path):
            extract_archive(archive_path, output_dir)
        elif archive_path.suffix.lower() in VIDEO_EXTENSIONS:
            _copy_single_video(archive_path, output_dir, overwrite)
        else:
            raise DatasetExtractionError(
                "Downloaded file is neither a supported video nor a zip "
                f"archive: {archive_path}"
            )

    verified_videos = verify_dataset_videos(output_dir, min_videos)
    return DatasetSummary(
        output_dir=output_dir,
        video_count=len(verified_videos),
        verified_videos=verified_videos,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the D3S dataset preparation command."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        summary = prepare_d3s_dataset(
            url=args.url,
            output_dir=args.output_dir,
            archive_path=args.archive_path,
            archive_name=args.archive_name,
            skip_download=args.skip_download,
            overwrite=args.overwrite,
            min_videos=args.min_videos,
            retries=args.retries,
            timeout=args.timeout,
        )
    except (
        DatasetDownloadError,
        DatasetExtractionError,
        DatasetVerificationError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "D3S dataset ready: %d readable video(s) under %s",
        summary.video_count,
        summary.output_dir,
    )
    return 0


def _download_once(url: str, destination: Path, timeout: float) -> None:
    """Stream one download attempt and detect incomplete responses."""

    try:
        response = _open_download_response(url, timeout)
    except DatasetDownloadError:
        raise
    except urllib.error.HTTPError as exc:
        retryable = exc.code in TRANSIENT_HTTP_STATUS_CODES
        raise DatasetDownloadError(
            f"HTTP {exc.code} {exc.reason} while requesting {url}",
            retryable=retryable,
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise DatasetDownloadError(
            f"Network connection failed while requesting {url}: {exc}"
        ) from exc

    try:
        with response, destination.open("wb") as output_file:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise DatasetDownloadError(
                    "The download URL returned an HTML page instead of "
                    "dataset content. Check the URL and sharing permissions.",
                    retryable=False,
                )

            expected_bytes = _content_length(response)
            downloaded_bytes = 0
            last_progress = time.monotonic()
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output_file.write(chunk)
                downloaded_bytes += len(chunk)

                now = time.monotonic()
                if now - last_progress >= DOWNLOAD_PROGRESS_INTERVAL_SECONDS:
                    LOGGER.info(
                        "Downloaded %s%s",
                        _format_bytes(downloaded_bytes),
                        _expected_size_text(expected_bytes),
                    )
                    last_progress = now
    except DatasetDownloadError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        message = f"Download stream was interrupted: {exc}"
        raise DatasetDownloadError(message) from exc

    if expected_bytes is not None and downloaded_bytes != expected_bytes:
        raise DatasetDownloadError(
            "Incomplete download: received "
            f"{downloaded_bytes} of {expected_bytes} bytes."
        )


def _open_download_response(
    url: str,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> urllib.response.addinfourl:
    """Open a normal URL or follow Google Drive confirmation pages."""

    file_id = _extract_google_drive_file_id(url)
    if file_id is None:
        request = _build_request(url)
        return urllib.request.urlopen(request, timeout=timeout)

    cookie_jar = CookieJar()
    cookie_processor = urllib.request.HTTPCookieProcessor(cookie_jar)
    opener = urllib.request.build_opener(cookie_processor)
    request_url = _build_google_drive_download_url(file_id)

    for _ in range(3):
        response = opener.open(_build_request(request_url), timeout=timeout)
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            return response

        try:
            request_url = _find_google_drive_confirmation_url(
                response,
                cookie_jar,
                file_id,
            )
        finally:
            response.close()

        LOGGER.debug(
            "Following Google Drive confirmation URL: %s",
            request_url,
        )

    raise DatasetDownloadError(
        "Google Drive returned too many confirmation pages.",
        retryable=False,
    )


def _extract_google_drive_file_id(url: str) -> str | None:
    """Extract a Google Drive file id from common sharing URL formats."""

    parsed_url = urllib.parse.urlparse(url)
    if "drive.google.com" not in parsed_url.netloc:
        return None

    query_params = urllib.parse.parse_qs(parsed_url.query)
    if query_params.get("id"):
        return query_params["id"][0]

    match = re.search(r"/file/d/([^/]+)", parsed_url.path)
    if match:
        return match.group(1)

    return None


def _build_google_drive_download_url(
    file_id: str,
    confirm_token: str | None = None,
) -> str:
    """Build a Google Drive direct download URL."""

    query_params = {"export": "download", "id": file_id}
    if confirm_token is not None:
        query_params["confirm"] = confirm_token
    encoded_params = urllib.parse.urlencode(query_params)
    return f"https://drive.google.com/uc?{encoded_params}"


def _find_google_drive_confirmation_url(
    response: urllib.response.addinfourl,
    cookie_jar: CookieJar,
    file_id: str,
) -> str:
    """Build a continuation URL from a Google Drive warning page."""

    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return _build_google_drive_download_url(file_id, cookie.value)

    preview = response.read(GOOGLE_DRIVE_HTML_LIMIT).decode(
        "utf-8",
        errors="ignore",
    )
    form_url = _parse_google_drive_confirmation_form(
        preview,
        response.geturl(),
        file_id,
    )
    if form_url is not None:
        return form_url

    match = re.search(r"confirm=([0-9A-Za-z_-]+)", preview)
    if match:
        return _build_google_drive_download_url(file_id, match.group(1))

    reason = _describe_google_drive_html(preview)
    raise DatasetDownloadError(reason, retryable=False)


def _parse_google_drive_confirmation_form(
    html: str,
    response_url: str,
    file_id: str,
) -> str | None:
    """Parse Drive's modern download form into a validated URL."""

    parser = _GoogleDriveFormParser()
    parser.feed(html)
    if parser.action is None:
        return None

    action_url = urllib.parse.urljoin(response_url, parser.action)
    action_host = urllib.parse.urlparse(action_url).hostname
    if action_host not in GOOGLE_DRIVE_HOSTS:
        raise DatasetDownloadError(
            "Google Drive supplied an unexpected confirmation host: "
            f"{action_host or 'missing host'}.",
            retryable=False,
        )

    form_file_id = parser.fields.get("id")
    if form_file_id is not None and form_file_id != file_id:
        raise DatasetDownloadError(
            "Google Drive confirmation page referenced a different file.",
            retryable=False,
        )

    parser.fields.setdefault("id", file_id)
    parser.fields.setdefault("export", "download")
    encoded_fields = urllib.parse.urlencode(parser.fields)
    return f"{action_url}?{encoded_fields}"


def _describe_google_drive_html(html: str) -> str:
    """Convert common Google Drive HTML failures into actionable errors."""

    normalized = re.sub(r"\s+", " ", html).lower()
    if "too many users have viewed or downloaded this file" in normalized:
        return (
            "Google Drive download quota is exceeded for the D3S file. "
            "Wait for the quota to reset or use --archive-path with a "
            "manually downloaded copy."
        )
    if "request access" in normalized or "you need access" in normalized:
        return (
            "The D3S Google Drive file is not publicly accessible. "
            "Request access or use --archive-path with an authorized copy."
        )
    if "sign in" in normalized and "download" in normalized:
        return (
            "Google Drive requires authentication for the D3S file. "
            "Download it in a browser and pass the file via --archive-path."
        )

    return (
        "Google Drive returned an unsupported HTML response instead of the "
        "D3S archive. The sharing page format may have changed; use "
        "--archive-path with a manually downloaded copy."
    )


def _build_request(url: str) -> urllib.request.Request:
    """Build a download request with headers accepted by common hosts."""

    return urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream,*/*;q=0.8",
            "User-Agent": USER_AGENT,
        },
    )


def _content_length(response: urllib.response.addinfourl) -> int | None:
    """Read a valid non-negative Content-Length response header."""

    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError:
        LOGGER.warning("Ignoring invalid Content-Length header: %s", value)
        return None
    return length if length >= 0 else None


def _validate_downloaded_file(path: Path) -> None:
    """Reject empty or structurally invalid downloaded files."""

    if not path.is_file() or path.stat().st_size == 0:
        raise DatasetDownloadError(f"Downloaded file is empty: {path}")

    if path.suffix.lower() != ".zip":
        return

    if not zipfile.is_zipfile(path):
        message = f"Downloaded ZIP is incomplete or corrupt: {path}"
        raise DatasetDownloadError(message)

    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                message = f"Downloaded ZIP contains no files: {path}"
                raise DatasetDownloadError(message)
    except zipfile.BadZipFile as exc:
        message = f"Downloaded ZIP is corrupt: {path}"
        raise DatasetDownloadError(message) from exc


def _expected_size_text(expected_bytes: int | None) -> str:
    """Format optional total size text for progress logging."""

    if expected_bytes is None:
        return ""
    return f" of {_format_bytes(expected_bytes)}"


def _format_bytes(byte_count: int) -> str:
    """Format a byte count for concise terminal progress messages."""

    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a strictly positive command-line float."""

    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _copy_single_video(
    source_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    """Copy one standalone video into the D3S output directory."""

    source_path = source_path.resolve()
    target_path = (output_dir / source_path.name).resolve()
    _ensure_child_path(target_path, output_dir.resolve())

    if target_path.exists() and not overwrite:
        LOGGER.info("Using existing video: %s", target_path)
        return target_path

    shutil.copy2(source_path, target_path)
    LOGGER.info("Copied video to %s", target_path)
    return target_path


def _ensure_child_path(path: Path, parent: Path) -> None:
    """Ensure ``path`` resolves inside ``parent``."""

    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise DatasetExtractionError(
            f"Refusing to write outside dataset directory: {path}"
        ) from exc


if __name__ == "__main__":
    sys.exit(main())
