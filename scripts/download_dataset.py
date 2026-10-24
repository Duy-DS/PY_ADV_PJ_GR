"""Tải xuống, giải nén và xác minh tập dữ liệu buồn ngủ của tài xế D3S.

Kịch bản này giữ nguyên bố cục thư mục do kho lưu trữ dữ liệu cung cấp
để các ranh giới về đối tượng, phiên và hành vi/phân loại được giữ nguyên vẹn.
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
    "https://drive.google.com/file/d/1r27hqFlvznT8f7FyV7ipUtfOJ2nio_LA/view?usp=sharing"
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
    """Báo lỗi khi không thể tải tập dữ liệu."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        """Khởi tạo trạng thái lỗi tải xuống và cho biết có thể thử lại hay không."""

        super().__init__(message)
        self.retryable = retryable


class DatasetExtractionError(RuntimeError):
    """Báo lỗi khi không thể giải nén an toàn kho lưu trữ tập dữ liệu."""


class DatasetVerificationError(RuntimeError):
    """Báo lỗi khi OpenCV không thể đọc được các video dữ liệu đã giải nén."""


@dataclass(frozen=True)
class VideoVerification:
    """Chi tiết khả năng đọc của OpenCV cho một video dữ liệu."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class DatasetSummary:
    """Bản tóm tắt về thư mục tập dữ liệu D3S đã được chuẩn bị."""

    output_dir: Path
    video_count: int
    verified_videos: tuple[VideoVerification, ...]


class _GoogleDriveFormParser(HTMLParser):
    """Thu thập các trường từ biểu mẫu xác nhận tải xuống của Google Drive."""

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
        """Đọc action của biểu mẫu xác nhận và các trường nhập liệu ẩn."""

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
        """Dừng thu thập các trường khi biểu mẫu tải xuống đóng."""

        if tag == "form" and self._inside_download_form:
            self._inside_download_form = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Phân tích các tùy chọn dòng lệnh cho công cụ chuẩn bị tập dữ liệu."""

    parser = argparse.ArgumentParser(
        description="Tải xuống, giải nén và xác minh video của tập dữ liệu D3S."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_D3S_VIDEO_URL,
        help="URL của file nén tập dữ liệu. Mặc định là liên kết video D3S công khai.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thư mục nơi các video D3S sẽ được giải nén.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Sử dụng file nén hoặc file video cục bộ thay vì tải xuống.",
    )
    parser.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE_NAME,
        help="Tên file sử dụng cho file nén tập dữ liệu đã tải xuống.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Chỉ xác minh các video đã có sẵn trong --output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Tải lại hoặc ghi đè lên bản sao file nén/video cục bộ đã có.",
    )
    parser.add_argument(
        "--retries",
        type=_positive_int,
        default=DEFAULT_DOWNLOAD_RETRIES,
        help="Số lần thử tải xuống tối đa sau các lỗi tạm thời.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        help="Thời gian chờ (timeout) socket mạng tính bằng giây cho mỗi yêu cầu/đọc.",
    )
    parser.add_argument(
        "--min-videos",
        type=int,
        default=1,
        help="Số lượng video có thể đọc tối thiểu cần thiết để báo thành công.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Mức độ chi tiết của log trên console.",
    )
    return parser.parse_args(argv)


def download_file(
    url: str,
    destination: Path,
    overwrite: bool = False,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Tải xuống một URL một cách nguyên tử (atomic) với các lần thử lại và kiểm tra tính toàn vẹn."""

    destination = destination.resolve()
    if destination.exists() and not overwrite:
        try:
            _validate_downloaded_file(destination)
        except DatasetDownloadError as exc:
            LOGGER.warning(
                "Bản tải xuống hiện tại không hợp lệ và sẽ bị thay thế: %s (%s)",
                destination,
                exc,
            )
        else:
            LOGGER.info("Sử dụng bản tải xuống đã xác thực có sẵn: %s", destination)
            return destination

    if retries < 1:
        raise ValueError("retries phải ít nhất là 1")
    if timeout <= 0:
        raise ValueError("timeout phải lớn hơn 0")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_name = f"{destination.stem}.part{destination.suffix}"
    partial_path = destination.with_name(partial_name)
    partial_path.unlink(missing_ok=True)

    last_error: DatasetDownloadError | None = None
    for attempt in range(1, retries + 1):
        LOGGER.info(
            "Nỗ lực tải xuống lần %d/%d: %s",
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
                "Nỗ lực tải xuống lần %d/%d thất bại: %s. Đang thử lại trong %.1fs.",
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)
        else:
            size = destination.stat().st_size
            LOGGER.info(
                "Đã tải xuống và xác thực %s (%s)",
                destination,
                _format_bytes(size),
            )
            return destination

    if last_error is None:
        raise DatasetDownloadError("Tải xuống thất bại vì một lý do không xác định.")

    failure_message = "Tải tập dữ liệu D3S thất bại sau "
    failure_message += f"{attempt} nỗ lực: {last_error}"
    raise DatasetDownloadError(
        failure_message,
        retryable=last_error.retryable,
    ) from last_error


def extract_archive(archive_path: Path, output_dir: Path) -> list[Path]:
    """Giải nén an toàn kho lưu trữ zip trong khi vẫn giữ nguyên bố cục bên trong của nó."""

    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(archive_path):
        raise DatasetExtractionError(
            f"Định dạng nén không được hỗ trợ hoặc zip bị hỏng: {archive_path}"
        )

    extracted_paths: list[Path] = []
    LOGGER.info("Đang giải nén %s vào %s", archive_path, output_dir)

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
            f"Giải nén thất bại {archive_path}: {exc}"
        ) from exc

    LOGGER.info("Đã giải nén %d file", len(extracted_paths))
    return extracted_paths


def discover_video_files(root_dir: Path) -> list[Path]:
    """Tìm các video của tập dữ liệu, loại trừ các file cache của trình tải xuống."""

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
    """Xác minh rằng OpenCV có thể mở một video và đọc khung hình đầu tiên của nó."""

    video_path = video_path.resolve()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            message = f"OpenCV không thể mở video: {video_path}"
            raise DatasetVerificationError(message)

        readable, frame = capture.read()
        if not readable or frame is None:
            raise DatasetVerificationError(
                f"OpenCV không thể đọc khung hình đầu tiên: {video_path}"
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
    """Xác minh tất cả các video dưới một thư mục đầu ra D3S đã chuẩn bị."""

    video_files = discover_video_files(output_dir)
    if len(video_files) < min_videos:
        raise DatasetVerificationError(
            "Cần ít nhất "
            f"{min_videos} video, tìm thấy {len(video_files)} trong {output_dir}"
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
            f"{len(failures)} video(s) thất bại khi xác minh:\n{failure_text}"
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
    """Chuẩn bị thư mục tập dữ liệu D3S cục bộ và xác minh các video có thể đọc được."""

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
                "File tải xuống không phải là video được hỗ trợ hay file nén zip: "
                f"{archive_path}"
            )

    verified_videos = verify_dataset_videos(output_dir, min_videos)
    return DatasetSummary(
        output_dir=output_dir,
        video_count=len(verified_videos),
        verified_videos=verified_videos,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy lệnh chuẩn bị tập dữ liệu D3S."""

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
        "Tập dữ liệu D3S đã sẵn sàng: %d video có thể đọc được dưới thư mục %s",
        summary.video_count,
        summary.output_dir,
    )
    return 0


def _download_once(url: str, destination: Path, timeout: float) -> None:
    """Phát (stream) một nỗ lực tải xuống và phát hiện các phản hồi không đầy đủ."""

    try:
        response = _open_download_response(url, timeout)
    except DatasetDownloadError:
        raise
    except urllib.error.HTTPError as exc:
        retryable = exc.code in TRANSIENT_HTTP_STATUS_CODES
        raise DatasetDownloadError(
            f"Lỗi HTTP {exc.code} {exc.reason} khi đang yêu cầu {url}",
            retryable=retryable,
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise DatasetDownloadError(
            f"Mất kết nối mạng khi đang yêu cầu {url}: {exc}"
        ) from exc

    try:
        with response, destination.open("wb") as output_file:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise DatasetDownloadError(
                    "URL tải xuống trả về một trang HTML thay vì nội dung "
                    "tập dữ liệu. Hãy kiểm tra URL và quyền chia sẻ.",
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
                        "Đã tải %s%s",
                        _format_bytes(downloaded_bytes),
                        _expected_size_text(expected_bytes),
                    )
                    last_progress = now
    except DatasetDownloadError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        message = f"Luồng tải xuống bị gián đoạn: {exc}"
        raise DatasetDownloadError(message) from exc

    if expected_bytes is not None and downloaded_bytes != expected_bytes:
        raise DatasetDownloadError(
            "Tải xuống không đầy đủ: đã nhận "
            f"{downloaded_bytes} trên tổng số {expected_bytes} bytes."
        )


def _open_download_response(
    url: str,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> urllib.response.addinfourl:
    """Mở một URL thông thường hoặc theo dõi các trang xác nhận của Google Drive."""

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
            "Đang theo dõi URL xác nhận của Google Drive: %s",
            request_url,
        )

    raise DatasetDownloadError(
        "Google Drive trả về quá nhiều trang xác nhận.",
        retryable=False,
    )


def _extract_google_drive_file_id(url: str) -> str | None:
    """Trích xuất file ID của Google Drive từ các định dạng URL chia sẻ phổ biến."""

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
    """Xây dựng một URL tải xuống trực tiếp của Google Drive."""

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
    """Xây dựng một URL tiếp nối từ một trang cảnh báo của Google Drive."""

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
    """Phân tích biểu mẫu tải xuống hiện đại của Drive thành một URL đã được xác thực."""

    parser = _GoogleDriveFormParser()
    parser.feed(html)
    if parser.action is None:
        return None

    action_url = urllib.parse.urljoin(response_url, parser.action)
    action_host = urllib.parse.urlparse(action_url).hostname
    if action_host not in GOOGLE_DRIVE_HOSTS:
        raise DatasetDownloadError(
            "Google Drive đã cung cấp một host xác nhận không mong muốn: "
            f"{action_host or 'không có host'}.",
            retryable=False,
        )

    form_file_id = parser.fields.get("id")
    if form_file_id is not None and form_file_id != file_id:
        raise DatasetDownloadError(
            "Trang xác nhận của Google Drive tham chiếu đến một file khác.",
            retryable=False,
        )

    parser.fields.setdefault("id", file_id)
    parser.fields.setdefault("export", "download")
    encoded_fields = urllib.parse.urlencode(parser.fields)
    return f"{action_url}?{encoded_fields}"


def _describe_google_drive_html(html: str) -> str:
    """Chuyển đổi các lỗi HTML phổ biến của Google Drive thành thông báo lỗi cụ thể."""

    normalized = re.sub(r"\s+", " ", html).lower()
    # Các chuỗi tĩnh bên dưới giữ nguyên tiếng Anh để khớp với thông báo máy chủ
    if "too many users have viewed or downloaded this file" in normalized:
        return (
            "Quota tải xuống của Google Drive đã vượt giới hạn cho file D3S. "
            "Hãy đợi quota reset hoặc dùng tham số --archive-path với một "
            "bản copy đã tải xuống thủ công."
        )
    if "request access" in normalized or "you need access" in normalized:
        return (
            "File Google Drive D3S không ở trạng thái truy cập công khai. "
            "Hãy yêu cầu quyền truy cập hoặc dùng --archive-path với bản copy được ủy quyền."
        )
    if "sign in" in normalized and "download" in normalized:
        return (
            "Google Drive yêu cầu đăng nhập đối với file D3S. "
            "Hãy tải file trên trình duyệt web rồi truyền tham số --archive-path."
        )

    return (
        "Google Drive trả về phản hồi HTML không được hỗ trợ thay vì "
        "file nén D3S. Định dạng trang chia sẻ có thể đã thay đổi; hãy dùng "
        "--archive-path với bản copy tải thủ công."
    )


def _build_request(url: str) -> urllib.request.Request:
    """Xây dựng một yêu cầu tải xuống với các header được các host phổ biến chấp nhận."""

    return urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream,*/*;q=0.8",
            "User-Agent": USER_AGENT,
        },
    )


def _content_length(response: urllib.response.addinfourl) -> int | None:
    """Đọc một header phản hồi Content-Length hợp lệ không âm."""

    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError:
        LOGGER.warning("Bỏ qua header Content-Length không hợp lệ: %s", value)
        return None
    return length if length >= 0 else None


def _validate_downloaded_file(path: Path) -> None:
    """Từ chối các file tải về rỗng hoặc sai cấu trúc."""

    if not path.is_file() or path.stat().st_size == 0:
        raise DatasetDownloadError(f"File tải xuống bị rỗng: {path}")

    if path.suffix.lower() != ".zip":
        return

    if not zipfile.is_zipfile(path):
        message = f"File ZIP tải xuống bị thiếu hoặc hỏng: {path}"
        raise DatasetDownloadError(message)

    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                message = f"File ZIP tải xuống không chứa file nào: {path}"
                raise DatasetDownloadError(message)
    except zipfile.BadZipFile as exc:
        message = f"File ZIP tải xuống bị hỏng: {path}"
        raise DatasetDownloadError(message) from exc


def _expected_size_text(expected_bytes: int | None) -> str:
    """Định dạng văn bản hiển thị tổng kích thước nếu có để ghi log tiến trình."""

    if expected_bytes is None:
        return ""
    return f" trên tổng số {_format_bytes(expected_bytes)}"


def _format_bytes(byte_count: int) -> str:
    """Định dạng số byte để xuất ra thông báo tiến trình ngắn gọn trên terminal."""

    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def _positive_int(value: str) -> int:
    """Phân tích một số nguyên dương từ dòng lệnh."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("giá trị phải ít nhất là 1")
    return parsed


def _positive_float(value: str) -> float:
    """Phân tích một số thực dương từ dòng lệnh."""

    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("giá trị phải lớn hơn 0")
    return parsed


def _copy_single_video(
    source_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    """Sao chép một video độc lập vào thư mục đầu ra D3S."""

    source_path = source_path.resolve()
    target_path = (output_dir / source_path.name).resolve()
    _ensure_child_path(target_path, output_dir.resolve())

    if target_path.exists() and not overwrite:
        LOGGER.info("Sử dụng video có sẵn: %s", target_path)
        return target_path

    shutil.copy2(source_path, target_path)
    LOGGER.info("Đã sao chép video tới %s", target_path)
    return target_path


def _ensure_child_path(path: Path, parent: Path) -> None:
    """Đảm bảo ``path`` thuộc bên trong ``parent``."""

    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise DatasetExtractionError(
            f"Từ chối ghi ra ngoài thư mục tập dữ liệu: {path}"
        ) from exc


if __name__ == "__main__":
    sys.exit(main())
