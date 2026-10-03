"""Script tự động tải các model của MediaPipe."""

import urllib.request
from pathlib import Path
import sys

# Xác định thư mục gốc dựa trên vị trí của file này (nằm trong thư mục scripts/)
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models" / "pretrained"

# Link tải model chính thức từ Google MediaPipe
MODELS_URLS = {
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def download_progress(count: int, block_size: int, total_size: int):
    """Hiển thị % tiến độ tải file."""
    percent = int(count * block_size * 100 / total_size)
    # Giới hạn phần trăm ở mức 100%
    percent = min(100, percent)
    sys.stdout.write(f"\r  Tiến độ: {percent}%")
    sys.stdout.flush()


def download_model(url: str, dest_path: Path):
    """Tải một file model từ URL về máy."""
    print(f"\nĐang tải {dest_path.name}...")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
        print(f"\n[SUCCESS] Đã lưu tại: {dest_path}")
    except Exception as e:
        print(f"\n[ERROR] Lỗi khi tải {dest_path.name}: {e}", file=sys.stderr)
        # Xóa file lỗi nếu tải dở dang
        if dest_path.exists():
            dest_path.unlink()


def main():
    # 1. Tạo thư mục models/pretrained nếu chưa tồn tại
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Đã chuẩn bị thư mục: {MODELS_DIR}")

    # 2. Kiểm tra và tải từng model
    for filename, url in MODELS_URLS.items():
        dest_path = MODELS_DIR / filename
        if dest_path.exists():
            print(f"\n[INFO] Model '{filename}' đã tồn tại. Bỏ qua tải.")
        else:
            download_model(url, dest_path)

    print("\n[INFO] HOÀN TẤT CHUẨN BỊ MODEL!")


if __name__ == "__main__":
    main()
