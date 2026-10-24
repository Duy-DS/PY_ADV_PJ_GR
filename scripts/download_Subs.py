"""
Script tự động tải Dataset (Sub1, Sub2, Sub3) từ Google Drive, giải nén và dọn dẹp.
File: scripts/download_data.py
"""

import sys
import shutil
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    print("[ERROR] Thư viện 'gdown' chưa được cài đặt.")
    print("Vui lòng chạy lệnh: pip install gdown")
    sys.exit(1)

# Cấu hình đường dẫn
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
TEMP_DIR = REPO_ROOT / "temp_download"
DRIVE_URL = "https://drive.google.com/drive/folders/17CcewK43h1R74wxAWPulqdCuVv9MAi2w?usp=sharing"


def main():

    # 1. Chuẩn bị thư mục
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Tải toàn bộ nội dung từ Link Folder Drive
    print(f"\n!!! Kết nối tới Google Drive và tải dữ liệu")
    try:
        # Tải folder Drive vào thư mục tạm (hỗ trợ cả link folder)
        gdown.download_folder(
            url=DRIVE_URL, output=str(TEMP_DIR), quiet=False, use_cookies=False
        )
    except Exception as e:
        print(f"\nLỗi khi tải từ Google Drive: {e}")
        shutil.rmtree(TEMP_DIR)
        sys.exit(1)

    # 3. Tìm và giải nén file .zip (nếu có)
    print("\n!!! Kiểm tra định dạng dữ liệu và giải nén")
    zip_files = list(TEMP_DIR.rglob("*.zip"))

    if zip_files:
        for zip_path in zip_files:
            print(f"  -> Đang giải nén: {zip_path.name}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(TEMP_DIR)
    else:
        print("  -> Không tìm thấy file .zip, hệ thống sẽ quyét thư mục trực tiếp.")

    # 4. Tìm các thư mục mục tiêu (Sub1, Sub2, Sub3) và chuyển về data/raw
    print("\n!!! Cấu trúc lại dữ liệu vào data/raw/")
    target_folders = ["Sub1", "Sub2", "Sub3"]
    folders_found = 0

    for target in target_folders:
        # Quét sâu (rglob) để tìm folder bất kể nó nằm ở cấp độ nào trong file zip
        found_paths = list(TEMP_DIR.rglob(target))
        for path in found_paths:
            if path.is_dir():
                dest_path = RAW_DIR / path.name

                # Xóa folder cũ trong data/raw nếu đã tồn tại để tránh xung đột
                if dest_path.exists():
                    shutil.rmtree(dest_path)

                # Di chuyển folder từ thư mục tạm sang data/raw
                shutil.move(str(path), str(RAW_DIR))
                print(
                    f"  -> Đã chuyển {path.name} thành công tới: {dest_path.relative_to(REPO_ROOT)}"
                )
                folders_found += 1
                break  # Chuyển xong thì dừng tìm target này để tránh lặp (do có thể có sub-folder trùng tên)

    if folders_found == 0:
        print(f"Không tìm thấy thư mục {target_folders} nào trong dữ liệu tải về!")

    # 5. Dọn dẹp thư mục tạm (Bao gồm xóa cả file zip và các thư mục rác khi giải nén)
    print("\n!!! Đang dọn dẹp các tập tin tạm")
    shutil.rmtree(TEMP_DIR)

    print("\n" + "=" * 50)
    print("HOÀN TẤT")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
