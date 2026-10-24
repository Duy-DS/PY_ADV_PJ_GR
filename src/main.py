"""
Điểm thực thi chính (Main Entry Point) của Hệ thống Phát hiện Buồn ngủ D3S.
Hỗ trợ cả giao diện đồ họa (GUI) cho Demo và giao diện dòng lệnh (CLI).
File: src/main.py
"""

import sys
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# Đảm bảo import được các module từ thư mục gốc
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.realtime_pipeline import RealtimePipeline


def run_system(source: str, mode: str, config_path=None):
    """Hàm bọc khởi chạy luồng Pipeline lõi và in tổng kết."""
    print("\n" + "=" * 45)
    print(" HỆ THỐNG PHÁT HIỆN BUỒN NGỦ TÀI XẾ (D3S)")
    print("=" * 45)
    print(f" Nguồn dữ liệu: {source}")
    print(f" Chế độ hoạt động: {mode.upper()}")
    print(" Đang nạp mô hình AI, vui lòng đợi...\n")

    try:
        # Khởi tạo Pipeline (Tích hợp MediaPipe, Rules, Alert, Temporal Analyzer, HUD)
        pipeline = RealtimePipeline(config_path=config_path, mode=mode)

        # Chạy luồng xử lý và nhận về báo cáo tổng kết
        summary = pipeline.run(source=source, display=True)

        print("\n=== TỔNG KẾT ===")
        print(f"- Tổng số khung hình đã xử lý : {summary.total_frames}")
        print(f"- Số khung hình báo buồn ngủ  : {summary.drowsiness_frames}")
        print(f"- Số lần bật âm thanh cảnh báo: {summary.alert_count}")
        print(f"- Tốc độ xử lý (FPS trung bình): {summary.average_fps:.1f}")
        print(f"- Độ trễ trung bình           : {summary.average_latency_ms:.1f} ms")
        print(f"- Tổng thời gian hoạt động    : {summary.elapsed_seconds:.1f} giây")
        print("==================================\n")

    except Exception as e:
        print(f"\n[ERROR] Lỗi hệ thống nghiêm trọng: {e}")
        sys.exit(1)


def show_gui_launcher():
    """Hiển thị giao diện người dùng để chọn cấu hình khởi chạy dễ dàng."""
    root = tk.Tk()
    root.title("D3S Launcher")

    window_width, window_height = 350, 250
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
    root.configure(padx=20, pady=20)

    # Biến lưu trữ trạng thái chọn
    mode_var = tk.StringVar(value="eye_yawn")

    tk.Label(root, text="CHỌN CHẾ ĐỘ HOẠT ĐỘNG", font=("Arial", 10, "bold")).pack(
        pady=(0, 5)
    )
    tk.Radiobutton(
        root,
        text="Mode B: Mắt + Ngáp (Khuyên dùng)",
        variable=mode_var,
        value="eye_yawn",
    ).pack()
    tk.Radiobutton(
        root, text="Mode A: Chỉ theo dõi Mắt", variable=mode_var, value="eye_only"
    ).pack()

    tk.Label(root, text="CHỌN NGUỒN DỮ LIỆU", font=("Arial", 10, "bold")).pack(
        pady=(15, 5)
    )

    def on_webcam():
        root.destroy()
        run_system(source="webcam", mode=mode_var.get())

    def on_video():
        file_path = filedialog.askopenfilename(
            title="Chọn Video D3S",
            initialdir=str(REPO_ROOT / "data" / "raw"),
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")],
        )
        if file_path:
            root.destroy()
            run_system(source=file_path, mode=mode_var.get())

    tk.Button(
        root,
        text="Khởi chạy với Webcam",
        bg="#2ecc71",
        fg="white",
        width=25,
        command=on_webcam,
    ).pack(pady=2)
    tk.Button(
        root,
        text="Phân tích File Video",
        bg="#3498db",
        fg="white",
        width=25,
        command=on_video,
    ).pack(pady=2)

    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Hệ thống Phát hiện Buồn ngủ D3S")
    parser.add_argument("--source", type=str, help="Nguồn: 'webcam' hoặc file path")
    parser.add_argument(
        "--mode", type=str, choices=["eye_only", "eye_yawn"], help="Chế độ phân tích"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Đường dẫn file cấu hình YAML"
    )
    return parser.parse_args()


def main():
    # Nếu chạy có tham số dòng lệnh -> dùng CLI
    if len(sys.argv) > 1:
        args = parse_args()
        if not args.source or not args.mode:
            print("[ERROR] CLI yêu cầu truyền đủ --source và --mode.")
            sys.exit(1)
        run_system(source=args.source, mode=args.mode, config_path=args.config)
    # Nếu chạy không có tham số -> bật giao diện GUI trực quan
    else:
        show_gui_launcher()


if __name__ == "__main__":
    main()
