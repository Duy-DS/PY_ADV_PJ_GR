import os
import pygame


class Alarm:
    """Quản lý và phát âm thanh cảnh báo bằng pygame (không gây block luồng chính)."""

    def __init__(self, sound_enabled=True, sound_file="data/alarm.wav"):
        self.sound_enabled = sound_enabled
        self.is_playing = False
        self.sound = None

        if self.sound_enabled:
            try:
                pygame.mixer.init()

                # Nạp file âm thanh (đảm bảo bạn đã lưu file alarm.wav vào thư mục data/)
                if os.path.exists(sound_file):
                    self.sound = pygame.mixer.Sound(sound_file)
                else:
                    print(f" Không tìm thấy file âm thanh tại: {sound_file}")
            except Exception as e:
                print(f" Không thể khởi tạo hệ thống âm thanh: {e}")

    def activate(self):
        """Kích hoạt âm thanh cảnh báo lặp lại liên tục."""
        if self.sound_enabled and not self.is_playing:
            self.is_playing = True
            if self.sound:
                # Tham số loops=-1 giúp âm thanh lặp lại cho đến khi có lệnh dừng
                self.sound.play(loops=-1)

    def deactivate(self):
        """Tắt âm thanh cảnh báo."""
        if self.is_playing:
            self.is_playing = False
            if self.sound:
                self.sound.stop()
