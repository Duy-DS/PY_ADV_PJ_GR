
class Alarm:
    """Quan ly am thanh canh bao."""

    def __init__(self, sound_enabled=True):
        self.sound_enabled = sound_enabled
        self.is_playing = False

    def activate(self):
        """Kich hoat am thanh canh bao neu duoc cho phep."""
        if self.sound_enabled:
            self.is_playing = True

    def deactivate(self):
        """Tat am thanh canh bao."""
        self.is_playing = False
