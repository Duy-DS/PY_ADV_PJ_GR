"""Input acquisition modules for video files and camera streams."""

from src.acquisition.camera import CameraStream
from src.acquisition.video_reader import VideoReader

__all__ = ["CameraStream", "VideoReader"]
