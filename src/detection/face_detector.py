"""Module phát hiện khuôn mặt tài xế (MediaPipe Tasks API) (Task T0401)."""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Dict, Any


class FaceDetector:
    def __init__(
        self,
        model_path: str = "models/pretrained/blaze_face_short_range.tflite",
        min_detection_confidence: float = 0.5,
    ):
        """Khởi tạo FaceDetector với Tasks API."""
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options, min_detection_confidence=min_detection_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Phát hiện khuôn mặt tài xế và trả về bounding box."""
        # Chuyển đổi frame sang định dạng mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.detect(mp_image)

        # Trả về UNKNOWN nếu không có mặt (FReq004)
        if not results.detections:
            return {"face_detected": False, "state": "UNKNOWN", "bbox": None}

        # Lấy bounding box của khuôn mặt đầu tiên (Tasks API trả về pixel tuyệt đối)
        bbox = results.detections[0].bounding_box
        x, y = max(0, bbox.origin_x), max(0, bbox.origin_y)

        return {
            "face_detected": True,
            "state": "DETECTED",
            "bbox": (x, y, bbox.width, bbox.height),
        }

    def release(self):
        self.detector.close()
