"""Module trích xuất mốc hình học khuôn mặt (MediaPipe Tasks API) (Task T0402)."""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Dict, Any, List, Tuple


class LandmarkDetector:
    def __init__(
        self,
        model_path: str = "models/pretrained/face_landmarker.task",
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        """Khởi tạo FaceLandmarker với Tasks API."""
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE_INDICES = [362, 385, 386, 263, 374, 373]
        self.MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Trích xuất tọa độ điểm mốc mắt và môi."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect(mp_image)

            # Đảm bảo có landmarks (FReq031)
            if not results.face_landmarks:
                return self._invalid_result()

            landmarks = results.face_landmarks[0]
            ih, iw, _ = frame.shape

            def extract_pixel_coords(indices: List[int]) -> List[Tuple[int, int]]:
                return [
                    (int(landmarks[i].x * iw), int(landmarks[i].y * ih))
                    for i in indices
                ]

            return {
                "landmark_valid": True,
                "left_eye_points": extract_pixel_coords(self.LEFT_EYE_INDICES),
                "right_eye_points": extract_pixel_coords(self.RIGHT_EYE_INDICES),
                "mouth_points": extract_pixel_coords(self.MOUTH_INDICES),
            }

        except Exception:
            return self._invalid_result()

    def _invalid_result(self) -> Dict[str, Any]:
        return {
            "landmark_valid": False,
            "left_eye_points": None,
            "right_eye_points": None,
            "mouth_points": None,
        }

    def release(self):
        self.landmarker.close()

    def extract_landmarks(self, frame):
        res = self.process(frame)
        valid = bool(res.get('landmark_valid', False))
        return {
            'face_detected': valid,
            'landmark_valid': valid,
            'left_eye_landmarks': res.get('left_eye_points'),
            'right_eye_landmarks': res.get('right_eye_points'),
            'mouth_landmarks': res.get('mouth_points'),
            'left_eye_points': res.get('left_eye_points'),
            'right_eye_points': res.get('right_eye_points'),
            'mouth_points': res.get('mouth_points'),
        }
