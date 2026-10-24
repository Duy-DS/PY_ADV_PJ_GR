"""
Module xử lý ngoại tuyến để đánh giá hàng loạt ảnh tĩnh (Task T1105 & T1202).
File: src/pipeline/offline_pipeline.py
"""

import sys
import time
import cv2
import pandas as pd
from contextlib import ExitStack
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection.face_detector import FaceDetector
from src.detection.landmark_detector import LandmarkDetector
from src.features.measurement_adapter import measure_eye, measure_mouth
from src.temporal.temporal_analyzer import TemporalAnalyzer
from src.drowsiness.decision_engine import DecisionEngine
from src.utils.contracts import unknown_eye, unknown_mouth


class OfflinePipeline:
    def __init__(self, config_path=None, mode="eye_yawn"):
        self.face_detector = FaceDetector()
        self.landmark_detector = LandmarkDetector()
        self.temporal_analyzer = TemporalAnalyzer(config_path)

        self.decision_engine = DecisionEngine()
        self.decision_engine.mode = mode

    def process_image_sequence(self, image_paths: list) -> pd.DataFrame:
        """Chạy pipeline tuần tự trên một danh sách các đường dẫn ảnh tĩnh."""
        results = []

        with ExitStack() as cleanup:
            # Chỉ reset bộ đếm thời gian sau mỗi chuỗi ảnh, KHÔNG release detector ở đây
            cleanup.callback(self.temporal_analyzer.reset)

            for frame_id, img_path in enumerate(image_paths):
                # Giả lập timestamp (mỗi frame cách nhau ~33ms tương đương 30fps)
                timestamp = frame_id * (1.0 / 30.0)

                # Đọc ảnh tĩnh
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                t0 = time.perf_counter()

                # Face Detection
                detection = self.face_detector.process(frame)
                face_detected = bool(detection.get("face_detected", False))
                t1 = time.perf_counter()

                # Landmark & Features
                eye, mouth = unknown_eye(), unknown_mouth()
                if face_detected:
                    mesh = self.landmark_detector.process(frame)
                    if mesh.get("landmark_valid", False):
                        eye = measure_eye(
                            mesh["left_eye_points"],
                            mesh["right_eye_points"],
                            self.temporal_analyzer.settings,
                        )
                        mouth = measure_mouth(
                            mesh["mouth_points"], self.temporal_analyzer.settings
                        )
                t2 = time.perf_counter()

                # Temporal & Decision
                temporal_result = self.temporal_analyzer.update(
                    face_detected, timestamp, eye, mouth
                )
                decision = self.decision_engine.evaluate(temporal_result.model_dump())
                t3 = time.perf_counter()

                results.append(
                    {
                        "image_path": str(img_path),
                        "face_detected": face_detected,
                        "ear": eye.avg_ear,
                        "mar": mouth.mar,
                        "predicted_state": decision["state"],
                        "latency_face_ms": (t1 - t0) * 1000,
                        "latency_landmark_ms": (t2 - t1) * 1000,
                        "latency_decision_ms": (t3 - t2) * 1000,
                        "latency_total_ms": (t3 - t0) * 1000,
                    }
                )

        return pd.DataFrame(results)

    def release(self):
        """Giải phóng bộ nhớ mô hình AI một cách thủ công khi kết thúc toàn bộ."""
        self.face_detector.release()
        self.landmark_detector.release()
