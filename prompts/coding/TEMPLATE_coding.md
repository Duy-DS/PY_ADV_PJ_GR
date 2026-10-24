# 💻 CODE GENERATION INSTRUCTION: {{MODULE_NAME}}

## 1. TECH STACK & ARCHITECTURE CONTEXT
* **Language:** Python 3.10+
* **Core Libraries:** OpenCV, MediaPipe, NumPy, Pydantic v2, PyYAML.
* **Architecture:** Directed Acyclic Graph (DAG) for processing video frames.

## 2. SPECIFIC FUNCTIONAL REQUIREMENTS
* Implement/Refactor: [Module logic, e.g., EAR calculation, Temporal state aggregation].
* Data Contracts: Ensure outputs conform to schemas like `EyeMeasurement`, `MouthMeasurement`, or `TemporalBehaviorState`.

## 3. STRICT CODING GUARDRAILS
* **Invalid States:** Always assign `UNKNOWN` if landmarks are invalid or `face_detected == False`.
* **Formatting:** Adhere to PEP 8 / Black.
* **Thresholds:** Load all operational thresholds via `src/utils/config.py`.
* **Resource Management:** Release `cv2.VideoCapture` and `cv2.destroyAllWindows()` cleanly.