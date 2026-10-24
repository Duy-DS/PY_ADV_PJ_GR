# TASK INSTRUCTION FOR CODEX AGENT: T0301 — Implement Input Acquisition Modules

## 1. Role & Context
You are the **Lead Computer Vision & Software Engineer (Codex Agent)** on the **`PY_ADV_PJ_GR`** repository (Driver Drowsiness Detection System).
You are implementing **Task T0301** under **WP03 (Input & Preprocessing)**, assigned to **Member A**, fulfilling requirements **FReq001**, **FReq002**, and **FReq032**.

## 2. Objectives
1. **Implement `src/acquisition/video_reader.py`:**
   - Create class `VideoReader(video_path: Union[str, Path])`.
   - Extract technical properties upon opening: `fps`, `width`, `height`, `total_frames`, and total `duration`.
   - Implement an iterable generator yielding `(frame_id: int, timestamp: float, frame: np.ndarray)`.
   - Compute `timestamp` accurately in seconds ($t = \text{frame\_id} / \text{fps}$ or via `cv2.CAP_PROP_POS_MSEC / 1000.0`).
   - Detect End-of-Stream (`ret == False` or empty frame) cleanly and implement a context manager (`__enter__` / `__exit__`) ensuring `cap.release()`.

2. **Implement `src/acquisition/camera.py`:**
   - Create class `CameraStream(device_id: int = 0, width: Optional[int] = None, height: Optional[int] = None)`.
   - Stream frames with monotonic timestamps derived from `time.perf_counter()`.
   - Implement clean resource release (`cap.release()`) upon exit, stream interruption, or camera disconnection.

3. **Author Comprehensive Unit Tests:**
   - Create `tests/unit/T0301_260930_0830_test_acquisition.py` adhering to the mandatory naming format `[Task ID]_[YYMMDD]_[HHMM]_[file name].py`.
   - Test normal playback, timestamp calculation accuracy, non-existent video handling, EOF behavior (`FReq032`), and context manager cleanup.
   - Use mock tests with synthetic video generation or mock `cv2.VideoCapture` to run deterministically in CI environments.

## 3. Invariants & Guardrails
- **Boundary Restriction:** Strictly I/O and frame acquisition. DO NOT include face detection, MediaPipe imports, EAR/MAR calculation, or drowsiness classification inside `src/acquisition/`.
- **Global Pipeline Schema:** The tuple yielded by both modules `(frame_id, timestamp, raw_frame)` must strictly match the data types `(int, float, np.ndarray)` as defined in `AGENTS.md`.
- **No Hardcoded Absolute Paths:** Use `pathlib.Path` for file handling and resolve paths relative to the repository root.
- **Clean Failure:** If a video fails to open, raise an actionable `FileNotFoundError` or `ValueError` rather than silently continuing or returning `None`.
- **Code Standards:** Ensure 100% PEP 8 compliance, provide type hints on all public methods/classes, write comprehensive docstrings, and ensure zero flake8/linter errors before finalizing.

## 4. Deliverables Checklist
- [ ] `src/acquisition/__init__.py` (Ensure module is importable)
- [ ] `src/acquisition/video_reader.py`
- [ ] `src/acquisition/camera.py`
- [ ] `tests/unit/T0301_260930_0830_test_acquisition.py`
- [ ] Verification command passing 100%: `pytest tests/unit/T0301_260930_0830_test_acquisition.py -v`
