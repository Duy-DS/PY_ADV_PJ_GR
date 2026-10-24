# BUG FIX REPORT: T0201 Test Suite Errors & Dependency Resolution

- **Task ID:** T0201 (Setup & Download Dataset D3S)
- **Work Package:** WP02 (Dataset Preparation)
- **Author/Agent:** Codex Agent
- **Date:** 2026-09-28
- **Status:** Resolved

---

## 1. Problem Description & Root Cause Analysis

- **Symptom:** The reported initial outcome was `2 passed, 5 errors in
  1.72s` for `tests/unit/T0201_260928_1830_test_download_dataset.py`.
- **Local reproduction:** The requested bare `pytest` command used the global
  Python 3.11.1 interpreter and stopped during collection with
  `ModuleNotFoundError: No module named 'cv2'`. It collected zero tests because
  OpenCV was absent from that interpreter.
- **Environment mismatch:** The project `.venv` used Python 3.13.2 and already
  contained OpenCV. The same seven-test module passed there before any source
  or test changes.
- **Fixture analysis:** Two tests do not request `tmp_path`; the other five do,
  matching the shape of the reported `2 passed, 5 errors` result. After OpenCV
  was installed into the Python 3.11 environment, all five `tmp_path` tests
  passed without fixture or mock changes. This rules out a defect in
  `tmp_path`, `cv2.VideoCapture`, or the fake HTTP response contract in the
  current test module.
- **Root cause:** Dependencies had been declared without compatibility ranges,
  and the interpreter selected by bare `pytest` had not installed the declared
  `opencv-python` package. The downloader and test logic were not the cause.

## 2. Changes Made

- **Modified Source / Test Files:**
  - `requirements.txt`: Added compatible ranges for the T0201 runtime, test,
    formatting, and linting dependencies.
  - `tests/unit/T0201_260928_1830_test_download_dataset.py`: No change was
    necessary; existing fixtures and mocks pass under both verified Python
    environments when dependencies are installed.
  - `scripts/download_dataset.py`: No change was necessary; the downloader
    remained stable throughout this dependency repair.
- **Environment repair:** Installed `opencv-python==5.0.0.93` into the global
  Python 3.11.1 environment used by the bare `pytest` executable.

## 3. Dependency Modifications (requirements.txt)

| Library | Version / Spec | Rationale |
| :--- | :--- | :--- |
| `opencv-python` | `>=4.10,<6` | Provides `cv2`, MP4 fixture encoding, and video readability verification. |
| `numpy` | `>=2.0,<3` | Creates deterministic image arrays for generated MP4 test frames. |
| `pytest` | `>=8.3,<10` | Supplies the test runner, `tmp_path`, `monkeypatch`, and assertion support. |
| `flake8` | `>=7.1,<8` | Enforces the repository's zero-warning PEP 8 quality gate. |
| `black` | `>=24.10,<27` | Enforces the repository's formatter quality gate. |

No `gdown`, `requests`, `pytest-mock`, archive, or hash package was added.
T0201 uses standard-library `urllib` for network requests, `zipfile` for archive
handling, and Pytest's built-in `monkeypatch` fixture. Python also provides
`hashlib` if checksum verification is introduced later.

## 4. Verification & Validation Evidence

- **Execution Command:**
  `pytest tests/unit/T0201_260928_1830_test_download_dataset.py -v --tb=short`
- **Interpreter:** Python 3.11.1 with Pytest 9.1.1 and OpenCV 5.0.0.93.
- **Output Result:** `7 passed in 0.56s`.
- **Regression environment:** Python 3.13.2 in `.venv` also passed all seven
  tests.

Terminal result:

```text
collecting ... collected 7 items

test_default_download_name_matches_published_video PASSED
test_prepare_existing_zip_preserves_layout_and_verifies_video PASSED
test_extract_archive_rejects_path_traversal PASSED
test_verify_dataset_videos_fails_for_unreadable_video PASSED
test_skip_download_verifies_existing_dataset PASSED
test_google_drive_confirmation_form_uses_modern_download_url PASSED
test_download_retries_and_replaces_invalid_archive_atomically PASSED

7 passed in 0.56s
```

### Task T0201 Definition of Done (DoD) Checklist

- [ ] Full D3S video retained under `data/raw/d3s/`. The live source and MP4
  signature were validated previously, but this dependency-only repair did not
  store the approximately 373 MB raw asset.
- [x] Generated MP4 assets are structurally intact and decoded using OpenCV in
  the unit suite.
- [x] Raw video and archive extensions are excluded from Git tracking by the
  existing `*.mp4` and `*.zip` rules.

## 5. Guardrails & Invariants

- **Git Protection:** No raw video or archive was staged or committed. Existing
  `.gitignore` rules continue to protect `*.zip` and `*.mp4` files.
- **Path Resolution:** T0201 continues to derive repository and data locations
  with `pathlib.Path`; no absolute path was added to project code.
- **Code Standards:** The source and test files retain complete type hints and
  docstrings. Black and Flake8 checks complete without warnings.
- **Data Isolation:** Tests preserve subject/category paths and do not shuffle
  or split frames across subjects or videos.
