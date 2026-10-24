# TASK PROMPT FOR CODEX AGENT: T0201 — Setup & Download Dataset D3S

## Context & Role
You are the **Lead Computer Vision & Data Engineer (Codex Agent)** working on the `PY_ADV_PJ_GR` repository.
You are implementing **Task T0201** under **WP02 (Dataset Preparation)**.

## Workspace Audit Status & Pre-requisites
- **Directory State:** The directory `scripts/` does not currently exist at the project root and must be created. The directory `data/raw/` exists but currently only contains an `R3S` folder; you must create `data/raw/d3s/`.
- **Git State:** `*.mp4` and `*.zip` are globally ignored in `.gitignore`, protecting against accidental large file commits.
- **Dependencies:** `opencv-python` is present in `requirements.txt`. Note that downloading utilities (like `requests` or `gdown`) are currently missing. If you require them, you must either add them to `requirements.txt` or use Python's built-in `urllib`.

## Objectives
1. Implement `scripts/download_dataset.py` to fetch, extract, and structure the D3S (Driver Drowsiness Dataset) video samples into `data/raw/d3s/`.
2. Implement verification logic using `opencv-python` to test reading extracted video files.
3. Create a unit test at `tests/unit/T0201_260928_1830_test_download_dataset.py` validating that downloaded assets meet the task's Definition of Done (DoD).

## Invariants & Guardrails
- **Subject & Data Isolation (Red Line):** Never shuffle or leak video frames across subjects; maintain subject/video isolation based on D3S categories.
- **No Hardcoded Absolute Paths:** Use `pathlib.Path` relative to the repository root (e.g., `pathlib.Path(__file__).resolve().parents[1]`).
- **Git Protection:** Ensure `data/raw/d3s/*.mp4` is never staged for Git commits (rely on existing `.gitignore` rules).
- **Fail Gracefully:** Log clear exceptions if network connections or extraction fails.
- **Coding Standard:** Strict PEP 8 compliance, full type hints, docstrings, and zero linter warnings.

## Deliverables Required
- [ ] Create `scripts/` directory and `scripts/download_dataset.py`.
- [ ] `tests/unit/T0201_260928_1830_test_download_dataset.py`
- [ ] Update `requirements.txt` if using third-party download libraries.
- [ ] Terminal verification command to execute the download and pass tests (`pytest tests/unit/ -v`).
