# TASK INSTRUCTION FOR CODEX AGENT: T0202 — D3S Dataset Inventory & Exploration

## 1. Role & Context
You are the **Lead Computer Vision & Data Engineer (Codex Agent)** on the **`PY_ADV_PJ_GR`** repository (Driver Drowsiness Detection System).
You are implementing **Task T0202** under **WP02 (Dataset Investigation & Preparation)**, assigned to **Member A**, satisfying requirement **FReq029** and Milestone **M02**.

## 2. Objectives
1. **Generate Dataset Inventory Script & CSV (`d3s_inventory.csv`):**
   - Scan video files located in `data/raw/d3s/`.
   - Use OpenCV (`cv2.VideoCapture`) to extract `resolution`, `fps`, `total_frames`, and `duration_sec`.
   - Parse and assign `subject_id` (3 subjects: `sub1`, `sub2`, `sub3`) and `label` (`eyeclose`, `yawning`, `neutral`, `happy`).
   - Export the structured data to `d3s_inventory.csv` (or `data/metadata/d3s_inventory.csv`).

2. **Author Exploratory Notebook (`notebooks/01_dataset_exploration.ipynb`):**
   - Build a fully executed Jupyter Notebook reading the inventory catalog.
   - Include visual distribution charts (class frequencies, duration histograms, subject balances).
   - Document subject-level splitting guidelines to prevent data leakage per `AGENTS.md`.

3. **Implement Automated Unit Test:**
   - Create `tests/unit/T0202_260929_2131_test_d3s_inventory.py` testing CSV schema validation, non-empty records, and valid frame rates/durations.

## 3. Invariants & Guardrails
- **Path Resolution:** Dynamically locate repository paths using `pathlib.Path(__file__).resolve().parents[...]` or relative project roots; never use hardcoded absolute machine paths.
- **Git Protection:** Ensure raw `.mp4` / `.avi` files are not staged for git commits; only commit the notebook, test, and CSV catalog.
- **Data Leakage Prohibition:** Enforce subject/video-level separation rules in the exploration analysis.
- **Coding Standard:** Strict PEP 8 compliance, explicit type annotations, clear docstrings, and zero linter errors.

## 4. Deliverables Checklist
- [ ] `d3s_inventory.csv`
- [ ] `notebooks/01_dataset_exploration.ipynb` (clean, executed top-to-bottom)
- [ ] `tests/unit/T0202_260929_2131_test_d3s_inventory.py`
- [ ] Terminal verification pass via `pytest tests/unit/T0202_260929_2131_test_d3s_inventory.py -v`.
