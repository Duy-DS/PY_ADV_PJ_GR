# 🎯 Task Prompt Template - Task ID: {{TASK_ID}}

## 1. PIPELINE CONTEXT (Context)
* Project Phase: [e.g., MVP Implementation / Evaluation]
* Target Module: `src/{{MODULE_NAME}}`
* Operating Mode: [e.g., Experiment A: `eye_only` / Experiment B: `eye_yawn`]

## 2. SPECIFIC OBJECTIVE (Objective)
* Action Required: [Detailed description of the CV/Python task]
* Input/Output Data Contracts: Must strictly use `{{PYDANTIC_SCHEMA}}` from Pydantic v2 definitions.

## 3. STRICT GUARDRAILS (Task Constraints)
* **Invalid Measurement Safety:** If `face_detected == False`, must emit `UNKNOWN` state immediately.
* **Config Decoupling:** NEVER hardcode thresholds (`ear_threshold`, `mar_threshold`, etc.). Always load via `src/utils/config.py`.
* **Dataset Integrity:** For D3S dataset, splits must remain strictly at the subject/video level (no random frame leakage).
* **Graceful Exit:** Always handle End-of-Input streams gracefully.

## 4. EXPECTED OUTPUT (Output Format)
* Code artifacts must adhere to PEP 8.
* Test files must be named `test_{{module_name}}.py` and placed in `tests/unit/`.
* Report artifacts (if any): `[YYMMDD]-[HHMM]-{{REPORT_NAME}}.md`.