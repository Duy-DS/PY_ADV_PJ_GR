# 🧠 AI Skill: {{SKILL_NAME}} (e.g., MediaPipe Landmark Extraction)

## 1. Core Skill Logic
* **Capability:** [What does this skill enable the AI to do? e.g., correctly parse 468 mesh landmarks].
* **Trigger Condition:** [When to apply this skill? e.g., when implementing `src/detection/`].

## 2. Associated Tools & Boundaries
* **Target Modules:** `src/features/eye.py`, `src/features/mouth.py`.
* **Required Params:** `raw_frame` (BGR), `configs/default.yaml` thresholds.

## 3. Few-shot Execution Examples
* *Input:* "Calculate EAR given MediaPipe eye indices."
* *Thought:* "I must ensure landmarks are valid. I will calculate Euclidean distance. If `face_detected` is false, I return `UNKNOWN`."
* *Action:* "Write EAR function compliant with `EyeMeasurement` schema."
* *Output:* "[Correctly formatted Pydantic compliant function]"