# 🧪 AUTOMATED TEST CASE GENERATION INSTRUCTION

## 1. CODE UNDER TEST (Target)
* Target Function/Class: `src/{{path_to_module}}`

## 2. TESTING FRAMEWORK & TARGETS
* **Framework:** `pytest`
* **Goal:** Validate logic without requiring physical hardware (e.g., use mock BGR frames or dummy Pydantic objects).

## 3. REQUIRED TEST SCENARIOS
* **Happy Path:** Correct EAR/MAR evaluation when face and landmarks are valid.
* **Edge Cases / Guardrails:** 
  - `face_detected == False` results in `UNKNOWN` state.
  - Video EOF or broken stream gracefully exits.
  - Single blink does NOT trigger prolonged closure.
* **Data Contracts:** Ensure exceptions are raised if Pydantic model validation fails.