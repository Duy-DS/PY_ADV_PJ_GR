# 💻 Coding Standards: Computer Vision & Python

## 1. Naming Conventions
* **Variables/Functions:** `snake_case`
* **Classes (Pydantic Models):** `PascalCase` (e.g., `DrowsinessDecision`)
* **Constants:** `UPPER_SNAKE_CASE`

## 2. Pipeline State Matrix & Pydantic Constraints
* **Type Hinting:** Mandatory across all module boundaries.
* **State Emission:** 
  - `face_detected == False` ➔ `UNKNOWN`
  - Valid Eye/Mouth ➔ `OPEN`/`CLOSED`
* **Strict Avoidance:** No loose tuples/dicts passed between `src/features` and `src/temporal`.

## 3. Testing & CI/CD Integrity
* **Unit Tests:** Placed in `tests/unit/`.
* **Linter:** `flake8` with 0 warnings.
* **Graceful Exit:** Always handle OpenCV streams safely via `cap.release()`.