# T1002 — End-to-End Pipeline Integration & CLI Entrypoint

Date: October 18, 2026. Work package: WP10. Milestone: M05.

## Branch synchronization

- Fetched origin with pruning; initial tracked worktree was clean.
- Created `feat/T1002-End-to-End-Pipeline` from
  `origin/feat/T0902-Visualization` at `28e0467`.
- Verified ancestry includes T0704 (`5f1de3e`), T0801/T0802
  (`d452570`), and T0901 (`536cfcb`); no additional merge or conflict
  resolution was necessary. No commit, staging, or push was performed.
- Configuration changes already present in upstream history were inherited;
  this task does not modify configs, requirements, AGENTS.md, data, or models.

## Implementation and adapted contracts

- `src/pipeline/realtime_pipeline.py`: single-use runner for sequential video
  or camera acquisition, face detection, landmarks, typed EAR/MAR measurement,
  temporal analysis, decision, alert, and HUD rendering.
- Existing acquisition tuples are unpacked at their legacy boundary.
  Detector dictionaries and decision dictionaries have local TypedDict
  adapters; temporal measurements retain existing Pydantic v2 contracts.
  No shared schema or upstream return contract was replaced.
- The decision adapter explicitly maps
  `behavior.is_prolonged_closure` to `prolonged_eye_closure` and
  `behavior.yawn_frequency_in_window` to `yawn_frequency`. It passes
  `temporal.is_valid` as the decision input validity gate.
- `DecisionEngine(config=None)` now accepts the caller's loaded mapping;
  no-argument construction remains backward compatible. Temporal analysis,
  measurement, decisions, and alerts use the selected YAML configuration;
  CLI mode overrides are in-memory only.
- Missing faces and invalid geometry yield UNKNOWN eye, mouth, and decision
  states. Single blinks remain non-drowsy. Existing alert semantics preserve
  a prior warning during unknown observations; UNKNOWN is not reclassified
  as DROWSY, and an initially normal stream never warns from missing data.
- HUD receives current inference latency and a moving processing FPS over
  up to 30 previous frames. Snapshot/average latency includes HUD rendering
  but excludes acquisition, callbacks, and GUI. Average run FPS includes
  initialization, I/O, display if enabled, and teardown; these metrics must
  not be interpreted as equivalent performance measurements.
- Frozen frame/summary dataclasses expose measurements and counters through
  an optional streaming callback and JSON CLI summary without retaining all
  frames. Alert count means transitions into WARNING, not audio play count.
- EOF, frame limits, q/Q/ESC, Ctrl+C, stage errors, and partial initialization
  reach finally-based cleanup. All cleanup actions are attempted even if one
  fails. Temporal/alert state is reset and both model instances are closed.
- VideoReader now explicitly releases an unsuccessfully opened capture.
- `src/main.py`: argparse source/config/mode options, visual/headless aliases,
  positive frame limits, repository-relative defaults, error exit codes,
  and model-independent help. An ASCII help description avoids Windows
  redirected-console encoding failures; code docstrings remain Vietnamese.

## Baseline repairs and naming

The first unmodified full-suite attempt blocked in the upstream HUD test's
`waitKey(0)`. A diagnostic run excluding that one module passed 155 tests.
The HUD test now asserts rendering headlessly under pytest and retains its
interactive standalone demo. It is not skipped or removed.

The inherited source/test tree contained lint violations. Black formatting,
comment/docstring wrapping, unused-import removal, and explicit annotations
for intentional post-bootstrap imports restore the requested global lint
gate without disabling lint rules. Six revised legacy test files missing a
valid timestamp were renamed with their original task ID plus
`261018_0800`; their tests remain collected. Existing compliant artifact
filenames remain unchanged. The integration suite uses the required
`T1002_261018_0800_test_end_to_end.py` name; no noncompliant duplicate shim
was created. Run it by its timestamped path or run all of `tests/`.

## Verification

- `python -m pytest tests/ -v`: **191 passed in 11.07 seconds**, exit 0
  (156 upstream tests plus 35 T1002 integration cases; no skips).
- `python -m flake8 src/ tests/`: **zero findings**, exit 0.
- `python -m black --check --line-length 79 src/ tests/`: all 48 Python
  files unchanged. `python -m compileall -q src tests`: exit 0.
- The exact bare `pytest tests/ -v` and `flake8 src/ tests/` commands were
  attempted but their existing Windows executable launchers fail before
  running Python: they embed the former workspace path containing
  `Môn Ngành` rather than `specialized_subject`. Module invocation with
  the active environment's Python succeeds and is the verified equivalent.
  No dependency file or environment launcher was rewritten to hide this
  local environment issue. Use `python -m pytest` and `python -m flake8`
  here, or recreate the virtual environment separately.
- Integration coverage: both modes; closure/warning/cooldown/recovery;
  repeated yawns; single blink; missing/corrupt observations; missing face
  after warning; EOF; q/Q/ESC and Ctrl+C; stage/initialization/cleanup errors;
  frame limits; custom YAML; camera selection; empty stream; CLI aliases,
  invalid arguments/configuration, and help outside the repository.
- Both real D3S CLI subprocess tests process exactly 15 frames when assets
  exist. Both assets and video were present here; these tests did not skip.
- Manual terminal CLI `--help`: exit 0.
- Manual real-video Mode A: 15 frames, UNKNOWN 0, DROWSY 0, alerts 0,
  mean processing latency 44.09 ms, whole-run FPS 12.45, exit 0.
- Manual real-video Mode B: 15 frames, UNKNOWN 0, DROWSY 0, alerts 0,
  mean processing latency 35.05 ms, whole-run FPS 15.31, exit 0.
- MediaPipe emits native feedback-tensor/acceleration diagnostics on stderr;
  they did not prevent inference or successful exits.
- Black uses line length 79; flake8 checks the entire source/test tree.

## M05 status and limitations

T1002 functional integration and automated M05 acceptance checks are
implemented and passed with module-based quality gates. The exact bare
launcher gate remains an environment limitation documented above. Final
acceptance remains subject to normal peer review; no external milestone tracker
was modified. Sustained real-time throughput, live camera hardware, and
manual visual usability were not certified by short headless smoke runs.
The short-run whole-run FPS figures do not demonstrate a sustained 20 FPS
target. This remains a research/course prototype, not a certified driver
safety system. The inherited Alarm tracks activation state; this task does
not add an audible playback backend or WP11 experiment logging.