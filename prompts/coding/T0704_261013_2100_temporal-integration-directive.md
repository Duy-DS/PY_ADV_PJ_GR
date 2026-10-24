# T0704 — Codex implementation directive

Status: implementation blocked on user-managed configuration; not M04 sign-off.
Audit date: October 13, 2026, 21:00 local time.
Workspace: `f:\Study Materials\specialized_subject\PY NC\PY_ADV_PJ_GR`.
Scope: WP07 temporal integration and WP12 utilities/testing.

## 1. Authority and approval

Read the workspace AGENTS.md and any directory-specific instructions first.
The user approved shared typed contracts and necessary caller migration.
The user chose implementation AFTER manually adding missing YAML settings.
Two subsequent reads still showed the original ten-line configuration.
Do not interpret the user's confirmation as evidence that disk contents changed.
Do not edit configs/, requirements.txt, AGENTS.md, data/, or model weights.
Do not introduce dependencies, root folders, downstream inference, or commits.
If configuration is still incomplete, stop with an actionable error/report.

## 2. Verified local facts, not assumptions

- Eye features expose calculate_eye_state(left_eye_points, right_eye_points),
  returning ear and eye_state in a dictionary. Bilateral EAR is computed but
  individual EAR values are not exposed by that wrapper.
- Mouth features expose calculate_mouth_state(mouth_points, landmark_valid=True),
  returning mar and mouth_state in a dictionary. Invalid landmarks yield UNKNOWN.
  This module contains a pre-existing hard-coded MAR fallback; flag it separately.
- EyeClosureTracker() reads eye.closure_time_threshold using load_config().
  update(eye_state, current_time) returns a dictionary containing
  prolonged_eye_closure, blink_detected, closed_start_time,
  consecutive_closed_frames, and closure_duration.
  UNKNOWN resets the active closure. Reopening retains the completed event.
  There is no reset method, cumulative blink counter, or monotonic-time check.
- YawningTracker() reads mouth.yawn_duration_threshold and optionally
  mouth.yawn_time_window. Missing keys fall back to numeric algorithm constants.
  update(mouth_state, current_time) returns yawn_event, yawn_duration,
  yawn_count, and yawn_frequency. Count is cumulative; frequency is the number
  of event timestamps in an inclusive sliding window, not a rate per second.
  An event is emitted once per open cycle. Closing zeros reported duration.
  Its actual start attribute is open_start_time, not mouth_open_start.
- Existing TemporalAnalyzer(missing_frame_timeout=5) only implements
  process_tolerance(face_detected, current_features), reset_states(), and
  get_current_state(). It holds dictionaries and includes drowsiness_status.
  It has no timestamp, tracker integration, seconds timeout, or EOF reset API.
- scripts/test_detection_face_detected.py is a legacy GUI consumer: it uses
  missing_frame_timeout=15, process_tolerance(), result["action"], and
  non-contract TRACKING feature states. Migrate this consumer explicitly.
- No implemented shared Pydantic contracts were found. Pydantic v2 is installed
  and already declared in requirements.txt. Copy the documented field definitions
  exactly; do not invent additional fields in the prescribed three schemas.
- src/main.py, pipeline modules, decision modules, alert_manager.py, and
  visualization/overlay.py are empty. Do not claim a runnable full pipeline.
- docs/project/PROJECT_DOCUMENTATION.md, tests/integration/, tests/test_data/,
  and notebooks/03_yawning_analysis.ipynb are absent.
- data/raw/d3s/d3s_video_dataset.mp4 opens and its first frame decodes:
  1080 x 1920 BGR, FPS approximately 23.976. This verifies readability only,
  NOT subjects, labels, annotation provenance, or temporal accuracy.
- Baseline: python -m pytest tests/ -q returned 94 passed in 10.40 seconds.
- Baseline: python -m flake8 src/ tests/ --statistics --count reported
  67 violations. Do not silently suppress them or rename unrelated tests.

## 3. Required configuration gate

Resolve the default path through src/utils/config.py, not process working directory.
Read YAML before constructing trackers or changing files. Required entries:

```yaml
eye:
  ear_threshold: <user-selected finite non-negative value>
  closure_time_threshold: <existing finite non-negative seconds>
mouth:
  mar_threshold: <user-selected finite non-negative value>
  yawn_duration_threshold: <existing finite non-negative seconds>
  yawn_frequency_window: <user-selected positive finite seconds>
temporal:
  missing_face_tolerance_frames: <user-selected non-negative integer>
  missing_face_timeout_sec: <user-selected positive finite seconds>
```

This is a key specification, NOT executable YAML or approved numeric values.
Retain existing closure_time_threshold as the canonical eye duration key;
do not silently require closure_duration_threshold instead.
Reject missing sections/keys, booleans used as numbers, NaN/inf, bad types,
negative thresholds, and non-positive window/timeout with named-key errors.
Never provide numerical threshold fallbacks in algorithm modules.
Allow an explicit config_path for test/reproduction isolation if needed.
Pass the same validated configuration to both trackers; no mixed default paths.
Preserve legacy no-argument tracker behavior only where it remains compliant.

## 4. Architecture and typed public API

Use a shared module within an existing production package, for example
src/utils/contracts.py, containing EXACT AGENTS.md definitions for
EyeMeasurement, MouthMeasurement, and TemporalBehaviorState.
All new classes/functions/methods require Vietnamese docstrings with
Parameters, Returns, and Raises sections, even if the section states none.

Proposed public coordinator API:

```python
TemporalAnalyzer(config_path=None)
analyzer.update(
    face_detected: bool,
    timestamp: float,
    eye: EyeMeasurement,
    mouth: MouthMeasurement,
) -> TemporalAnalysisResult
analyzer.reset() -> None
```

TemporalAnalysisResult is a SEPARATE typed envelope containing current eye and
mouth measurements, behavior: TemporalBehaviorState, and gap metadata such as
action (CONTINUE/HOLD/RESET), is_valid, and consecutive_missing_frames.
Do not extend TemporalBehaviorState to carry UNKNOWN or a decision verdict.
Canonical mappings:

| Requested term | Documented behavior field |
| --- | --- |
| eye_closed_duration | eye_closed_duration |
| prolonged_eye_closure | is_prolonged_closure |
| blink_count | blink_count |
| yawn_event | is_yawning_event |
| yawn_count | yawn_count |
| yawn_duration | mouth_open_duration |
| window event count | yawn_frequency_in_window |

Expose aliases only as documented read-only properties if required; do not
change serialized schema fields. Do not include DROWSY/NON_DROWSY or confidence.
The downstream decision engine must inspect current validity independently.

Adapt legacy features explicitly: unpack their dictionary fields at the caller,
construct typed measurements, then pass only models to the coordinator.
Do not fabricate bilateral values from average EAR; calculate bilateral EAR
from the available points using existing calculate_ear(), or add an approved
typed feature adapter returning the actual bilateral observations.
Use UNKNOWN with is_valid=False for invalid measurements; non-negative numeric
placeholders are not valid evidence. Unknown measurement construction must not
allow NaN/inf into tracker calculations. Validate timestamp before mutation.

Tracker outputs crossing into the coordinator must be typed. Prefer explicit
typed update methods/results while keeping legacy dictionary-returning update()
wrappers for existing callers/tests. Do not simply annotate loose dictionaries
and claim compliance. Remove numerical fallbacks from the new execution path.
Add tracker reset APIs and use them rather than reaching into internals from
the coordinator. Preserve known event semantics unless tests justify a change.

## 5. Gap, event, and lifecycle policy

This is a reversible implementation proposal, not an existing project decision.

1. Reject non-finite or decreasing timestamps before any state mutation.
   Choose and document duplicate timestamp handling; never inflate duration.
2. Missing face, invalid dependent measurement, or corrupted landmarks exposes
   both current measurement states as UNKNOWN immediately. Never return a held
   OPEN/CLOSED state as a current observation. Apply conservative whole-frame
   invalidation for the combined coordinator and document that policy.
3. HOLD internal timers and counters only while missing count <= configured N
   AND elapsed time since last valid observation < configured timeout.
   Suppress prolonged-closure and yawn-event flags in every invalid output;
   preserve diagnostic counters/durations only with envelope is_valid=False.
4. Exclude the unobserved interval from eye and mouth active duration. Keep
   sliding-window timestamps on original video time so expired yawns age out.
   Implement an explicit suspension/resume mechanism in trackers, rather than
   rebasing the yawn window to a paused global clock.
5. Evaluate limits on every gap frame AND on recovery before updating trackers.
   A late valid frame cannot bridge an expired gap into a closure/yawn event.
   At elapsed == timeout reset; at missing count == N hold if time permits;
   count > N resets. Handle a first-ever invalid frame without prior evidence.
6. If no invalid frame is emitted for a decoder skip, there is insufficient
   metadata to identify that gap reliably. Document this limit; do not invent
   a maximum-valid-frame-interval threshold or reinterpret sparse samples.
7. A single isolated CLOSED sample followed by OPEN cannot become a prolonged
   closure solely from a long interval to the OPEN sample. Require corroborating
   closed observations for prolonged behavior, including zero duration threshold.
   Apply analogous corroboration to yawn events; emit once per open cycle.
8. Timeout resets ALL active timers, blink/yawn counters, and window buffers;
   retain only gap diagnostics needed for the current output. Full reset clears
   last timestamp and all metadata, allowing the next video to start at zero.
9. Do not replay completed events during HOLD/recovery. Distinguish cumulative
   yawn_count from the count within the sliding window, with documented endpoints.
10. Coordinator owns no captures/windows. At EOF/interruption the actual caller
    must use finally/context managers to reset the analyzer, release captures
    and detectors, flush any real logs, and destroy windows. Do not fabricate logs.

## 6. Required implementation artifacts and caller migration

- src/temporal/temporal_analyzer.py: complete coordinator, no placeholders.
- Shared models and minimal tracker adapters/reset/suspension changes within
  existing src/ packages. Maintain imports without cycles.
- src/utils/config.py: validate temporal configuration and preserve existing
  load_config() callers, without changing user-owned YAML.
- scripts/test_detection_face_detected.py: replace TRACKING placeholders with
  actual features, typed coordinator calls, and model attribute access.
  Remove hard-coded tolerance override; read configured values.
  Wrap initialization/execution cleanup safely, including partial failure.
- tests/unit/T0704_261013_2100_test_temporal.py, or use actual revision timestamp
  if implementation occurs later. Do not modify older uncompliant test names.
- notebooks/03_yawning_analysis.ipynb: synthetic measurements with timestamped
  duration/count/event/validity plots; explicit gap-policy explanation and no
  empirical dataset performance claims. Read the real config through load_config.
- Task-prefixed delivery/validation report under docs/reports/.

## 7. Deterministic verification suite

Use pytest parametrization, synthetic streams, and tmp_path YAML fixtures.
Temporary files are ephemeral fixtures, not tracked test_data artifacts.
Assert complete typed outputs, boundaries, event counts, and reset state.
No sleep, webcam, GUI, remote downloads, or dataset dependency for core tests.

Cover all of the following:

- OPEN/closed mouth baseline; short blink increments once and not prolonged.
- Sustained closure below/exactly/above YAML threshold; consecutive count.
- Single isolated CLOSED sample with delayed reopening cannot be prolonged.
- Zero configured duration cannot turn the first closed/open-mouth frame into
  a confirmed behavior; corroborating observations are still required.
- Brief speech-like opening does not yawn; sustained opening emits exactly once;
  closing and reopening permits a new event.
- Window expiry and endpoint behavior, cumulative vs. window count distinction.
- First-ever missing face and invalid feature independently return UNKNOWN.
- HOLD below/at frame tolerance preserves internal counters, emits no flags,
  and does not accumulate the unobserved interval.
- Frame count overflow, seconds timeout below/at/above boundary, and recovery
  after timeout without an additional invalid frame all reset correctly.
- Invalid EAR/MAR, UNKNOWN state, malformed/non-finite/decreasing timestamp;
  invalid timestamp does not partially mutate buffers.
- reset() idempotence, empty buffers, next sequence timestamp zero, no event leak.
- Actual feature calculations on synthetic geometry -> typed adapter -> both
  trackers -> typed envelope, with real bilateral EAR, not copied average values.
- Default YAML loading once configured, custom path loading, missing-key and
  bad-type errors, and changed YAML thresholds changing behavior.
- Mock caller EOF/exception cleanup without real windows or camera hardware.

Run from the workspace using the active environment:

```powershell
python -m pytest tests/ -v
python -m black --line-length 79 --check <absolute changed Python paths>
python -m flake8 <absolute changed Python paths>
python -m flake8 src/ tests/ --statistics --count
git diff --check
git status --short
```

Format changed Python files with black at line length 79 before checking.
Execute every notebook code cell in order using installed notebook tooling,
validate notebook JSON, and report any errors; do not only inspect cell text.
Re-read all edited files, inspect git diff, and ensure no protected files changed.
Run the readable local video only as an optional bounded input smoke test with
guaranteed release, not as fabricated accuracy validation.

## 8. Definition of done and reporting

Report actual commands, exit codes, pass counts, changed paths, configuration
values loaded, and residual baseline lint issues. All modified files must be
lint-clean. Do not claim repository-wide lint success if unrelated violations
remain. Do not expand scope to repair old unrelated tests without instruction.

FReq024 and FReq027 can be assessed against the prompt and implemented tests.
SCri004/008/009 authoritative definitions are not present locally; request the
actual criteria before claiming formal compliance. M04 cannot be signed off
by the coding agent alone. Separate code-ready, verified, blocked, and
team-approval statuses. Never claim implementation success for this directive.
