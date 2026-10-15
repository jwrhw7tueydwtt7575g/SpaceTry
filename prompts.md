# SpaceForge Codex/Luna Development Prompts

These prompts are an execution sequence for Codex/Luna/GPT-style coding agents. They are based on
`SpaceForge_End_to_End_Blueprint.pdf` and the current pure-Python MVP in this repository.

Run them in order. Each prompt is intentionally scoped to one milestone. The agent must inspect the
repository before editing, preserve existing user changes, avoid destructive commands, and finish
with tests plus a concise implementation report. Do not claim ROS 2, Gazebo, perception, or LLM
functionality until it is actually implemented and tested.

## Prompt 1 — Establish the engineering baseline

```text
You are Codex/Luna working in the SpaceForge repository. Read the current source tree, README,
MVP_SUMMARY.md, CRITIQUE.md, tests, and the End-to-End Blueprint PDF if available. Produce an
evidence-based implementation baseline: what exists, what is broken, what is duplicated, what the
blueprint requires, and what must remain out of scope for this milestone. Run the existing tests and
record the real result. Do not edit files yet. End with a decision-complete staged implementation
plan, risk list, and acceptance criteria for the next milestone.
```

## Prompt 2 — Make configuration and mission contracts explicit

```text
Implement the configuration foundation for SpaceForge. Keep physical model constants, safety
constraints, environment/reward parameters, and per-run mission inputs in separate typed structures.
Use the existing MissionConfig as the starting point. Validate vector dimensions, units, ranges,
duration/dt consistency, safety modes, and controller choices. Ensure CLI, Python API, telemetry,
training, evaluation, and visualization all consume the same resolved configuration. Remove mission-
critical literals from runtime code. Preserve backward compatibility where it does not undermine
explicit configuration. Add tests for validation, serialization, reproducibility, and nonzero target
coordinates. Run the full test suite before reporting completion.
```

## Prompt 3 — Harden the numerical simulation core

```text
Improve the current CW simulation without changing its public mission semantics. Add an independent
reference integration test (for example RK4 or a separately derived short-step propagator) against
the analytical CW STM, including forced and unforced motion. Audit units, timestep handling, target-
relative conversion, fuel accounting, docking termination, and safety-filter authority. Ensure the
recorder, Gymnasium environment, evaluator, and CLI use one simulation path. Add regression tests
for custom dt, custom mean motion, arbitrary target state, fuel depletion, and deterministic replay.
Do not add Gazebo or ROS in this prompt.
```

## Prompt 4 — Make RL training scientifically usable

```text
Audit and improve the PPO control setup using the current glideslope controller as the benchmark.
Redesign reward terms so progress, divergence, velocity, fuel, action effort, collision risk, and
docking success have clear units and documented weights. Add bounded initial-state sampling and
optional curriculum-friendly reset distributions. Add evaluation against fixed seeds and the
classical baseline, checkpoint metadata, TensorBoard metrics, and success/fuel/safety reports.
Retrain only when the code and tests are ready; never overwrite the shipped checkpoint without an
explicit output path. Define measurable acceptance thresholds and report whether the new policy
actually beats the baseline.
```

## Prompt 5 — Create the Gazebo foundation

```text
Begin Blueprint Phase 1. Inspect the host for ROS 2 Humble/Gazebo Harmonic availability before
adding dependencies. Create a separate, buildable Gazebo package with a minimal leo_orbit world,
chaser satellite, target satellite, and camera/IMU/LiDAR sensor placeholders. Implement the CW
orbital dynamics as a C++17 Gazebo system plugin only after defining the coordinate-frame, units,
reset, timestep, and command interfaces. Keep the existing Python simulator as a reference oracle.
Add launch files, package metadata, smoke tests, and a documented fallback when ROS/Gazebo is not
installed. Demonstrate that the world starts and ground-truth state is observable; do not pretend
that a launch file is validated if the required tools are unavailable.
```

## Prompt 6 — Implement the ROS/Gazebo bridge

```text
Implement the Blueprint sim_bridge node as the only component that translates Gazebo transport to
ROS 2 and ROS 2 commands back to Gazebo. Define and document all topic names, message types, frame
IDs, timestamps, rates, QoS, and units for clock, odometry, IMU, camera, LiDAR, and thrust/wrench
commands. Add launch parameters for scenario, rates, namespaces, and simulation time. Add isolated
topic publication tests and an integration smoke test that verifies timestamps and command flow.
Do not place policy, safety decisions, perception, or LLM logic inside sim_bridge.
```

## Prompt 7 — Build the live ROS Gymnasium control environment

```text
Implement the Blueprint rl_env package on top of ROS topics while preserving the pure-Python
SpacecraftEnv as a fast unit-test backend. First freeze the observation/action contract: target and
vehicle frames, ordering, units, normalization, 18-dimensional observation fields, and 6-dimensional
force/torque action limits. Add reset/step synchronization, episode timeout, simulator reset, stale
sensor handling, and safe termination behavior. Provide a mock transport adapter so tests run without
Gazebo. Prove that a deterministic classical controller can complete the Stage-1 1-D approach before
starting RL training.
```

## Prompt 8 — Add RL trainer and evaluation orchestration

```text
Implement the Blueprint rl_trainer workflow using Stable-Baselines3 PPO first, with SAC as an
explicitly selectable alternative. Add single and parallel environment modes, checkpoint/evaluation
callbacks, seed propagation, resume support, TensorBoard logging, and machine-readable reports.
Expose algorithm, environments, timesteps, output directory, scenario, and evaluation settings via
CLI/YAML without hidden constants. Compare every trained policy against the classical controller
and the safety-filtered controller. Add smoke tests using the mock environment and reserve live
Gazebo tests for environments that advertise the required dependencies.
```

## Prompt 9 — Implement perception with a testable pose contract

```text
Implement the Blueprint perception package as a ROS 2 node with a clean detector-to-pose interface.
Create camera calibration, target CAD keypoint, coordinate-frame, timestamp, confidence, and latency
contracts before coding. Support a deterministic mock detector and synthetic test images first, then
an optional YOLOv8 detector and OpenCV solvePnP backend. Publish target pose and detection/debris
messages with confidence and failure status. Measure pose error against ground truth and add tests
for valid detections, no detection, ambiguous PnP, bad calibration, stale images, and frame
transforms. Keep perception independent from policy and mission safety.
```

## Prompt 10 — Implement deterministic mission control and RTA

```text
Implement the Blueprint mission_ctrl package as a safety-critical deterministic boundary. Define a
typed goal schema for APPROACH, INSPECT, SAFE_HOLD, RETREAT, and ABORT, plus explicit fuel, distance,
closing-speed, actuator, keep-out, timeout, and stale-telemetry constraints. Reuse the proven Python
RTA logic where appropriate, but separate goal validation from action filtering. Invalid or unsafe
goals must produce a deterministic rejection reason and safe-hold/ braking result; no LLM or learned
policy may bypass it. Add exhaustive unit tests, property tests for safety invariants, ROS message
tests, and event telemetry for every override.
```

## Prompt 11 — Add the LLM mission planner safely

```text
Implement the Blueprint llm_planner as a strategic, tool-calling component only. Define a versioned
structured goal schema and tools for get_spacecraft_state, approach_target, inspect_surface,
return_to_safe_hold, and abort. The planner may propose semantic goals but must never publish direct
thrust or torque. Add provider-neutral adapters for OpenAI/compatible and offline mock backends,
timeouts, retries, token/cost limits, prompt/version metadata, and complete decision traces. Send
all proposals through mission_ctrl and test malformed outputs, contradictory goals, stale state,
provider failures, prompt-injection-like text, and emergency-hold priority. No network call may be
required for the default test suite.
```

## Prompt 12 — Build curriculum management and HPO

```text
Implement the Blueprint auto_trainer package. Encode the five curriculum stages explicitly:
1-D approach, 3-D static rendezvous, tumbling target, debris avoidance, and full inspection. Track a
rolling window of episode outcomes, enforce the documented advancement thresholds, persist curriculum
state, and make advancement reproducible. Add Optuna SQLite-backed hyperparameter search for PPO/SAC
with bounded trial budgets, resumability, artifact metadata, and objective metrics for success, fuel,
safety overrides, and collision risk. Test threshold boundaries, failed trials, resume behavior, and
stage rollback/hold policy before running expensive studies.
```

## Prompt 13 — Implement observability and operator tools

```text
Implement the Blueprint observability layer with a web_bridge/WebSocket service and a React/Three.js
dashboard only after defining a versioned telemetry schema. Stream odometry, target pose, commands,
applied commands, fuel, safety events, episode metrics, curriculum state, TensorBoard links, and
LLM tool traces with timestamps and frame IDs. Provide a replay mode backed by the existing JSON
telemetry so the dashboard can be tested without ROS or Gazebo. Add connection loss, stale data,
backpressure, privacy/secrets, and malformed-message handling. Keep the existing standalone dashboard
working while adding the live path.
```

## Prompt 14 — Harden, validate, document, and release

```text
Perform the final Blueprint Phase 6 hardening pass across the complete project. Add domain
randomization for mass/inertia and thruster noise, camera/IMU noise, configurable 1–10 ms communication
latency, randomized initial conditions, and fault-injection scenarios. Run the validation matrix:
over 100 evaluation episodes, docking success above the agreed threshold, fuel benchmark, pose error,
LLM goal validity, curriculum completion, safety invariants, and measured simulation-to-simulation
gap. Add CI for Python, ROS/colcon, C++, frontend, linting, and dependency-optional tests. Update
architecture docs, topic contracts, setup instructions, reproducible commands, known limitations,
and release artifacts. Report measured results only; clearly label unavailable hardware or simulator
checks as unverified.
```

## Agent execution rules

For every prompt above, Codex/Luna/GPT must:

1. Inspect relevant files and current git status before editing.
2. Preserve unrelated user changes and avoid destructive git/filesystem commands.
3. Prefer the smallest coherent implementation that satisfies the prompt.
4. Keep safety logic deterministic and outside learned/LLM components.
5. Add or update tests before claiming completion.
6. Run the narrowest relevant checks, then the full available test suite.
7. Report changed files, commands run, measured results, remaining limitations, and the next
   recommended prompt.
