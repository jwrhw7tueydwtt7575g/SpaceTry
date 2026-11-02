# SpaceForge — Autonomous Spacecraft Rendezvous & Docking Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-84%20MVP%20%2B%20170%20ROS-brightgreen.svg)]()

> **Status** — `MVP_SUMMARY.md` is the source of truth for what works and what does not.
> The shipped PPO checkpoint docks in **100% of `run_eval`'s 50 seeds and 500/500 fresh seeds**
> (zero collisions), though it is slower and less fuel-efficient than the certified glideslope
> baseline, which docks **200/200 seeds**. The predictive runtime-assurance filter (v2) turns a
> reckless full-thrust slam into a soft capture. `ros2_ws/` holds the ROS 2 / Gazebo / perception /
> LLM integration (7 Python packages + a C++ Gazebo plugin); it builds and its 170 tests pass in a
> ROS 2 Humble + Gazebo Harmonic environment matching the `ros` CI job.

**SpaceForge** is a simulation-first testbed for autonomous spacecraft rendezvous, proximity operations,
and docking (RPO/Docking). It combines orbital relative dynamics, continuous reinforcement learning (PPO),
deterministic runtime-assurance safety interlocks, a certified classical guidance baseline, and a
fully CLI-driven mission orchestration layer into one unified pure-Python stack.

---

## Table of Contents
1. [What This Repository Contains](#1-what-this-repository-contains)
2. [Architecture & Component Breakdown](#2-architecture--component-breakdown)
3. [CLI Simulation — Quick Reference](#3-cli-simulation--quick-reference)
4. [Local Setup](#4-local-setup)
5. [Command Reference](#5-command-reference)
6. [Output Files](#6-output-files)
7. [Full Production Architecture (Blueprint)](#7-full-production-architecture-blueprint)
8. [Future Roadmap](#8-future-roadmap)

---

## 1. What This Repository Contains

```
spaceforgemvp/
├── pyproject.toml                 ← PEP 517 build config (pip install -e .)
├── requirements.txt               ← Pinned dependencies
├── checkpoints/
│   ├── ppo_spacecraft_final.zip  ← Pre-trained PPO checkpoint (docks; see metadata)
│   └── ppo_spacecraft_final.metadata.json ← Provenance, selection rule, measured results
└── spaceforge_mvp/
    ├── config.py                  ← PhysicsConfig, SafetyConfig, EnvConfig, MissionConfig
    ├── simulate.py                ← CLI mission orchestrator (main entry point)
    ├── core/
    │   └── cw_dynamics.py         ← Exact CW State Transition Matrix propagator
    ├── env/
    │   └── spacecraft_env.py      ← Gymnasium env — accepts injected MissionConfig
    ├── safety/
    │   └── mission_safety.py      ← Safety interlock v1 (zeroing veto)
    ├── control/
    │   ├── glideslope.py          ← Certified guidance law (200/200 dockings)
    │   └── rta.py                 ← Runtime Assurance v2 (predictive filter + certified backup)
    ├── training/
    │   └── train.py               ← SB3 PPO training pipeline
    ├── eval/
    │   ├── run_eval.py            ← Multi-episode tabular evaluator
    │   └── plot_trajectory.py     ← 2D trajectory plotter
    ├── viz/
    │   ├── record.py              ← Episode telemetry recorder (JSON traces)
    │   ├── scenarios.py           ← 5 benchmark scenario builder
    │   ├── render.py              ← 1920×1080 MP4 renderer
    │   ├── make_videos.py         ← Video compilation (3 acts + film)
    │   ├── dashboard.py           ← Standalone interactive HTML dashboard
    │   └── dashboard.html         ← Generated artifact (open in any browser)
    └── tests/                     ← 84 automated tests (all passing)
        ├── test_cw_dynamics.py
        ├── test_spacecraft_env.py
        ├── test_mission_safety.py
        ├── test_env_safety_integration.py
        ├── test_rta.py
        ├── test_mission_cli.py    ← NEW: CLI parsing, relative state, output files
        ├── test_train_smoke.py
        ├── test_eval_smoke.py
        └── test_integration_smoke.py
```

### Quick Folder & Directory Map

| Folder / File | What it contains | Simple Description |
|---|---|---|
| 📁 `core/` | [`cw_dynamics.py`](spaceforge_mvp/core/cw_dynamics.py) | **Physics Engine:** Math for orbital motion (Clohessy-Wiltshire relative dynamics & state matrix). |
| 📁 `env/` | [`spacecraft_env.py`](spaceforge_mvp/env/spacecraft_env.py) | **Gym Environment:** 10D sensor inputs, 3D thrusters, docking reward calculations. |
| 📁 `safety/` | [`mission_safety.py`](spaceforge_mvp/safety/mission_safety.py) | **Safety Interlock (v1):** Basic safety rules (fuel floor check & proximity speed veto). |
| 📁 `control/` | [`glideslope.py`](spaceforge_mvp/control/glideslope.py), [`rta.py`](spaceforge_mvp/control/rta.py) | **Controllers & Filters:** Certified Glideslope guidance law + v2 predictive runtime-assurance filter. |
| 📁 `training/` | [`train.py`](spaceforge_mvp/training/train.py) | **AI Training:** Script to train PPO reinforcement learning models with Stable-Baselines3. |
| 📁 `eval/` | [`run_eval.py`](spaceforge_mvp/eval/run_eval.py), [`plot_trajectory.py`](spaceforge_mvp/eval/plot_trajectory.py) | **Evaluation & 2D Plots:** Multi-episode metrics benchmark and trajectory PNG image generator. |
| 📁 `viz/` | `record.py`, `dashboard.py`, `render.py`, `make_videos.py` | **Visuals & Telemetry:** Generates interactive HTML dashboard and broadcast MP4 videos. |
| 📁 `tests/` | 15 test files | **Test Suite:** 84 automated unit, integration and regression tests (math, env, safety, CLI, viz, shipped policy). |
| 📁 `checkpoints/` | `ppo_spacecraft_final.zip` | **Model Weights:** Saved PPO reinforcement learning policy checkpoints. |
| 📁 `runs/` | `mission.json`, `telemetry.json`, `metrics.json` | **Output Results:** Saved telemetry logs and metrics from simulation runs. |
| 📄 `simulate.py` | CLI Entrypoint | **Main Simulator:** Single command to run missions with custom positions, fuel, and controllers. |
| 📄 `config.py` | Config Classes | **Constants & Configs:** Physical constants, reward weights, and `MissionConfig` data structures. |

---

## 2. Architecture & Component Breakdown

```mermaid
flowchart TD
    CLI["simulate.py CLI\nMissionConfig"]

    subgraph S_CORE ["Core Dynamics"]
        A["cw_dynamics.py\nExact CW STM Propagator"]
    end

    subgraph S_ENV ["Environment Layer"]
        B["spacecraft_env.py\nSpacecraftEnv — MissionConfig injected"]
    end

    subgraph S_SAFETY ["Safety Assurance Layer"]
        C1["mission_safety.py\nv1 — zeroing veto"]
        C2["rta.py\nv2 — predictive RTA"]
    end

    subgraph S_CTRL ["Controller Layer"]
        D1["glideslope.py\nCertified Guidance Law"]
        D2["PPO Checkpoint\nSB3 RL Policy"]
        D3["hot_approach\nAdversarial Stress Test"]
    end

    subgraph S_OUT ["Structured Outputs"]
        E1["mission.json\nResolved CLI inputs"]
        E2["telemetry.json\nPer-step trace"]
        E3["metrics.json\nSuccess / fuel / dv"]
        E4["dashboard.html\nInteractive replay"]
        E5["MP4 videos\n3 acts + film"]
    end

    CLI --> B
    A --> B
    C1 --> B
    C2 --> B
    D1 --> B
    D2 --> B
    D3 --> B
    B --> E1
    B --> E2
    B --> E3
    B --> E4
    B --> E5
```

### Module Details

| Module | Lines | What it does |
|---|---|---|
| `config.py` | 385 | Typed frozen configs: physics, actuator (0.1 m/s^2 per unit command), safety envelope, reward, reset, domain, faults, `MissionConfig`, `RunConfig` (JSON round trip). |
| `simulate.py` | 124 | CLI entry point. All mission values are required flags. Writes `mission.json`, `telemetry.json`, `metrics.json`. |
| `core/cw_dynamics.py` | 136 | Exact 6×6 CW STM: `X(k+1) = Φ(dt)·X(k) + Γ(dt)·u`. No numerical integrator; matches an independent RK4 to ~1e-11 m. |
| `env/spacecraft_env.py` | 415 | 10-D obs, 3-D continuous action, safety filter wired in, collision termination (swept capture-sphere test), fault/domain injection. |
| `safety/mission_safety.py` | 195 | Interlock v1 (geometric speed gate; only genuine braking passes) and the closing-speed envelope shared with v2. |
| `control/glideslope.py` | 116 | Exponential range decay + CW feed-forward at the mission's mean motion; dt- and authority-aware. 200/200 dockings, 0 interlock fires. |
| `control/rta.py` | 179 | Interlock v2: predictive envelope check; unsafe commands are replaced by the certified glideslope (Simplex) or a speed-sized brake. |
| `viz/record.py` | 274 | Rolls out an episode via `SpacecraftEnv`, honours every episode end, returns a JSON trace (pos, vel, cmd, applied, executed, fuel, overrides). |
| `viz/dashboard.py` | 509 | Self-contained HTML replay: play/pause, frame step, speed control, log-scale range chart; captions computed from the traces. |
| `viz/render.py` | 637 | Broadcast-grade MP4: HUD, thrust decomposition, range strip, title/result cards. |
| `training/train.py` | 245 | SB3 PPO pipeline. Accepts optional `MissionConfig` for fixed-state training runs. |
| `eval/validation_matrix.py` | 493 | Phase 6 matrix: 120 episodes × domain levels; pose error under pixel noise, LLM containment, curriculum, sim-to-sim vs independent RK4; JSON report. |
| `config.py: DomainConfig/FaultConfig` | — | Mass/inertia, thrust & camera/IMU noise, fuel spread, stuck/bias/dropout/freeze/leak faults. Identity by default. |

---

## 3. CLI Simulation — Quick Reference

Every mission value is supplied on the command line. Nothing is hardcoded.

```
CLI positions/velocities → Hill-frame (x=radial, y=in-track, z=cross-track)
relative_pos = start_pos − target_pos   (computed automatically)
Target is NOT required to be at origin.
```

**Controller choices:**

| `--controller` | What runs |
|---|---|
| `glideslope` | Certified deterministic guidance — always works |
| `rl` | Loaded PPO checkpoint — requires `--checkpoint` |
| `hot` | Reckless full-thrust slam — safety stress test |

**Safety mode choices:**

| `--safety` | Behaviour |
|---|---|
| `off` | No interlock — counterfactual baseline |
| `v1` | Veto only: inside 5 m above 0.2 m/s, any command that is not genuine braking is zeroed (the chaser coasts — can still collide) |
| `v2` | Predictive RTA: each step is checked against a closing-speed envelope (0.2 m/s at 5 m tapering to 0.05 m/s); unsafe commands are replaced by the certified glideslope or a speed-sized brake. Still brakes below the fuel floor. |

The proximity rule applies to every `--goal` by default; a goal label cannot switch it off.
`--max-accel` sets the actuator authority (m/s^2 per unit command, default 0.1).

---

## 4. Local Setup

```bash
git clone https://github.com/jwrhw7tueydwtt7575g/SpaceTry.git
cd SpaceTry

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Verify installation:

```bash
python3 -m pytest spaceforge_mvp/tests -q        # 84 passed
```

---

## 5. Command Reference

### Run a Mission Simulation

```bash
python3 -m spaceforge_mvp.simulate \
  --controller glideslope \
  --start-pos 12 -7 3.5 \
  --target-pos 2 1 0 \
  --start-vel 0 0 0 \
  --target-vel 0 0 0 \
  --fuel 0.8 \
  --duration 120 \
  --dt 1 \
  --mean-motion 0.00113 \
  --safety v2 \
  --seed 42 \
  --output-dir runs/my_mission
```

Prints metrics summary to console. Writes `mission.json`, `telemetry.json`, `metrics.json` to `runs/my_mission/`.

### Run RL Checkpoint

```bash
python3 -m spaceforge_mvp.simulate \
  --controller rl \
  --checkpoint checkpoints/ppo_spacecraft_final.zip \
  --start-pos 12 -7 3.5 --target-pos 0 0 0 \
  --start-vel 0 0 0 --target-vel 0 0 0 \
  --fuel 1.0 --duration 300 --dt 1 --mean-motion 0.00113 \
  --safety v1 --seed 42 \
  --output-dir runs/rl_run
```

### Train a New PPO Policy

```bash
# Smoke test (500 steps)
python3 -m spaceforge_mvp.training.train --total_timesteps 500

# Full training (200,000 steps)
python3 -m spaceforge_mvp.training.train --total_timesteps 200000 --seed 42
```

Training is always separate from simulation. `simulate` requires an existing checkpoint.

### Evaluate a Checkpoint (multi-episode table)

```bash
python3 -m spaceforge_mvp.eval.run_eval \
  --checkpoint checkpoints/ppo_spacecraft_final.zip \
  --episodes 50
```

### Generate the Interactive Dashboard

```bash
python3 -m spaceforge_mvp.viz.scenarios   # records 5 benchmark traces
python3 -m spaceforge_mvp.viz.dashboard   # builds dashboard.html
xdg-open spaceforge_mvp/viz/dashboard.html
```

### Render MP4 Videos (requires ffmpeg)

```bash
python3 -m spaceforge_mvp.viz.make_videos --film
# Writes: 01_docking.mp4, 02_rl_policy.mp4, 03_runtime_assurance.mp4, 00_presentation.mp4
```

---

## 6. Output Files

Every `simulate` run writes three files to `--output-dir`:

| File | Contents |
|---|---|
| `mission.json` | Resolved CLI inputs: start/target positions, relative state, fuel, dt, safety mode, seed |
| `telemetry.json` | Per-step frames: `t`, `pos`, `vel`, `cmd`, `applied`, `executed` (m/s^2 actually delivered), `override`, `dist`, `speed`, `fuel`, `dv` |
| `metrics.json` | Summary: `success`, `steps`, `start_dist`, `end_dist`, `end_speed`, `fuel_used`, `dv` (m/s), `overrides`, `terminal_reason` (docked / collision / fuel / out_of_bounds / timeout) |

---

## 7. Full Production Architecture (Blueprint)

The production system is a 5-layer robotics stack. The Phase-1 ROS 2 packages already
exist under `ros2_ws/src/` (gazebo, rl_env, rl_trainer, mission_ctrl, perception,
llm_planner, auto_trainer). They require ROS 2 Humble + Gazebo Harmonic to run live; the
pure-Python MVP remains the reference oracle. Launch headless with
`ros2 launch spaceforge_gazebo leo_orbit.launch.py headless:=true`.

```mermaid
flowchart TD
    subgraph S5 ["Layer 5: Mission Intelligence"]
        L5["llm_planner Node — LangChain Tool-Calling Agent"]
    end
    subgraph S_SAFE ["Safety Assurance Interlock"]
        LSA["mission_ctrl Node — Deterministic Runtime Assurance"]
    end
    subgraph S4 ["Layer 4: Decision Making"]
        L4["rl_env + rl_trainer — Gymnasium + PPO/SAC Policy"]
    end
    subgraph S3 ["Layer 3: Perception"]
        L3["perception Node — YOLOv8 + OpenCV solvePnP"]
    end
    subgraph S2 ["Layer 2: Middleware"]
        L2["sim_bridge Node — ros_gz Transport Bridge"]
    end
    subgraph S1 ["Layer 1: Simulation"]
        L1["Gazebo Harmonic — Orbital Dynamics & Sensor Plugins"]
    end
    L5 -->|Proposed Goal| LSA
    LSA -->|Validated Goal| L4
    L1 -->|Camera/LiDAR/IMU| L2
    L2 -->|Telemetry Topics| L3
    L3 -->|3D Target Pose| L4
    L4 -->|Thrust Vector| L2
    L2 -->|Gazebo Actuation| L1
```

---

## 8. Future Roadmap

| Phase | What | Status |
|---|---|---|
| **Done** | Physics engine, Gym env, CLI simulator, glideslope baseline, predictive v2 RTA, telemetry recorder, dashboard, videos | ✅ |
| **Done** | Independent RK4 reference for the CW STM; regression tests for every review finding | ✅ |
| **Done** | PPO that docks (realistic actuator authority, held-out checkpoint selection) | ✅ |
| **Done** | ROS 2 / Gazebo workspace builds; plugin + bridge + live integration test pass in the `ros` CI job | ✅ |
| **Next** | Stabilize RL training; beat the glideslope on time or fuel | ⬜ |
| **Next** | Train/validate the policy under domain randomization and faults | ⬜ |
| **Next** | Closed-loop docking run inside Gazebo | ⬜ |
| **Phase 1** | YOLOv8 + solvePnP perception (pose error 3.7 cm mean at 0.5 px keypoint noise) | 🟡 |
| **Phase 1** | LLM mission planner (+ offline default) behind the deterministic mission_ctrl boundary | 🟡 |
| **Phase 1** | Optuna HPO curriculum manager | 🟡 |

---

## License & Citation

Distributed under the **MIT License**.

```bibtex
@misc{spaceforge2026,
  author    = {Chaudhari, Vivek},
  title     = {SpaceForge: Autonomous Spacecraft Rendezvous \& Docking Testbed},
  year      = {2026},
  publisher = {GitHub},
  journal   = {GitHub Repository},
  howpublished = {\url{https://github.com/jwrhw7tueydwtt7575g/SpaceTry}}
}
```
