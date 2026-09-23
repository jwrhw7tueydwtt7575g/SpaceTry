# SpaceForge MVP — source of truth

This is the single reference for this project: what it is, how it works, what shipped, what was
wrong with it, everything that has been added, and every number that gets quoted. Where a claim can
be checked, the command that checks it is written next to it.

Verified on this machine (Python 3.12, numpy, gymnasium, stable-baselines3, matplotlib, pytest).
If a number here disagrees with what you measure, the measurement wins — and this file needs fixing.

The original version of this document, as shipped by the author, is still in git:
`git show HEAD:MVP_SUMMARY.md`. It is worth reading once, because the contrast between what it
claimed and what the repository did is itself one of the findings.

---

## 0. Where to start

| If you want to... | Read |
|---|---|
| Understand the problem and the physics | §1, §2 |
| Understand the RL setup and the safety layer | §3 |
| Know what files exist and who wrote them | §4 |
| Know what was broken when this was handed over | §5 |
| Know what has been built since (including CLI orchestrator) | §6 |
| Have numbers you can quote | §7 |
| Actually run it | §8 |
| Know what is still unfinished | §9 |
| Look up a term | §10 |

Two companion documents: **CRITIQUE.md** is the engineering critique with the evidence behind every
finding in §5. **PRESENTATION.md** is the script for showing this to people, with what to say over
each clip. This file is the one that must stay correct.

---

## 1. What the project is

A spacecraft rendezvous and docking testbed, in pure Python, no GPU required.

Picture a small chaser spacecraft a few tens of metres from a target. It has to fly up to the
target, arrive *gently enough to latch onto it*, and it has to do so while obeying a safety layer
that can override whatever it wants to do. That is the entire problem, and it is a real one — the
same problem as a cargo ship approaching the ISS.

Why it is not as easy as "point at the target and thrust":

- There is no air, so there is no drag. Nothing slows you down for free. Every metre-per-second you
  gain has to be taken back off with thrust in the opposite direction.
- You are in orbit. "Sideways" and "up" are not fixed directions — the relative motion between two
  nearby orbits is a rotating, coupled system where a push in one axis produces drift in another.
- The contact has to be soft. In this project a dock only counts if the chaser is **within 0.5 m of
  the target and moving slower than 0.1 m/s** relative to it. That is one centimetre per second —
  about a slow blink. Arriving at 3 m/s instead is not a docking, it is a collision.

The project contains three ways to fly: a **reinforcement-learned policy** (the PPO AI that
ships with the repo), a **classical guidance law** (Glideslope baseline), and an **adversarial hot-approach**
stress test. And it contains a configurable **safety interlock** (`off`, `v1` zeroing veto, or `v2` active braking).

What the repository is *not*: it is not a robotics stack. There is no ROS 2, no Gazebo, no
perception, no camera, no 3D view. The README describes a five-layer production architecture with
ROS nodes, YOLOv8 perception and a LangChain planner, and labels that diagram "Conceptual Layer" —
none of it exists in this repo. What exists is ~3,700 lines of numpy physics, a Gymnasium environment, a
rule engine, a CLI simulation orchestrator, and a recording and rendering pipeline.

---

## 2. The physics, in plain terms

### 2.1 The frame

Everything is expressed in the **Hill frame** (also called the Clohessy-Wiltshire or LVLH frame),
which is a coordinate system that rides along with the target:

- `x` — **radial**: away from Earth, i.e. "up" from the target. Positive `x` is further out.
- `y` — **in-track**: along the direction of orbital travel (the "forward" direction).
- `z` — **cross-track**: out of the orbital plane.

Under the CLI orchestration layer, users can specify arbitrary world Hill-frame coordinates for both
the chaser and the target; the simulation automatically subtracts the target state to form relative coordinates:
`relative_pos = start_pos - target_pos` and `relative_vel = start_vel - target_vel`.

Mean motion `n = 0.00113 rad/s` (`config.py`), which corresponds to a low Earth orbit with a period
of about 92 minutes. The simulation step is `dt = 1 s` (configurable via CLI), and an episode lasts at most 300 steps
(300 seconds, configurable via CLI).

### 2.2 The equations

The free-motion (no thrust) Clohessy-Wiltshire equations of relative motion are:

```
ax = 3n²x  + 2n·vy
ay = -2n·vx
az = -n²z
```

Read the first line carefully, because it is the whole reason naive pursuit guidance fails: an
object sitting *radially above* the target is pushed *away* from it (`+3n²x`), and an object moving
*in-track* is pushed *radially* (`+2n·vy`). Thrust along the line of sight therefore does not
produce motion along the line of sight. That coupling is what makes a chaser spiral instead of
approach, and it is why the guidance law in §6.2 cancels it in feed-forward.

For a starting point 18 m out, doing nothing at all over 300 s leaves you 20.2 m away and drifting —
measured, §7.4.

The code applies this exactly: `core/cw_dynamics.py` builds the analytic **state transition matrix**
(STM) that maps `(position, velocity)` from step *k* to step *k+1* under a held acceleration, and
uses that directly. There is no numerical integrator in the loop — the propagation is the closed-form
matrix.

### 2.3 The state vector

10 numbers, all normalised, in this order (`env/spacecraft_env.py`):

| # | Meaning | Normalisation |
|---|---|---|
| 1-3 | relative position `x, y, z` (m) | divided by 20.0 |
| 4-6 | relative velocity `vx, vy, vz` (m/s) | divided by 2.0 |
| 7 | fuel fraction remaining | 1.0 → 0.0 |
| 8 | elapsed fraction of the episode | 0.0 → 1.0 over max_steps |
| 9-10 | unit vector pointing at the target (2D bearing) | already a unit vector |

The action is 3 continuous numbers `[ax, ay, az]` in m/s², clipped to ±1.0.

---

## 3. The environment

`env/spacecraft_env.py` is a standard Gymnasium environment that accepts an injected `MissionConfig`.
One `step(action)` does:

1. Hand the commanded action to the active **safety interlock** (`off`, `v1`, or `v2`) and take back what the interlock allows.
2. Propagate the physics with the *allowed* acceleration using the exact CW STM.
3. Debit fuel by the **applied** (post-interlock) magnitude: `0.001 × Σ|applied|` per step off the tank.
4. Compute the reward, and decide whether the episode ended.

**Reward** (from `config.py`):

| Term | Value | Notes |
|---|---|---|
| Radial progress | `+5.0 × (d_prev − d_now)` | pays for closing distance |
| Speed penalty | `−0.5 × speed` | every step |
| Effort penalty | `−0.01 × Σ\|a_i\|` | sum of absolute *applied* accelerations |
| Docking jackpot | `+100` | on a successful capture |

**Termination.** Success (distance < 0.5 m AND speed < 0.1 m/s) ends the episode with +100. Fuel
below zero truncates without success. Running out of the step budget truncates without success.

**The safety interlocks:**
- **v1 (Shipped zeroing rule engine)**: Fuel floor (<5% -> emergency hold); inside 5m if speed > 0.2 m/s, zero the command.
- **v2 (Runtime Assurance Active Braking)**: Inside 5m if speed > 0.2 m/s, replaces command with maximum braking opposing velocity vector ($u \cdot v \le 0$).
- **off**: Direct feedthrough for adversarial/counterfactual benchmarking.

---

## 4. The repository, file by file

`spaceforge_mvp/` totals **~3,700 Python lines**:

| Path | Lines | Origin | What it does |
|---|---|---|---|
| `config.py` | 82 | updated | Frozen dataclasses + `MissionConfig` (CLI parameters, derived relative coordinates) |
| `simulate.py` | 120 | **added** | CLI simulation orchestrator: loads controllers, runs missions, writes structured JSON outputs |
| `core/cw_dynamics.py` | 136 | shipped | CW acceleration + exact discrete STM + `cw_step` |
| `env/spacecraft_env.py` | 247 | **added/updated** | Gymnasium env: accepts `MissionConfig`, supports `off`/`v1`/`v2` safety modes |
| `safety/mission_safety.py` | 90 | shipped | The v1 interlock: three rules, returns a verdict dict |
| `training/train.py` | 206 | updated | Stable-Baselines3 PPO training entry point with `MissionConfig` support |
| `eval/run_eval.py` | 149 | shipped | Runs a checkpoint over N episodes, prints a metrics table |
| `eval/plot_trajectory.py` | 168 | shipped | 2D `x`-`y` trajectory plot to `trajectory.png` |
| `control/glideslope.py` | 87 | **added** | Certified guidance law + adversarial hot approach |
| `control/rta.py` | 97 | **added** | Interlock v2: substitute braking instead of zeroing |
| `viz/record.py` | 220 | updated | Rolls out an episode via `SpacecraftEnv`, writes telemetry trace |
| `viz/render.py` | 638 | **added** | Broadcast-style MP4 renderer (HUD, strip, cards, fades) |
| `viz/make_videos.py` | 247 | **added** | Builds the three acts and stitches the film |
| `viz/dashboard.py` | 497 | **added** | Emits self-contained interactive HTML replay |
| `viz/scenarios.py` | 89 | **added** | Defines and records the five benchmark scenarios |
| `viz/preview.py` | 59 | **added** | Renders single frames for inspection |
| `tests/` (9 files) | 658 | mixed | 28 tests total (20 shipped, 6 RTA, 2 CLI mission tests in `test_mission_cli.py`) |

### Quick Directory & Folder Summary

| Directory / File | Purpose & Contents | Simple Explanation |
|---|---|---|
| 📁 `core/` | [`cw_dynamics.py`](file:///home/vivek/Desktop/SpaceForge/spaceforgemvp/spaceforge_mvp/core/cw_dynamics.py) | **Physics Engine:** Pure orbital mechanics math (Clohessy-Wiltshire state transition matrix). |
| 📁 `env/` | [`spacecraft_env.py`](file:///home/vivek/Desktop/SpaceForge/spaceforgemvp/spaceforge_mvp/env/spacecraft_env.py) | **Gym Environment:** Spacecraft step/reset, 10D observations, fuel tracking, and docking rewards. |
| 📁 `safety/` | [`mission_safety.py`](file:///home/vivek/Desktop/SpaceForge/spaceforgemvp/spaceforge_mvp/safety/mission_safety.py) | **Safety Layer (v1):** Rule-based safety guard that zeroes thrust if too fast near target. |
| 📁 `control/` | [`glideslope.py`](file:///home/vivek/Desktop/SpaceForge/spaceforgemvp/spaceforge_mvp/control/glideslope.py), [`rta.py`](file:///home/vivek/Desktop/SpaceForge/spaceforgemvp/spaceforge_mvp/control/rta.py) | **Controllers:** 100% working Glideslope guidance baseline + v2 active-braking safety filter. |
| 📁 `training/` | [`train.py`](file:///home/vivek/Desktop/SpaceForge/spaceforgemvp/spaceforge_mvp/training/train.py) | **RL Training:** Stable-Baselines3 PPO training loop with checkpointing. |
| 📁 `eval/` | [`run_eval.py`](file:///home/vivek/Desktop/SpaceForge/spaceforgemvp/spaceforge_mvp/eval/run_eval.py), [`plot_trajectory.py`](file:///home/vivek/Desktop/SpaceForge/spaceforgemvp/spaceforge_mvp/eval/plot_trajectory.py) | **Evaluation:** Runs multi-episode benchmark tests and generates 2D trajectory plots (`.png`). |
| 📁 `viz/` | `record.py`, `dashboard.py`, `render.py`, `make_videos.py` | **Visuals & Telemetry:** Records JSON traces, builds interactive HTML dashboard, and renders MP4s. |
| 📁 `tests/` | 9 test files | **Automated Tests:** 28 unit and integration tests (`pytest`). |
| 📁 `checkpoints/` | `ppo_spacecraft_final.zip` | **Model Files:** Saved neural network weights from training runs. |
| 📁 `runs/` | `mission.json`, `telemetry.json`, `metrics.json` | **Outputs:** Output logs from CLI simulation runs. |
| 📄 `simulate.py` | Main CLI Simulator | Single command to simulate missions with custom positions, velocities, and fuel. |
| 📄 `config.py` | Configurations | Centralized constants, dataclasses, and `MissionConfig`. |

---

## 5. What shipped, and what was wrong with it

The author delivered: six modules, one trained PPO checkpoint
(`checkpoints/ppo_spacecraft_final.zip`), twenty tests, and a README describing a five-layer
production architecture. The physics module is genuinely correct. Everything downstream of it had
problems:

### 5.1 The environment file was never committed
`.gitignore` had a bare `env/` rule ignoring `spaceforge_mvp/env/`. Fixed by reconstructing `env/spacecraft_env.py` and anchoring ignore rules.

### 5.2 The shipped policy never docks (0% success in 50 episodes)
The reward paid for radial progress but never penalized flying away. The policy learns to fly away: ends 3.72 km out at 31.2 m/s.

### 5.3 The policy is worse than doing nothing
Zero thrust: drifts from 18.2 m to 20.2 m over 300 s. Trained policy: ends 3,720 m away.

### 5.4 Fuel floor rule unreachable in 300 steps
Max possible burn rate leaves 10% tank at step 300; fuel floor is 5%.

### 5.5 Interlock v1 zeroing veto is not a brake
Denying thrust inside 5m allows craft to coast through target at 2.0 m/s. v2 active braking stops it at 0.01 m/s and docks in 8 s.

### 5.6 Interlock authority starts at 5 m
RL policy failure happens outside 5 m, so safety interlock never triggered.

### 5.7 CW dynamics tests were tautologies
Asserted `cw_step` against `Phi @ x + Gamma @ u` produced by same module.

---

## 6. What has been added

### 6.1 The environment module — `env/spacecraft_env.py`
Full Gymnasium implementation matching tests and documentation. Now accepts an injected `MissionConfig` and selectable `safety_mode` (`off`, `v1`, `v2`).

### 6.2 Certified guidance law — `control/glideslope.py`
Proportional speed decay with CW feed-forward cancellation. Docks **200/200 seeds**, 0 interlock fires.

### 6.3 Interlock v2 Runtime Assurance — `control/rta.py`
Substitutes maximum opposing braking when speed gate is exceeded.

### 6.4 CLI Simulation Orchestrator — `simulate.py` & `config.py` (NEW)
- Provides typed `MissionConfig` object.
- Accepts all mission inputs via CLI (`--start-pos`, `--target-pos`, `--start-vel`, `--target-vel`, `--fuel`, `--duration`, `--dt`, `--mean-motion`, `--safety`, `--seed`, `--output-dir`).
- Automatically computes relative coordinates.
- Outputs structured artifacts: `mission.json`, `telemetry.json`, `metrics.json`.

### 6.5 Telemetry Recorder, Videos & Dashboard
- `viz/record.py`: Unified episode recorder delegating to `SpacecraftEnv`.
- `viz/render.py` & `viz/make_videos.py`: 1080p MP4 renderer for 3 acts and stitched presentation film.
- `viz/dashboard.py`: Standalone HTML replay widget (`dashboard.html`).

### 6.6 Full Test Suite (28 Tests)
28 tests passing (`pytest -q`): CW dynamics, environment mechanics, safety interlocks, RTA active braking, integration smoke, and CLI mission parsing.

---

## 7. Measured results

### 7.1 The five recorded scenarios (`viz/scenarios/*.json`)

| Scenario | Safety | Seed | Steps | Start range | Closest approach | Speed there | Δv (m/s) | Interlock fires | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| `docking` (glideslope) | v1 | 42 | 97 | 14.33 m | 0.4945 m | 0.0173 m/s | 1.397 | 0 | **docked** |
| `rl_policy` (shipped PPO) | v1 | 42 | 300 | 18.22 m | 5.2681 m | 1.0234 m/s | 53.825 | 0 | timeout (3.72 km out) |
| `hot_open` (reckless) | off | 7 | 60 | 6.00 m | 0.0135 m | 6.4799 m/s | 61.491 | 0 | flew through target |
| `hot_v1` (zeroing veto) | v1 | 7 | 60 | 6.00 m | 0.0208 m | 1.9999 m/s | 53.008 | 17 | collision (coasted) |
| `hot_v2` (braking veto) | v2 | 7 | 8 | 6.00 m | 0.0033 m | 0.0100 m/s | 8.034 | 4 | **docked in 8 s** |

### 7.2 The certified guidance law over 200 seeds
- **200 / 200 dockings** (100% success rate)
- Mean steps: 106.3 (49–132)
- 0 interlock overrides
- Mean Δv: 1.418 m/s

---

## 8. Run everything

From the repository root:

```bash
# 0. Activate virtual environment
source .venv/bin/activate

# 1. Run full test suite (28 tests)
python3 -m pytest -q

# 2. Run CLI Mission Simulation
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
  --output-dir runs/demo

# 3. Record scenario traces & generate interactive HTML dashboard
python3 -m spaceforge_mvp.viz.scenarios
python3 -m spaceforge_mvp.viz.dashboard
xdg-open spaceforge_mvp/viz/dashboard.html

# 4. Render MP4 videos (requires ffmpeg)
python3 -m spaceforge_mvp.viz.make_videos --film

# 5. Evaluate shipped PPO policy
python3 -m spaceforge_mvp.eval.run_eval --checkpoint checkpoints/ppo_spacecraft_final.zip --episodes 50

# 6. Train a new PPO policy
python3 -m spaceforge_mvp.training.train --total_timesteps 500  # smoke test
```

---

## 9. What is still open

1. **Reward redesign & retraining**: Bound episode exploration, add boundary escape penalty, align effort penalty with fuel.
2. **Independent physics test**: Add RK4 numerical integration test against analytical CW STM matrix.
3. **Automated CI**: Add GitHub Actions workflow.
4. **Future robotics layers**: Gazebo/ROS 2 integration and YOLOv8 perception remain future milestones.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Chaser / Target** | Spacecraft performing docking, and the target object it approaches. |
| **Hill / LVLH Frame** | Coordinate system centered on target: `x` (radial), `y` (in-track), `z` (cross-track). |
| **CW STM** | Closed-form State Transition Matrix propagating orbital state analytically. |
| **Interlock v1** | Shipped rule engine that zeroes thrust commands upon proximity violations. |
| **Interlock v2 (RTA)** | Runtime Assurance filter that substitutes active opposing braking thrust. |
| **Glideslope** | Exponential range decay guidance law with CW drift feed-forward cancellation. |
