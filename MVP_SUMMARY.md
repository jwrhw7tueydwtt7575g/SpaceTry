# SpaceForge MVP — source of truth

This is the single reference for this project: what it is, how it works, what shipped, what was
wrong with it, everything that has been added, and every number that gets quoted. Where a claim can
be checked, the command that checks it is written next to it.

Verified on this machine (Python 3.12, numpy, gymnasium, stable-baselines3, matplotlib, pytest).
If a number here disagrees with what you measure, the measurement wins — and this file needs fixing.

The original version of this document, as shipped by the author, is still in git:
`git show HEAD:MVP_SUMMARY.md`. It is worth reading once, because the contrast between what it
claimed and what the repository did is itself one of the findings.

> **Status (fix/review-findings).** The shipped PPO checkpoint now docks: 100% on `run_eval`'s
> 50 seeds and 500/500 fresh seeds, zero collisions (§7.3). The runtime-assurance filter is
> predictive and no longer lets a coasting, sideways-thrusting or mislabelled vehicle through
> (§5.8). The ROS 2 / Gazebo stack builds and its tests run, including a live launch -> Gazebo ->
> bridge round trip, in a ROS 2 Humble + Gazebo Harmonic container that mirrors the `ros` CI job.

---

## 0. Where to start

| If you want to... | Read |
|---|---|
| Understand the problem and the physics | §1, §2 |
| Understand the RL setup and the safety layer | §3 |
| Know what files exist and who wrote them | §4 |
| Know what was broken when this was handed over | §5 (§5.8: second review) |
| Know what has been built since (including CLI orchestrator) | §6 |
| Have numbers you can quote | §7 |
| Actually run it | §8 |
| Know what is still unfinished | §9 |
| See Phase 6 hardening + measured validation numbers | §10 |
| Look up a term | §11 |

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

What the repository is *not*: it is not a validated robotics stack. `ros2_ws/` contains a
Phase-1 ROS 2 / Gazebo / perception / planner integration (7 contract-first packages),
but it needs ROS 2 Humble + Gazebo Harmonic and is unvalidated on a plain Python host.
What is validated is ~4,300 lines of numpy physics, a Gymnasium environment, a
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

**Reward** (from `config.py` + `env/spacecraft_env.py` — full breakdown in `info["reward_breakdown"]`):

| Term | Value | Notes |
|---|---|---|
| Radial progress | `+5.0 × (d_prev − d_now)` | pays for closing distance |
| Divergence penalty | `−1.0 × max(0, d_now − d_prev)` | flying away costs extra (added post-critique) |
| Speed penalty | `−0.5 × speed` | every step |
| Effort penalty | `−0.01 × Σ\|a_i\|` | sum of absolute *applied* accelerations |
| Fuel penalty | `−0.1 × fuel_used_this_step` | ties spend to tank |
| Collision-risk penalty | `−5.0 × max(0, (speed−0.2)/0.2)` inside 5 m | dense slow-down signal in bubble |
| Docking jackpot | `+100` | on a successful capture |

**Termination.** Success (distance < 0.5 m AND speed < 0.1 m/s) ends the episode with +100. Fuel
below zero truncates without success. Running out of the step budget truncates without success.
Leaving the 100 m work volume (`work_volume_bound`) also truncates — bounds the fly-away failure.
Note the 1 Hz sampler can step over the 0.5 m bubble at m/s speeds; use
`viz/record.py:replay_continuous_min` (10 ms replay) for ground-truth closest approach.

**The safety interlocks:**
- **v1**: Fuel floor (<5% -> emergency hold); inside 5 m if speed > 0.2 m/s, braking thrust (`u·v ≤ 0`) is accepted, accelerating thrust is zeroed. Legacy scope is `APPROACH`-only; set `SafetyConfig(enforce_proximity_for_all_goals=True)` for geometry-only enforcement so no goal label can bypass it.
- **v2 (Runtime Assurance Active Braking)**: Same veto rules; on veto substitutes maximum braking opposing velocity vector ($u \cdot v \le 0$) instead of zeroing.
- **off**: Direct feedthrough for adversarial/counterfactual benchmarking.
- Known limit: at default `fuel_rate_scale=0.001` the 5% floor is unreachable in 300 steps (max burn 0.9); raise the rate or lengthen the budget if the floor must be reachable, or configure a custom `ActuatorConfig`.

---

## 4. The repository, file by file

`spaceforge_mvp/` totals **~4,300 Python lines**:

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
| `tests/` (15 files) | ~1400 | mixed | 84 tests total (see `pytest --collect-only`) |

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
| 📁 `tests/` | 12 test files | **Automated Tests:** 40 unit and integration tests (`pytest spaceforge_mvp/tests`). |
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

### 5.2 The original shipped policy never docked (0% success in 50 episodes) -- fixed, §7.3
The reward paid for radial progress but never penalized flying away. The policy learns to fly away: ends 3.72 km out at 31.2 m/s.

### 5.3 The policy is worse than doing nothing
Zero thrust: drifts from 18.2 m to 20.2 m over 300 s. Trained policy: ends 3,720 m away.

### 5.4 Fuel floor rule unreachable in 300 steps
Max possible burn rate leaves 10% tank at step 300; fuel floor is 5%.

### 5.5 Interlock v1 zeroing veto is not a brake
Denying thrust inside 5m allows craft to coast through target at 2.0 m/s (original 1 m/s^2 authority). The original v2 "maximum braking" fix had its own holes -- see §5.8; the current predictive filter is described in §6.3.

### 5.6 Interlock authority starts at 5 m
RL policy failure happens outside 5 m, so safety interlock never triggered.

### 5.7 CW dynamics tests were tautologies
Asserted `cw_step` against `Phi @ x + Gamma @ u` produced by same module.

### 5.8 Second review: what the hardening pass still got wrong (fixed on this branch)
Each item has a regression test (`tests/test_review_regressions.py`, `tests/test_rta.py`,
`mission_ctrl/test/test_safety_boundary.py`, `llm_planner/test/test_planner_boundary.py`).

| Defect | Effect before the fix |
|---|---|
| Interlock counted any `u . v <= 0` as braking | A zero command (coasting) passed: a vehicle closing at 0.6 m/s from 3 m passed 2 cm from the target with 0 overrides under v2 |
| Same rule, lateral thrust | `(0.01, 1, 1)` inside 5 m was "braking" and raised speed 0.3 -> 1.44 m/s |
| v2 brake was always full authority | Reversed slow vehicles and raised speed (0.25 -> 0.75 m/s) |
| Proximity gate scoped by goal label | `--goal DOCK` disabled it: the reckless slam hit at 3.46 m/s with 0 overrides |
| Fuel floor zeroed thrust in v2 | A vehicle closing fast below the floor coasted into the target |
| Fly-throughs did not end an episode | Collision was not a terminal condition; the recorder reported "timeout" |
| Recorder ignored work-volume truncation | Kept stepping finished episodes ("timeout" 3.7 km out) |
| Glideslope used module-default n, dt, authority | Unstable at `--dt 5` (ended 115 m out at 4.5 m/s); wrong units at other authorities |
| Fault injection aliased the filter's output | A stuck thruster rewrote `applied_action`; bias faults burned no fuel |
| mission_ctrl braked by `force . velocity` | A RETREAT moving away was braked back toward the target |
| mission_ctrl held (zero thrust) on any rejected goal | Bad timeout / over-limit goal / LLM outage while closing fast -> coast into target |
| mission_ctrl clipped before the keep-out check | A 4 N push at the target came back as 1 N at the target |
| `validate_goal` compared NaN with `<`/`>` | `{"desired_distance_m": NaN}` from an LLM reply was ACCEPTED end to end |
| OpenAI adapter ignored tool calls; cost was always 0 | Tool-call replies became `None`; the cost limit could never fire |
| matplotlib pinned to 3.6.3 | The video renderer crashed (needs >= 3.8) |
| ROS package never built; bridge aborted on start | CMake signature mix, missing `config/`, wrong executable path, nonexistent `Node::Publish`, duplicate `use_sim_time`, topic mismatch with the plugin |
| ROS CI could not fail | Python tests died at import with no results (reported as 0 failures); launch_testing + pytest 8 dropped most test files |
| Two validation metrics could not fail | "sim-to-sim" compared the env with a mock wrapping the same env; perception was solved on noise-free projections |

---

## 6. What has been added

### 6.1 The environment module — `env/spacecraft_env.py`
Full Gymnasium implementation matching tests and documentation. Now accepts an injected `MissionConfig` and selectable `safety_mode` (`off`, `v1`, `v2`).

### 6.2 Certified guidance law — `control/glideslope.py`
Proportional speed decay with CW feed-forward cancellation at the mission's mean motion, gain
limited to `1/dt`, closing speed capped by what the actuator can stop. `glideslope_controller(run_config)`
binds it to a run. Docks **200/200 seeds**, 0 interlock fires.

### 6.3 Interlock v2 Runtime Assurance — `control/rta.py`
Predictive, minimal-intervention filter. Every step is checked (sampled through the step, not just
at its end) against a closing-speed envelope that tapers from 0.2 m/s at 5 m to half the docking
speed near the target, plus a capture-sphere fly-through test. An unsafe policy command is replaced
by the certified glideslope (Simplex backup); if even that is unsafe, by a speed-sized brake on the
closing component. It never adds speed, never reverses the vehicle, never pushes a retreating vehicle
toward the target, and still brakes below the fuel floor (`emergency_brake`).

### 6.4 CLI Simulation Orchestrator — `simulate.py` & `config.py` (NEW)
- Provides typed `MissionConfig` object.
- Accepts all mission inputs via CLI (`--start-pos`, `--target-pos`, `--start-vel`, `--target-vel`, `--fuel`, `--duration`, `--dt`, `--mean-motion`, `--safety`, `--seed`, `--output-dir`).
- Automatically computes relative coordinates.
- Outputs structured artifacts: `mission.json`, `telemetry.json`, `metrics.json`.

### 6.5 Telemetry Recorder, Videos & Dashboard
- `viz/record.py`: Unified episode recorder delegating to `SpacecraftEnv`.
- `viz/render.py` & `viz/make_videos.py`: 1080p MP4 renderer for 3 acts and stitched presentation film.
- `viz/dashboard.py`: Standalone HTML replay widget (`dashboard.html`).

### 6.6 Test suites
- MVP: **84 tests** (`pytest spaceforge_mvp/tests -q`): CW dynamics, environment, interlocks, RTA,
  domain randomization, validation matrix, CLI, regression tests for every review finding,
  renderer/dashboard smoke, and a check that the shipped checkpoint docks.
- ROS packages offline (the `python` CI job's selection): **155 passed, 9 skipped**.
- ROS CI job (`colcon build` + `colcon test`, Humble + Harmonic container): **170 tests, 0 failures**,
  including the live Gazebo integration test.

---

## 7. Measured results

### 7.1 The five recorded scenarios (`viz/scenarios/*.json`)

Defaults: 0.1 m/s^2 per unit command, 1 s step. Closest approach and speed are the continuous-time
values (10 ms replay) where the trace was annotated.

| Scenario | Safety | Seed | Steps | Start range | Closest approach | Speed there | Δv (m/s) | Interlock fires | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| `docking` (glideslope) | v1 | 42 | 98 | 14.33 m | 0.4933 m | 0.0173 m/s | 1.381 | 0 | **docked** |
| `rl_policy` (shipped PPO) | v1 | 42 | 208 | 18.22 m | 0.4809 m | 0.0216 m/s | 2.148 | 0 | **docked** |
| `hot_open` (reckless) | off | 7 | 11 | 6.00 m | 0.0417 m | 1.0957 m/s | 1.106 | 0 | collision |
| `hot_v1` (zeroing veto) | v1 | 7 | 14 | 6.00 m | 0.2634 m | 0.4998 m/s | 0.500 | 9 | collision |
| `hot_v2` (predictive RTA) | v2 | 7 | 70 | 6.00 m | 0.4850 m | 0.0170 m/s | 0.795 | 66 | **docked** |

### 7.2 The certified guidance law over 200 seeds
- **200 / 200 dockings** (100% success rate)
- Mean steps: 107.2 (49–132)
- 0 interlock overrides
- Mean Δv: 1.411 m/s

### 7.3 The shipped PPO policy
Why the old one could not learn: at 1 m/s^2 per unit command, exploration noise alone moves the
vehicle about 1 m/s per step against a 0.1 m/s docking tolerance. Controlled comparison (400k steps,
50 held-out episodes): 1.0 m/s^2 -> 0% with either reward; 0.1 m/s^2 -> 100% with either reward.

| Evaluation | Result |
|---|---|
| `run_eval`, seeds 42–91 | 100% success, fuel 0.0241, 205.8 steps, 3 overrides, acceptance passed |
| 500 fresh seeds, safety v1 / v2 / off | 500/500 docked each, 0 collisions |
| vs glideslope | does **not** beat it: slower (206 vs 107 steps) and more fuel (0.024 vs 0.014) |
| Training stability | poor: held-out success across snapshots of one run: 100, 58, 31, 1, 86, 98% |

Provenance and selection rule: `checkpoints/ppo_spacecraft_final.metadata.json`.

---

## 8. Run everything

From the repository root:

```bash
# 0. Activate virtual environment
source .venv/bin/activate

# 1. Run full test suite (84 tests)
python3 -m pytest spaceforge_mvp/tests -q

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
  --max-accel 0.1 \
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

1. **RL training stability**: success swings between snapshots of a single run; checkpoint
   selection on held-out seeds hides that rather than fixing it. Try lower learning rate,
   larger batches, or SAC, and select by evaluation success instead of reward.
2. **RL vs baseline**: the policy docks but is slower and costlier than the glideslope.
3. **PPO under domain randomization and faults**: the shipped policy is validated only nominally.
4. **Closed-loop docking inside Gazebo**: the plugin/bridge/ROS round trip is tested; a full
   docking run in Gazebo is not.
5. **Fuel-floor reachability**: at defaults a 300-step episode cannot drain the tank to the floor.

---

## 10. Phase 6 hardening (domain randomization + validation matrix + CI)

- **Domain randomization** (`config.py: DomainConfig`, `env/spacecraft_env.py`): per-episode mass 0.8–1.2 kg scale, ±10% actuator effectiveness, thrust/observation noise, fuel spread. **Fault injection** (`FaultConfig`): stuck thruster, bias, dropout, sensor freeze, fuel leak. Both identity-by-default: 45/45 pre-existing tests pass unchanged, and identity-domain trajectories match bit-for-bit.
- **Communication latency** (`rl_env MockTransport`): whole-control-step delay line, `ceil(latency/dt)`; 1–10 ms studies need backend dt ≤ 0.01 s (at dt=1 s any nonzero latency conservatively costs one step). Seeded noise replays identically.
- **Validation matrix** (`eval/validation_matrix.py`, 120 episodes/level, seed 0): nominal/randomized/faults all **100% docking**, fuel 0.0141/0.0159/0.0363, Δv 1.41/1.71/1.51 m/s, 0 invariant violations; pose error under 0.5 px keypoint noise mean 3.7 cm (p95 13.9 cm); LLM valid 1.00/contained 1.00; curriculum advance+rollback true; exact STM vs independent RK4 max divergence 4.5e-11 m (a 0.1 s semi-implicit Euler engine step drifts 2.5 cm). Full table in `VALIDATION_RESULTS.md`.
- **CI**: `python` (MVP + all ROS offline suites + matrix smoke), `frontend` (node logic + jsdom browser smoke), `lint` (ruff F/E9), `ros` (colcon build + C++ plugin + all package tests + live Gazebo integration; verified in a matching container, 170 tests, 0 failures).
- **Known limitations**: see §9; latency quantization is conservative at dt=1 s.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Chaser / Target** | Spacecraft performing docking, and the target object it approaches. |
| **Hill / LVLH Frame** | Coordinate system centered on target: `x` (radial), `y` (in-track), `z` (cross-track). |
| **CW STM** | Closed-form State Transition Matrix propagating orbital state analytically. |
| **Interlock v1** | Shipped rule engine that zeroes thrust commands upon proximity violations. |
| **Interlock v2 (RTA)** | Runtime Assurance filter that substitutes active opposing braking thrust. |
| **Glideslope** | Exponential range decay guidance law with CW drift feed-forward cancellation. |
