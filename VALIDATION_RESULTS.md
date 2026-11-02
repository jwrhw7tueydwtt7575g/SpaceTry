# SpaceForge validation results (measured)

Command (reproducible, pure Python, no ROS/Gazebo):
`PYTHONPATH=.:ros2_ws/src/spaceforge_rl_env:ros2_ws/src/spaceforge_mission_ctrl:ros2_ws/src/spaceforge_llm_planner:ros2_ws/src/spaceforge_auto_trainer:ros2_ws/src/spaceforge_perception python3 -m spaceforge_mvp.eval.validation_matrix --episodes 120 --seed 0`

Measured with Python 3.12, numpy 1.26.4, matplotlib 3.8.4, SB3 2.9.0 on CPU torch, at the default
actuator authority of 0.1 m/s^2 per unit command and a 1 s step. Every number below is a
measurement from that run, not a projection.

## Docking success (glideslope, randomized ICs, 120 episodes/level, seed 0)

| Level | Success | Fuel mean (max) | Delta-v mean | Overrides/ep | Invariant violations |
|---|---|---|---|---|---|
| nominal | **120/120 (100.00%)** | 0.0141 (0.0174) | 1.411 m/s | 0.000 | 0 |
| randomized (mass 0.8-1.2, eff ±10%, thrust/obs noise) | **120/120 (100.00%)** | 0.0159 (0.0237) | 1.709 m/s | 0.000 | 0 |
| faults (bias, 2% dropout/freeze, leak) | **120/120 (100.00%)** | 0.0363 (0.0456) | 1.514 m/s | 0.000 | 0 |

Delta-v is the physical acceleration actually delivered (after faults, mass scaling and noise),
summed over axes and steps. Agreed gates: success >= 0.85, avg fuel <= 0.35. Verdicts:
success_above_threshold=true, fuel_within_budget=true, no_invariant_violations=true.

## Shipped PPO policy (`checkpoints/ppo_spacecraft_final.zip`)

| Evaluation | Result |
|---|---|
| `python3 -m spaceforge_mvp.eval.run_eval checkpoints/ppo_spacecraft_final.zip --episodes 50` | 100% success, fuel 0.0241, 205.8 steps, 3 overrides, acceptance passed, does not beat glideslope |
| 500 fresh seeds (5000–5499), safety v1 / v2 / off | 500/500 docked each, 0 collisions |
| Fastest keep-out (5 m) entry, same seeds | off / v1: 0.958 m/s, almost 5x the 0.2 m/s gate; v2: 0.194 m/s (10.8 takeovers/episode) |

Selected on held-out seeds 1000–1099, disjoint from the reporting seeds; see
`checkpoints/ppo_spacecraft_final.metadata.json`. Training is unstable across snapshots.

## Runtime assurance v2: the outer braking curve

v2 enforces `v_lim(r) = sqrt(0.2^2 + 2 * 0.5 * a_max * (r - 5))` outside the keep-out sphere and
the taper to capture speed inside it. Measured on the seeds above: the glideslope never trips it
(0 takeovers, fastest entry 0.175 m/s), and every v2 episode in the parity fixtures stays under the
envelope at every step (worst margin -0.002 m/s). Before this change the reckless law and the PPO
policy entered the sphere at 0.3–1 m/s.

## Mission Lab parity (browser simulation vs Python)

`node spaceforge_mvp/viz/lab/sim.test.js`: **1063/1063** checks. That is 160 random cases each for
CW propagation, glideslope, v2 filter, v1 interlock, step prediction and the brakes; 86 envelope
values; and 12 full episodes compared step by step (position and velocity to 1e-9, fuel to 1e-12,
identical safety overrides and outcomes). The browser PPO forward pass matches SB3 to 2e-5 per
action and reproduces the same docking when free-running.

## Perception pose error (OpenCV solvePnP vs ground truth, 120 random poses)

| Keypoint noise | Mean translation error | p95 | Max | Failures |
|---|---|---|---|---|
| 0 px (solver self-check only) | 7.5e-08 m | 2.5e-07 m | 5.8e-07 m | 0 |
| **0.5 px (headline)** | **0.037 m** | **0.139 m** | 0.194 m | 0 |
| 1.0 px | 0.076 m | 0.279 m | 0.382 m | 0 |

Targets are 2–10 m away with a 1 m square of keypoints. Detector error (missed or swapped
keypoints) is not modelled.

## LLM goal validity (mock provider through mission_ctrl)

Valid proposals handled 20/20 (1.00); adversarial contained 25/25 (1.00)
(forbidden actuator fields, contradictions, injection text, bad versions, NaN targets).
Non-finite goal numbers are now rejected both at the planner schema and in `validate_goal`.

## Curriculum completion (synthetic outcome streams)

Perfect stream advances stage1 -> stage2 in exactly 50 episodes: true.
Poor stream rolls stage2 back to stage1: true.

## Sim-to-sim gap (20 glideslope rollouts, identical executed commands)

| Reference | Max position divergence from the exact-STM environment |
|---|---|
| Independent RK4 of the CW ODE at 10 ms | **4.5e-11 m** |
| Semi-implicit Euler at a 0.1 s physics-engine step | 2.5e-02 m |

The Gazebo plugin propagates with the same exact state transition matrix, so it inherits the first
row, not the second.

Transport consistency (MVP env vs rl_env MockTransport): max divergence 5.4e-09 m, success
agreement 20/20. Both sides run the same simulator, so this checks the transport plumbing only; it
is not evidence about physics fidelity.

## ROS 2 / Gazebo

`.github/workflows/ros.yml`, reproduced in an Ubuntu 22.04 + ROS 2 Humble + Gazebo Harmonic
container: all 8 packages build (including the C++ plugin and bridge); `colcon test` runs
**170 tests, 0 failures, 10 skipped**. That includes a live test that launches Gazebo headless with
the bridge, receives the simulation clock and odometry on ROS, publishes a wrench command, and
checks that the chaser's bridged velocity responds.

## Explicitly unverified (never claimed as passing)

- Closed-loop docking inside Gazebo (the plugin/bridge/ROS round trip is tested; a docking run is not)
- ROS 2 hardware-in-the-loop runs (no hardware-in-the-loop rig)
- Physical spacecraft hardware (no hardware)
- Shipped PPO checkpoint under domain randomization and faults (validated nominally only)
