# SpaceForge Phase 6 validation results (measured)

Command (reproducible, pure Python, no ROS/Gazebo):
`PYTHONPATH=.:ros2_ws/src/spaceforge_rl_env:ros2_ws/src/spaceforge_mission_ctrl:ros2_ws/src/spaceforge_llm_planner:ros2_ws/src/spaceforge_auto_trainer:ros2_ws/src/spaceforge_perception python3 -m spaceforge_mvp.eval.validation_matrix --episodes 120 --seed 0`

Measured on this machine (Python 3.12, numpy 1.26.4, SB3/CPU torch stack).
Every number below is a measurement from that run, not a projection.

## Docking success (glideslope, randomized ICs, 120 episodes/level, seed 0)

| Level | Success | Fuel mean (max) | Delta-v mean | Overrides/ep | Invariant violations |
|---|---|---|---|---|---|
| nominal | **120/120 (100.00%)** | 0.0014 (0.0017) | 1.417 m/s | 0.000 | 0 |
| randomized (mass 0.8-1.2, eff ±10%, thrust/obs noise) | **120/120 (100.00%)** | 0.0046 (0.0061) | 4.593 m/s | 0.000 | 0 |
| faults (bias, 2% dropout/freeze, leak) | **120/120 (100.00%)** | 0.0294 (0.0462) | 3.708 m/s | 0.000 | 0 |

Agreed gates: success >= 0.85, avg fuel <= 0.35. Verdicts: success_above_threshold=true,
fuel_within_budget=true, no_invariant_violations=true (0 violations over ~40k checked steps:
actuator bounds, finiteness, fuel range).

## Perception pose error (OpenCV solvePnP vs ground truth, 120 random poses)

Mean translation error **7.5e-08 m**, max **5.8e-07 m**, 0 solver failures.
(Noise-free projections; sensor-noise sensitivity is covered by the randomized
matrix level, which perturbs observations while judging on truth.)

## LLM goal validity (mock provider through mission_ctrl)

Valid proposals handled 20/20 (1.00); adversarial contained 25/25 (1.00)
(forbidden actuator fields, contradictions, injection text, bad versions, NaN targets).

## Curriculum completion (synthetic outcome streams)

Perfect stream advances stage1 -> stage2 in exactly 50 episodes: true.
Poor stream rolls stage2 back to stage1: true.

## Sim-to-sim gap (20 paired rollouts, identical seeds/commands)

MVP SpacecraftEnv vs rl_env MockTransport: max divergence **1.5e-08 m**,
mean smaller; success agreement 20/20 (1.00). The two simulators are
numerically the same path (float32 action cast is the only difference).

## Explicitly unverified (no host available; never claimed as passing)

- Gazebo Harmonic closed-loop docking (no simulator host; see `.github/workflows/ros.yml`)
- ROS 2 hardware-in-the-loop runs (no ROS host)
- Physical spacecraft hardware (no hardware)
- Shipped PPO checkpoint under domain randomization (known 0% even nominally; retraining is future work)
