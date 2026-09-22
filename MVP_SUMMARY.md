# SpaceForge MVP — Simple Summary Guide

## 1. What is SpaceForge MVP?

**SpaceForge MVP** is a fast, lightweight, pure-Python simulation environment for training autonomous spacecraft to dock safely using **Reinforcement Learning (RL)** and **Flight Safety Rules**.

Instead of starting with heavy 3D simulation software (like Gazebo) or complex robotics frameworks (like ROS 2), this MVP focuses purely on the **math, physics, AI policy, and safety logic**. It runs fast, requires no GPU, and includes **20 automated unit tests** to guarantee correctness.

---

## 2. The 6 Core Modules (How the Code is Organized)

The project lives under the `spaceforge_mvp/` package and consists of 6 simple building blocks:

| Module | Core File | What It Does (In Simple Terms) |
|---|---|---|
| **Physics** | `core/cw_dynamics.py` | Calculates real spacecraft orbital physics using Clohessy-Wiltshire (CW) equations and exact State Transition Matrix (STM) math. |
| **Environment** | `env/spacecraft_env.py` | The Gymnasium RL playground. Translates physics into AI inputs (observations), actions (thrust commands), and rewards. |
| **Safety Interlock** | `safety/mission_safety.py` | The non-AI flight safety system. Validates actions before they hit thrusters to prevent high-speed crashes and fuel starvation. |
| **Training** | `training/train.py` | Runs Stable-Baselines3 PPO algorithm to train the spacecraft policy, save checkpoints, and log progress. |
| **Evaluation** | `eval/run_eval.py` | Tests a trained policy over 50 episodes and prints a simple performance summary table. |
| **Visualization** | `eval/plot_trajectory.py` | Plots the spacecraft's flight path in 2D ($x$ radial vs $y$ in-track) and saves it as an image (`trajectory.png`). |

---

## 3. How the AI Spacecraft Operates

### 1. What the Spacecraft Sees (Observation Vector: 10 numbers)
- **Position (3)**: Normalized $(x, y, z)$ distance to target (divided by 20.0).
- **Velocity (3)**: Normalized $(v_x, v_y, v_z)$ speed (divided by 2.0).
- **Fuel Fraction (1)**: Remaining fuel from $1.0$ (100%) down to $0.0$ (0%).
- **Mission Time (1)**: Elapsed step fraction from $0.0$ to $1.0$ (300 steps max).
- **Target Bearing (2)**: 2D direction unit vector pointing toward the target origin.

### 2. What the Spacecraft Can Do (Action Vector: 3 numbers)
- Continuous acceleration commands $[a_x, a_y, a_z]$ along 3 axes, ranging from $-1.0\text{ m/s}^2$ to $+1.0\text{ m/s}^2$.

### 3. How the AI Learns (Reward Function)
- **Progress Bonus**: Gets points for closing distance toward the target step-over-step ($5.0 \times [d_{\text{prev}} - d_{\text{curr}}]$).
- **Speed Penalty**: Small penalty for excessive relative speed ($-0.5 \times \text{speed}$).
- **Fuel Penalty**: Small penalty for burning thrusters ($-0.01 \times \sum |a_i|$).
- **Docking Jackpot**: Big **+100 bonus** when achieving successful docking!

### 4. How Accidents Are Prevented (Flight Safety Interlock)
The safety interlock is a **deterministic rule engine** (not AI) that sits between the AI policy and the spacecraft thrusters:
1. **Fuel Floor Rule**: If fuel $< 5\% \implies$ Trigger `emergency_hold` (zero out all thrust).
2. **Speed Gate Rule**: Inside $5\text{ meters}$, if speed $> 0.2\text{ m/s}$:
   - Allows **braking/decelerating thrust** (so the spacecraft can slow down).
   - Blocks **accelerating thrust** (to prevent crashing into the target).

### 5. Docking Success Definition
To successfully dock, the spacecraft must meet **two requirements simultaneously**:
1. Distance to target origin $< 0.5\text{ meters}$
2. Relative speed $< 0.1\text{ m/s}$

---

## 4. Simple Flowchart of a Simulation Step

```
   +-------------------------------------------------------+
   | 1. Spacecraft observes state (pos, vel, fuel, bearing)|
   +-------------------------------------------------------+
                               |
                               v
   +-------------------------------------------------------+
   | 2. PPO AI Policy chooses thrust action [ax, ay, az]   |
   +-------------------------------------------------------+
                               |
                               v
   +-------------------------------------------------------+
   | 3. Safety Interlock evaluates action:                 |
   |    - Fuel < 5%? -> Emergency Hold                     |
   |    - Dist < 5m & Speed > 0.2m/s & Accelerating? -> Block|
   |    - Braking or Safe? -> Allow                        |
   +-------------------------------------------------------+
                               |
                               v
   +-------------------------------------------------------+
   | 4. Physics Engine (CW STM) propagates spacecraft state|
   +-------------------------------------------------------+
                               |
                               v
   +-------------------------------------------------------+
   | 5. Environment calculates reward & checks docking     |
   |    (Success if distance < 0.5m AND speed < 0.1m/s)     |
   +-------------------------------------------------------+
```

---

## 5. Command Cheatsheet

### Install Package (Editable Mode)
```bash
pip install -e .
```

### Run All 20 Unit & Integration Tests
```bash
pytest -v
```

### Train Policy (Fast 500-Step Test vs Full 200k Run)
```bash
# Fast test run (500 steps)
python3 -m spaceforge_mvp.training.train --total_timesteps 500 --save_dir checkpoints

# Full training run (200,000 steps)
python3 -m spaceforge_mvp.training.train --total_timesteps 200000 --seed 42
```

### Evaluate Trained Policy Over 50 Episodes
```bash
python3 -m spaceforge_mvp.eval.run_eval --checkpoint checkpoints/ppo_spacecraft_final.zip --episodes 50
```

### Generate 2D Trajectory Image (`eval/trajectory.png`)
```bash
python3 -m spaceforge_mvp.eval.plot_trajectory --checkpoint checkpoints/ppo_spacecraft_final.zip --output eval/trajectory.png
```

---

## 6. Next Steps: Roadmap to Full Production System

Once the numerical policy and safety rules are validated here in MVP, the full SpaceForge system will scale into a full 3D robotics simulation:

1. **Gazebo Harmonic Sim**: Replace `cw_dynamics.py` with 3D Gazebo physics, solar radiation pressure, and C++ plugins.
2. **ROS 2 Middleware**: Wrap the environment in ROS 2 nodes (`sim_bridge`, `rl_env`, `mission_ctrl`).
3. **Perception**: Add optical cameras/LiDAR and run **YOLOv8 + OpenCV solvePnP** for 3D pose estimation.
4. **Cognition**: Add a **LangChain LLM Agent** (`llm_planner`) to command high-level goals ("Inspect North Face", "Hold Position").
5. **Web Dashboard**: Connect a **React + Three.js** web interface to view live 3D orbital trajectories in real time.
