# Design

How the simulator, the controllers and the safety filters work. Numbers quoted here are measured;
[results.md](results.md) has the full tables and the commands that produce them.

## Frame and dynamics

All states are relative to the target in its local orbital (Hill, or LVLH) frame:

- `x`: radial, away from Earth (R-bar)
- `y`: along the orbit (V-bar)
- `z`: normal to the orbit plane (H-bar)

Unforced relative motion follows the Clohessy-Wiltshire equations:

```
ax =  3n²x + 2n·vy
ay = -2n·vx
az = -n²z
```

with mean motion `n = 0.00113 rad/s` (about a 92-minute low Earth orbit). The coupling terms are
why pointing at the target and thrusting does not fly a straight line: radial offsets push you
outward, and along-track speed turns into radial motion.

`core/cw_dynamics.py` propagates one step with the closed-form state transition matrix, holding the
commanded acceleration constant over the step:

```
X(k+1) = Φ(dt)·X(k) + Γ(dt)·u
```

There is no numerical integrator in the loop. An independent RK4 of the same equations at 10 ms
agrees to 4.5e-11 m over 20 approaches.

## Environment

`env/spacecraft_env.py` is a Gymnasium environment.

**Observation** (10 values): position / 20 m, velocity / 2 m/s, fuel fraction, elapsed fraction of
the episode, and a 2-D unit bearing to the target.

**Action** (3 values in [-1, 1]): commanded acceleration as a fraction of the actuator authority,
0.1 m/s² by default.

**One step:**

1. The safety filter (off, v1 or v2) turns the command into the applied command.
2. Faults, if enabled, act on the applied command (stuck axis, bias, dropout).
3. Domain randomization, if enabled, scales and perturbs it (mass, thruster effectiveness, noise).
4. The state is propagated with the exact CW step.
5. Fuel drops by `0.001 × Σ|applied|`, plus any leak.

**Episode end:**

| Outcome | Condition |
|---|---|
| docked | within 0.5 m and slower than 0.1 m/s |
| collision | the step passed through the 0.5 m sphere at or above 0.1 m/s (checked at 10 points through the step) |
| fuel | tank empty |
| out of bounds | more than 100 m away |
| timeout | 300 steps |

**Reward:**

| Term | Value |
|---|---|
| progress | `+5 × (d_prev − d)` |
| moving away | `−1 × max(0, d − d_prev)` |
| speed | `−0.5 × speed` |
| thrust | `−0.01 × Σ\|applied\|` |
| fuel | `−0.1 × fuel used this step` |
| fast inside 5 m | `−5 × max(0, (speed − 0.2) / 0.2)` |
| collision | `−50` |
| docked | `+100` |

`info["reward_breakdown"]` reports each term every step.

## Controllers

**Glideslope** (`control/glideslope.py`). Commands a velocity toward the target of magnitude
`min(0.5, d / 30, 0.5 × sqrt(2 · a_max · d))` and tracks it with gain `min(0.7, 1/dt)`. It cancels
the CW drift terms in feed-forward. The square-root term keeps the commanded speed low enough to
stop in the remaining distance with half the authority. Docks 200/200 random starts in 107 steps
on average without the filter ever acting.

**PPO** (`checkpoints/ppo_spacecraft_final.zip`). Stable-Baselines3, MLP 64-64, trained 300k steps
on random starts within 20 m. The shipped weights are the 50k-step snapshot, chosen because it
scored best on held-out seeds 1000-1099. Snapshots of the same run scored between 1% and 100%, so
training is not stable. `checkpoints/ppo_spacecraft_final.metadata.json` records the run.

The first version of this policy never docked. At 1 m/s² of authority per unit command, the
exploration noise alone changes speed by about 1 m/s per step, against a docking limit of 0.1 m/s.
With the same PPO settings, 0.1 m/s² trains to 100% and 1 m/s² to 0%.

**Direct** (`hot_approach_action`, "hot" on the command line). Full thrust at the target, never brakes. It exists to test
the filters.

## Safety filters

Both filters treat the controller as untrusted. They see its command, the state and the fuel, and
decide what reaches the thrusters.

### v1: veto (`safety/mission_safety.py`)

- Below 5% fuel: no thrust.
- Within 5 m and faster than 0.2 m/s: a command passes only if it lowers the speed over the next
  step. Anything else is zeroed.

v1 cannot brake on its own. A vehicle that is already fast coasts on into the target: the direct
law hits at 0.50 m/s instead of 1.10 m/s.

### v2: runtime assurance (`control/rta.py`)

v2 follows the Simplex architecture. An unverified controller flies until its next step would be
unsafe; then a simple controller whose behaviour is known flies that step.

The safety condition is a limit on closing speed (the rate at which range shrinks) as a function of
range `r`:

```
r ≥ 5 m:  v_lim(r) = sqrt(0.2² + 2 · f · a_max · (r − 5))      f = 0.5
r < 5 m:  v_lim(r) = clamp(0.2 · r / 5, 0.05, 0.2)
```

Outside 5 m this is a braking curve. Any vehicle under it can slow to 0.2 m/s by the keep-out
boundary using half the available thrust. Inside 5 m the limit tapers to half the docking speed.
Inside the 0.5 m capture sphere, total speed must also stay under 0.1 m/s.

Each step v2:

1. Predicts the step under the controller's command, sampling 8 points through it. If every point
   is within the limit, the command passes unchanged.
2. Otherwise it predicts the glideslope's command. If that one is within the limit, the
   glideslope flies this step.
3. Otherwise it applies a brake on the closing component of velocity, sized to remove the excess
   in one step. It never reverses the vehicle and never adds sideways speed.
4. Below the fuel floor it still brakes when coasting would break the limit; otherwise it holds.

`SafetyConfig.outer_braking_fraction = None` restores the original behaviour, where the limit
applied only inside 5 m. That version let the PPO policy reach the boundary at 0.958 m/s.

## Browser simulator

`viz/lab/sim.js` is a line-by-line port of the dynamics, environment, both filters, the glideslope
and the PPO forward pass. `viz/lab_export.py` writes the policy weights (`policy.js`) and a fixture
file from the Python code. `viz/lab/sim.test.js` compares the two:

- 160 random cases each for the CW step, glideslope, v1, v2, step prediction and brakes
- 86 envelope values
- 12 full episodes, step by step: position and velocity to 1e-9, fuel to 1e-12, identical
  overrides and outcomes
- PPO actions to 2e-5 (float32)

`tests/test_lab_export.py` fails if the fixtures or weights no longer match the Python code and
checkpoint. Runs with observation noise or domain randomization are not matched step by step,
because the browser uses its own random number generator.

`viz/dashboard.py` inlines the page, styles and scripts into `viz/dashboard.html`, so the file
works offline.

## ROS 2 stack

The packages in `ros2_ws/src` mirror the Python pieces on ROS 2 Humble and Gazebo Harmonic:

- The Gazebo plugin propagates with the same state transition matrix, and the bridge maps ROS
  wrench commands to it.
- `mission_ctrl` validates goals and filters thrust independently of who proposes them. The LLM
  planner and the learned policy can only send it requests.
- `perception` estimates target pose from keypoints with OpenCV solvePnP: 3.7 cm mean error at
  0.5 px keypoint noise.

`mission_ctrl` has its own safety rules and does not yet include the outer braking curve.
