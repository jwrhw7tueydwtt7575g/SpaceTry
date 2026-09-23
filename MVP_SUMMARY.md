# SpaceForge MVP — source of truth

This is the single reference for this project: what it is, how it works, what shipped, what was
wrong with it, everything that has been added, and every number that gets quoted. Where a claim can
be checked, the command that checks it is written next to it.

Verified 2026-09-23 on this machine (Python 3.13, numpy 2.5.3, torch 2.14.0+cpu, gymnasium 1.3.0,
matplotlib 3.11.2, stable-baselines3, ffmpeg 6.1.1). If a number here disagrees with what you
measure, the measurement wins — and this file needs fixing.

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
| Know what has been built since | §6 |
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

The project contains two things that try to solve it: a **reinforcement-learned policy** (the AI that
ships with the repo) and a **classical guidance law** written later to have an honest baseline. And
it contains one thing that tries to *prevent* it: a deterministic **safety interlock** that can veto
the commanded thrust.

What the repository is *not*: it is not a robotics stack. There is no ROS 2, no Gazebo, no
perception, no camera, no 3D view. The README describes a five-layer production architecture with
ROS nodes, YOLOv8 perception and a LangChain planner, and labels that diagram "Conceptual Layer" —
none of it exists in this repo. What exists is ~3,600 lines of numpy physics, an RL environment, a
rule engine, and (added since) a recording and rendering pipeline.

---

## 2. The physics, in plain terms

### 2.1 The frame

Everything is expressed in the **Hill frame** (also called the Clohessy-Wiltshire or LVLH frame),
which is a coordinate system that rides along with the target:

- `x` — **radial**: away from Earth, i.e. "up" from the target. Positive `x` is further out.
- `y` — **in-track**: along the direction of orbital travel (the "forward" direction).
- `z` — **cross-track**: out of the orbital plane.

The target sits at the origin `(0, 0, 0)` of this frame. The chaser's state is therefore just
"where is it, and how fast is it moving, relative to the target", which is exactly what matters for
a docking.

Mean motion `n = 0.00113 rad/s` (`config.py`), which corresponds to a low Earth orbit with a period
of about 92 minutes. The simulation step is `dt = 1 s`, and an episode lasts at most 300 steps
(300 seconds).

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
matrix. (A commit message claims "semi-implicit" dynamics; that is not what the code does. See §5.8.)

### 2.3 The state vector

10 numbers, all normalised, in this order (`env/spacecraft_env.py`):

| # | Meaning | Normalisation |
|---|---|---|
| 1-3 | relative position `x, y, z` (m) | divided by 20.0 |
| 4-6 | relative velocity `vx, vy, vz` (m/s) | divided by 2.0 |
| 7 | fuel fraction remaining | 1.0 → 0.0 |
| 8 | elapsed fraction of the episode | 0.0 → 1.0 over 300 steps |
| 9-10 | unit vector pointing at the target (2D bearing) | already a unit vector |

The action is 3 continuous numbers `[ax, ay, az]` in m/s², clipped to ±1.0. That is a small
acceleration — about a tenth of a *g* — which is realistic for a spacecraft and is also why closing
14 m takes 97 seconds rather than a few seconds.

---

## 3. The environment

`env/spacecraft_env.py` is a standard Gymnasium environment. One `step(action)` does:

1. Hand the commanded action to the **safety interlock** and take back what the interlock allows.
2. Propagate the physics one second with the *allowed* acceleration.
3. Debit fuel by the **applied** (post-interlock) magnitude — thrusters only burn what they are allowed
   to fire: `0.001 × Σ|applied|` per step off the tank. The effort penalty in the reward uses the same
   applied magnitude: `0.01 × Σ|applied|` per step.
4. Compute the reward, and decide whether the episode ended.

**Reward** (as shipped, from `config.py`):

| Term | Value | Notes |
|---|---|---|
| Radial progress | `+5.0 × (d_prev − d_now)` | pays for closing distance |
| Speed penalty | `−0.5 × speed` | every step |
| Effort penalty | `−0.01 × Σ\|a_i\|` | sum of absolute *applied* accelerations |
| Docking jackpot | `+100` | on a successful capture |

**Termination.** Success (distance < 0.5 m AND speed < 0.1 m/s) ends the episode with +100. Fuel
below zero truncates without success. Running out of the 300-step budget truncates without success.
Note what is *not* in that list: there is no terminal penalty for flying away, and no bound on how
far the chaser may travel. The reward pays for progress and never punishes the absence of it — §5.2
shows what that produces.

**The safety interlock** (`safety/mission_safety.py`) is a deterministic rule engine, not AI. It sits
between the command and the thrusters and returns one of: accept, reject (zero the thrust), or
emergency-hold. The three rules, in priority order:

1. **Fuel floor** — if fuel < 5%, `emergency_hold`: all thrust zeroed. Highest priority, overrides
   everything.
2. **Speed gate at close range** — inside 5 m, if the relative speed exceeds 0.2 m/s *and* the
   commanded thrust would increase that speed, the command is **rejected** (zeroed).
3. Otherwise, accept.

The second rule is the interesting one, and §5.5 explains why "reject" is the wrong response. Note
also that rule 2's authority begins at 5 m: outside 5 m the interlock has no opinion about anything.

Elapsed fraction, fuel, and the loop above are all in `config.py` as frozen dataclasses —
`PHYSICS_CONFIG`, `ENV_CONFIG`, `SAFETY_CONFIG`, `REWARD_CONFIG` — so the constants live in one place.

---

## 4. The repository, file by file

`spaceforge_mvp/` totals **3,617 Python lines**: 1,264 that shipped with the repo, and 2,353 added
since (the `env/` reconstruction, the `control/` package, the whole `viz/` package, and one test
file).

| Path | Lines | Origin | What it does |
|---|---|---|---|
| `config.py` | 43 | shipped | Frozen dataclasses: physics, env, safety, reward constants |
| `core/cw_dynamics.py` | 136 | shipped | CW acceleration + exact discrete STM + `cw_step` |
| `env/spacecraft_env.py` | 232 | **added** | Gymnasium env: observation, reward, interlock call, termination |
| `safety/mission_safety.py` | 90 | shipped | The v1 interlock: three rules, returns a verdict dict |
| `training/train.py` | 178 | shipped | Stable-Baselines3 PPO training entry point |
| `eval/run_eval.py` | 148 | shipped | Runs a checkpoint over N episodes, prints a metrics table |
| `eval/plot_trajectory.py` | 168 | shipped | 2D `x`-`y` trajectory plot to `trajectory.png` |
| `control/glideslope.py` | 87 | **added** | Certified guidance law + the deliberately reckless one |
| `control/rta.py` | 97 | **added** | Interlock v2: substitute braking instead of zeroing |
| `viz/record.py` | 232 | **added** | Rolls out an episode, writes a full JSON telemetry trace |
| `viz/render.py` | 638 | **added** | Broadcast-style MP4 renderer (HUD, strip, cards, fades) |
| `viz/make_videos.py` | 247 | **added** | Builds the three acts and stitches the film |
| `viz/dashboard.py` | 497 | **added** | Emits the self-contained interactive HTML replay |
| `viz/scenarios.py` | 89 | **added** | Defines and records the five scenarios |
| `viz/preview.py` | 59 | **added** | Renders single frames for inspection |
| `tests/` (8 files) | 611 | mixed | 26 tests total: 20 shipped, 6 added (`test_rta.py`) |

Generated artifacts (not source). The MP4s are build products and are deliberately **not** committed
— `INSTRUCTIONS.md` rebuilds all four in a few minutes from the traces below.

| Artifact | Size | Notes |
|---|---|---|
| `spaceforge_mvp/viz/scenarios/*.json` | — | 5 recorded traces; every figure in the videos and the dashboard comes from these |
| `spaceforge_mvp/viz/videos/00_presentation.mp4` | 8.3 MB | the film: all three acts, 44.77 s |
| `spaceforge_mvp/viz/videos/01_docking.mp4` | 2.3 MB | act 1, 14.33 s |
| `spaceforge_mvp/viz/videos/02_rl_policy.mp4` | 3.4 MB | act 2, 18.03 s |
| `spaceforge_mvp/viz/videos/03_runtime_assurance.mp4` | 2.5 MB | act 3, 12.40 s |
| `spaceforge_mvp/viz/dashboard.html` | 182 KB | the interactive replay, one file, no server |

All four videos are 1920×1080, 30 fps.

---

## 5. What shipped, and what was wrong with it

The author delivered: six modules, one trained PPO checkpoint
(`checkpoints/ppo_spacecraft_final.zip`), twenty tests, and a README describing a five-layer
production architecture. The physics module is genuinely correct. Everything downstream of it had
problems, listed here in the order of how much they block the project.

### 5.1 The environment file was never committed — the tests could not run at all

`.gitignore` contains a bare `env/` rule (line 32). That pattern ignores *any* directory called
`env` at any depth, which included `spaceforge_mvp/env/`. The consequence is that the environment —
the central file of an RL project — is not in the repository, and nothing that imports it can run.

Reproduce on a pristine clone:

```bash
git clone https://github.com/jwrhw7tueydwtt7575g/SpaceTry.git && cd SpaceTry && pytest
# Interrupted: 5 errors during collection
# 5 errors in 1.19s        <- zero tests ran
PYTHONPATH=. python -m pytest -q --continue-on-collection-errors
# 9 passed, 5 errors in 1.05s
```

So the README's "20 automated unit and integration tests (100% passing)" is wrong as shipped: on a
fresh clone pytest aborts and runs nothing. Nine tests (CW dynamics, mission safety) pass if you
force it past the collection errors; the other eleven cannot run at all.

**Fixed:** the ignore rules were anchored so they only match a top-level environment directory, and
the environment module was written to the contract the shipped tests and the README imply it has.
See §6.1 — and note the honest caveat: that file is a reconstruction, not the author's original.

### 5.2 The shipped policy never docks. Not once in fifty episodes.

```bash
PYTHONPATH=. python -m spaceforge_mvp.eval.run_eval \
    --checkpoint checkpoints/ppo_spacecraft_final.zip --episodes 50
# Success Rate               | 0.00%
# Average Fuel Used          | 0.0522
# Average Episode Length     | 300.00 steps
# Safety Overrides Triggered | 64
```

Zero successes, and it uses the entire 300-second budget in every single episode — it never docks
and never gives up. Replayed, it accelerates away from the target: 3.72 km out and moving at
31.2 m/s when the clock runs out (trace: `viz/scenarios/rl_policy.json`).

Why: the radial progress term pays for closing distance but the episode does not end if the chaser
leaves, and nothing else punishes leaving. A policy that accelerates roughly in the direction the
target *was* collects progress early, then keeps collecting tiny speed-penalty terms while it flies
away to infinity. The optimal strategy under this reward is not "dock" — it is "get the +100 once if
it is easy, otherwise leave". At 200,000 timesteps it settled on leaving.

### 5.3 The policy is worse than doing nothing

```bash
# zero-thrust baseline, seed 42, 300 steps
# end distance 20.23 m, end speed 0.01 m/s, closest approach 18.22 m
```

Doing nothing at all, the chaser drifts from 18.2 m to 20.2 m over the episode. The trained policy
ends 3,720 m away. This is the single most quotable fact about the shipped checkpoint: it is not a
weak controller, it is worse than an empty one.

### 5.4 Fuel and effort are both nominal — and the fuel floor can never fire

The reward charges `−0.01 × Σ|applied|` per step for effort; the tank is debited `0.001 × Σ|applied|`
per step. Both are proportional to the same quantity, and both are too small to steer anything.

- **Effort is cheap by construction.** Held at full thrust on all three axes for the whole 300-step
  episode (Σ|a| = 3), the total effort penalty is `0.01 × 3 × 300 = 9.0` reward — against a single
  docking bonus of `+100` and progress terms worth more than that within the first few seconds.
- **The most wasteful possible run burns 0.9 of the tank, not all of it.** The arithmetic ceiling is
  `0.001 × 3 × 300 = 0.9`, so a 300-step episode cannot empty the tank. Verified by flying the most
  wasteful controller that exists — full thrust on all three axes, every step:

  ```
  steps run              300
  fuel remaining         0.1000  (10.00% of tank)
  fuel used              0.9000
  interlock decisions    none — the interlock never intervened
  fuel floor is 5%  -> floor reached? False
  ```

- **Consequence: the fuel-floor safety rule is unreachable.** Rule 1 (§3) is the interlock's
  highest-priority rule and it cannot fire inside a 300-step episode, because the worst case leaves
  10% of the tank and the floor is 5%. The fuel-exhaustion truncation path is dead for the same
  reason. Two of the project's stated safety behaviours — "fuel < 5% → `emergency_hold`" and
  "truncate on fuel exhaustion" — cannot occur in the shipped configuration. They are not wrong, they
  are simply outside the reachable state space at this step budget and burn rate; either would become
  live with a longer episode or a larger `fuel_rate_scale`. It also means the 64 safety overrides in
  §7.3 must all be rule 2 (the speed gate) — nothing else could have fired.
- The run in §5.2 burns 5.2% of the tank to fly 3.7 km away, which reads as frugal in the eval table
  only because it flew away from the problem: the table prints fuel and distance in separate columns
  and neither means much alone.

A fuel-aware design would make the reward care about fuel at the scale the tank does, bound the
progress term by the fuel remaining, and either raise `fuel_rate_scale` or shorten the budget so the
floor rule has a job to do.

### 5.5 The interlock's veto is not a brake

Inside 5 m, the interlock's response to a too-fast approach is to **zero the thrust**. But denying a
command is not the same as slowing down. With the thrusters off, the chaser keeps whatever velocity
it had and coasts straight on through. Measured, with one reckless law flown three ways
(`viz/scenarios/hot_*.json`):

| Protection | Command while vetoed | Closest approach | Speed at closest approach | Outcome |
|---|---|---|---|---|
| none (`safety="off"`) | full thrust, always | 0.0135 m | **6.48 m/s** | straight through the target |
| v1 (shipped, zeroing veto) | zero on all three axes | 0.0208 m | **2.00 m/s** | still a collision |
| v2 (braking substitution) | saturated, opposing velocity | 0.0033 m | **0.01 m/s** | **docked in 8 s** |

The v1 interlock does reduce the impact speed — from 6.5 to 2.0 m/s — and that is worth saying
plainly. It just does not prevent the collision, because its only remedy is to stop helping. v2 gets
the same situation down to a centimetre per second by *substituting* maximum braking along the
velocity vector, and the craft docks in 8 seconds instead of dying.

The numbers in that table are not frame samples: the environment samples at 1 Hz and a
metre-per-second fly-through can step clean over the 0.5 m capture bubble between samples. They are
the closest approach recovered by holding each recorded command and re-propagating at 10 ms
sub-steps (`replay_continuous_min` in `viz/record.py`). Each trace carries both: `min_dist` (the
1 Hz sample) and `continuous_min_dist` / `continuous_min_speed` (the fine replay). For the
no-protection case the difference is large — the sampler reports 1.50 m at 3.0 m/s; the true closest
approach is 0.0135 m at 6.48 m/s.

One more property of the v1 veto worth knowing, because it is an incentive problem rather than a
physics one: the fuel debit *and* the effort penalty are both computed from the **applied** command
(§3), which is physically right — thrusters only burn what they are allowed to fire — but it makes a
vetoed step free. No fuel, no effort penalty, while the chaser keeps coasting on the velocity it
already has. A learning controller that gets vetoed inside the 5 m bubble is therefore not paying for
that step, and if the fuel or effort terms ever became significant, a policy would have a reason to
seek the veto out rather than to fly efficiently. Nothing in the shipped setup does that — the policy
never enters the bubble — but this is the first thing to test if a zeroing veto is ever placed in
front of a *learning* controller.

### 5.6 The interlock's authority starts at 5 m — and the RL failure lives outside it

The interlock only forms an opinion inside 5 m. The shipped policy's closest approach is 5.27 m
(at 31 s, moving 1.02 m/s) and it fires the interlock **zero times** in 300 steps. So the failure is
not that the safety layer was overruled; it is that the safety layer never had jurisdiction. Worth
internalising before trusting any "the safety filter is active" claim: an envelope is only as good
as the set of states it covers, and the most dangerous behaviour may be outside it.

### 5.7 The CW dynamics tests are tautologies

`tests/test_cw_dynamics.py` has three tests. Each one does this:

```python
next_pos, next_vel = cw_step(pos, vel, accel_cmd, dt=dt, n=n)
Phi, Gamma = cw_stm_matrices(dt=dt, n=n)
x_expected = Phi @ np.concatenate([pos, vel]) + Gamma @ accel_cmd
np.testing.assert_allclose(next_pos, x_expected[0:3], atol=1e-9)
```

`cw_step` is implemented as `Phi @ x + Gamma @ a`. So the assertion is comparing the function to
itself: it passes for *any* `Phi` and `Gamma` the module produces, including badly wrong ones. It
proves internal consistency, not physics.

What the tests *do* verify correctly is the acceleration term: each case compares `cw_accel` against
hand-computed analytic values (e.g. `ax = 3n²x + 2n·vy` evaluated by hand to 0.00490307), and those
are real checks.

What is missing is the check that would catch a wrong STM: propagate the same state with an
independent integrator (RK4 at a small step) and compare to the closed form. That test does not
exist. Anyone touching the physics should add it first.

### 5.8 Documentation claims that do not match the repository

- The README's green **"tests: 20 passed"** badge is a hard-coded shields.io image with an empty
  link target — not CI, not generated from any run.
- The README's architecture diagram (5 layers, 7 ROS 2 nodes, Gazebo Harmonic, YOLOv8 perception, a
  LangChain tool-calling planner) is labelled "Conceptual Layer" and describes a production system
  that is not in this repository. Nothing in the code touches robotics middleware.
- Commit `6f8cc2b` says "semi-implicit CW dynamics". The shipped code uses the exact discrete state
  transition matrix, with no integrator involved. Different method, different error behaviour.

None of these break the code. They matter because this is a project whose entire value proposition is
"verify the physics and the safety layer before you fly it" — and the verification story was the
weakest part of it.

---

## 6. What has been added

Everything in this section was written since the handover. Each item lists what it does, why it
exists, and how to run it.

### 6.1 The missing environment module — `env/spacecraft_env.py` (232 lines)

**What:** a Gymnasium environment implementing the observation, action, reward, interlock call,
fuel bookkeeping and termination described in §3.

**Why:** it was not in the repository (§5.1). Without it the checkpoint cannot be replayed, the tests
cannot run, and nothing can be measured.

**Honest caveat:** this is a reconstruction against the tests and documentation that shipped with the
repo, not a recovery of the author's original file. The evidence that it matches: all twenty shipped
tests pass against it, the observation and action are the shapes and normalisations the README
specifies, and the reward reproduces the documented terms exactly. It is the best available
reconstruction, not proof of what was deleted.

**Command:** `PYTHONPATH=. python -m pytest -q` → 26 passed.

### 6.2 A certified guidance law — `control/glideslope.py` (87 lines)

**What:** `glideslope_action(rel_pos, rel_vel)` — the classical baseline. Desired velocity points at
the target with magnitude `min(0.5, distance/30)`, so the range decays exponentially; a first-order
velocity loop tracks it (gain 0.7); the natural CW drift acceleration is cancelled in feed-forward;
the result is clipped to the actuator limit.

**Why:** to answer "is the environment even solvable?" before blaming the policy. Percentages are
only meaningful against a known-good controller.

**Result:** 200/200 seeds dock, mean 106.3 steps (49-132), 0 interlock fires, mean Δv 1.418 m/s,
mean fuel 0.142%. The time constant is deliberately chosen so the commanded speed at 5 m is
0.167 m/s — under the interlock's 0.2 m/s gate — so the safety layer never has to speak. Measured in
§7.2.

**Also in this file:** `hot_approach_action` — the deliberately reckless full-thrust slam used as the
adversarial case for the interlock in act 3.

### 6.3 Interlock v2, braking instead of vetoing — `control/rta.py` (97 lines)

**What:** a drop-in replacement for the interlock's response to a too-fast approach. Where v1 zeroes
the command, v2 projects the command so that `u · v ≤ 0` — i.e. whatever it commands must oppose the
current velocity — and scales it to maximum braking.

**Why:** §5.5. Zeroing the thrust leaves the collision intact; substituting braking prevents it.
Measured: 6.48 m/s → 2.00 m/s (v1) → 0.01 m/s and docked (v2).

**Wiring, precisely:** the environment always applies **v1**. v2 is applied by the episode recorder
(`safety="v2"` in `viz/record.py`), not by the environment itself. If v2 ships, the integration point
is `env/spacecraft_env.py::step`.

**Tests:** 6, in `tests/test_rta.py` — including that a vetoed command is replaced rather than
zeroed, that the braking action never increases speed, and that the filter turns the reckless slam
into a soft capture.

### 6.4 Recorder and scenario builder — `viz/record.py` (232) and `viz/scenarios.py` (89)

**What:** `record_episode(controller, seed, label, initial, safety, max_steps)` rolls out one episode
and returns a JSON trace: per-frame time, position, velocity, commanded acceleration, *applied*
acceleration, interlock verdict, range, speed, fuel, cumulative Δv — plus a summary block. Any
controller works, classical or a wrapped SB3 policy, because the controller contract is simply
`callable(rel_pos, rel_vel) -> action`.

The `safety` argument is what makes the three-way comparison in act 3 possible: `"v1"` uses the
environment's built-in interlock, `"v2"` substitutes the braking filter, `"off"` bypasses it
entirely as the counterfactual. `annotate_continuous=True` adds the 10 ms subsampled closest
approach described in §5.5.

**Why:** separating *recording* from *rendering* is what makes the visuals cheap to change. Every
video and every number on the dashboard reads from these JSON files; nothing is typed in by hand, so
a layout change is a re-render, and a wrong number cannot survive a re-render.

**Command:** `PYTHONPATH=. python -m spaceforge_mvp.viz.scenarios` → writes the five traces.

### 6.5 The broadcast renderer — `viz/render.py` (638 lines)

**What:** turns a trace into a presentation-grade MP4. Left: the scene — target at the centre, 5 m
keep-out ring, 0.5 m capture ring, the chaser drawn to a fixed pixel size, a thrust-coloured trail,
a scale bar, and a mini-map. Right: a HUD panel — title, per-track name and status, range, closing
rate, relative speed, applied thrust decomposed into radial/in-track/cross-track, fuel bar, and the
interlock fire count. Below the HUD: a shared log-scale range strip showing every scenario at once,
with the played-so-far portion drawn thick and the future drawn thin.

Plus, added later: **title and result cards**. Each act opens on a card stating the question it
answers and closes on a card with the verdict and four statistics built from the summary JSON.

**Why the care about layout:** the HUD is drawn every frame at fixed figure coordinates, and text
does not push anything out of the way. Two real defects were found and fixed by measurement rather
than by looking: a HUD card whose last line fell outside its own box, and (at 1080p) a 77-character
note that ran to x=1919 of a 1920-pixel frame because it was never wrapped. Notes are now wrapped by
*measuring* the rendered text width against the panel, not by guessing a character count.

**How it performs:** the first version rendered quadratically — every frame re-drew every previous
frame's HUD, because `ax.clear()` does not remove artists added to the figure. After the fix the cost
is flat at roughly 0.18 s/frame on CPU, which is what makes a full re-render a coffee break rather
than an overnight job.

### 6.6 Videos and the film — `viz/make_videos.py` (247 lines)

**What:** renders the named scenarios to MP4, applies a fade in/out to each act, and stitches the
three acts into one film.

```bash
PYTHONPATH=. python -m spaceforge_mvp.viz.make_videos              # the three acts
PYTHONPATH=. python -m spaceforge_mvp.viz.make_videos --film       # acts + 00_presentation.mp4
```

The acts are concatenated with a stream copy, which is lossless, so the film is exactly the three
acts: 1343 frames = 430 + 541 + 372, verified by frame count.

### 6.7 The interactive dashboard — `viz/dashboard.py` (497 lines) → `dashboard.html`

**What:** a single self-contained HTML file — no server, no CDN, no internet — containing the same
telemetry as the videos, playable. Five scenarios as tabs; play/pause with the space bar; ← → to
step; 0.5×/1×/2× speed; a scrub slider. The right panel shows the live numbers including the
thrust decomposition; the lower chart shows range on a log scale for all scenarios at once with the
playhead, a ▼ at each closest approach, and a tick wherever the interlock fired.

**Why:** it answers "wait, what happened at 40 seconds?" without re-rendering anything. It is also
the only artifact a viewer can drive themselves.

**Verified:** the values read back from the rendered page match the traces exactly — e.g. on the
braking track, radial thrust saturated at +1.000 m/s², fuel 99.20%, 4 interlock fires, Δv 8.034 m/s,
docked at 8 s.

### 6.8 Tests — `tests/test_rta.py` (142 lines, 6 tests) and the `.gitignore` fix

**What:** the six tests covering the v2 filter, plus the `.gitignore` change that lets
`spaceforge_mvp/env/` be committed.

**What was not done:** the three tautological CW tests (§5.7) were left as they are. They pass, they
are the author's, and rewriting them changes the test count and the story — but a reviewer should
know they prove nothing about the physics. The independent-integrator test that would is still
missing; it is the first thing to write if this physics is ever flown.

### 6.9 Documents

- **CRITIQUE.md** — the engineering critique, severity ordered, with the command next to every claim.
- **PRESENTATION.md** — what to say over each clip, the numbers, and the questions you will get.
- **MVP_SUMMARY.md** — this file.

---

## 7. Measured results

Every number below was produced on this machine by the command shown. Nothing here is copied from a
summary the project printed about itself.

### 7.1 The five recorded scenarios

From `viz/scenarios/*.json` (summary blocks). "Closest" and "speed there" are the 10 ms subsampled
values where present.

| Scenario | Safety | Seed | Steps | Start range | Closest approach | Speed there | Δv (m/s) | Interlock fires | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| `docking` (certified glideslope) | v1 | 42 | 97 | 14.33 m | 0.4945 m | 0.0173 m/s | 1.397 | 0 | **docked** |
| `rl_policy` (shipped checkpoint) | v1 | 42 | 300 | 18.22 m | 5.2681 m | 1.0234 m/s | 53.825 | 0 | timeout, 3.72 km out at 31.18 m/s |
| `hot_open` (reckless, no protection) | off | 7 | 60 | 6.00 m | 0.0135 m | 6.4799 m/s | 61.491 | 0 | flew through the target |
| `hot_v1` (reckless, zeroing veto) | v1 | 7 | 60 | 6.00 m | 0.0208 m | 1.9999 m/s | 53.008 | 17 | still a collision |
| `hot_v2` (reckless, braking veto) | v2 | 7 | 8 | 6.00 m | 0.0033 m | 0.01 m/s | 8.034 | 4 | **docked in 8 s** |

Two details worth keeping straight:

- `min_dist` in a trace is the 1 Hz sample; `continuous_min_dist` is the subsampled truth. They
  differ a lot for the fast fly-throughs (1.5001 m vs 0.0135 m for `hot_open`) and hardly at all for
  the slow captures (0.4945 m vs 0.4945 m for `docking`) — which is exactly what you would expect,
  and a useful sanity signal that the subsampling is doing its job.
- "Interlock fires" is the count of frames where the interlock took a non-accept decision. The
  certified glideslope fires it zero times in 200 seeds — by design, not by luck (§6.2).

### 7.2 The certified guidance law over 200 seeds

| Metric | Value |
|---|---|
| Dockings | **200 / 200** |
| Steps to dock | mean 106.3, min 49, max 132 |
| Interlock fires | 0 across all 200 episodes |
| Δv | mean 1.418 m/s, max 1.737 m/s |
| Fuel used | mean 0.142% of tank |

Re-derive with the sweep in §8.5.

### 7.3 The shipped policy over 50 episodes

| Metric | Value |
|---|---|
| Success rate | **0.00%** |
| Average fuel used | 0.0522 (5.22% of tank) |
| Average episode length | 300.00 steps (every episode, full budget) |
| Safety overrides triggered | 64 |

### 7.4 The do-nothing baseline

| Metric | Seed 42, 300 steps, no thrust |
|---|---|
| End distance | 20.23 m (started at 18.22 m) |
| End speed | 0.01 m/s |
| Closest approach | 18.22 m |

The trained policy ends 3,720 m away; doing nothing ends 20 m away.

### 7.5 Verification of the artifacts themselves

| Check | Method | Result |
|---|---|---|
| Test suite | `pytest -q` | 26 passed |
| Fresh-clone behaviour of the original | clone the shipped commit, run pytest | aborts: 5 collection errors, 0 tests run (9 passed / 5 errors if forced past them) |
| Video geometry | `ffprobe` | all four 1920×1080, 30 fps |
| Film integrity | frame-count arithmetic | 1343 = 430 + 541 + 372 (exact) |
| Result cards complete | pixel ink in each of the four stat columns, on the last frame before the fade | present in all three acts |
| Nothing clipped | bright-ink bounding box vs the frame border, 9 sampled scene frames + all card frames | no ink touching any border |
| Notes inside the HUD panel | ink x-range in the note rows | 1335..1848, panel edge 1881 |
| Card reveal timing | outro length minus the fade, from the render constants | 0.96 / 1.02 / 1.08 s of settled hold |
| Dashboard vs trace | values read back from the rendered page | match exactly (§6.7) |

---

## 8. Run everything

From the repository root. `PYTHONPATH` is set explicitly because the package is not installed into
the venv in editable mode.

```bash
# 0. environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # numpy, gymnasium, stable-baselines3, matplotlib, pytest

# 1. tests (26)
PYTHONPATH=. .venv/bin/python -m pytest -q

# 2. record the five scenarios (writes viz/scenarios/*.json)
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.scenarios

# 3. the videos and the film (writes viz/videos/*.mp4)
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.make_videos --film

# 4. the interactive dashboard (writes viz/dashboard.html, then open it in a browser)
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.dashboard

# 5. the certified baseline over 200 seeds (§7.2, ~20 s)
PYTHONPATH=. .venv/bin/python -c "
from spaceforge_mvp.control.glideslope import glideslope_action
from spaceforge_mvp.viz.record import record_episode
ok = steps = fires = 0
for seed in range(200):
    s = record_episode(glideslope_action, seed=seed, safety='v1')['summary']
    ok += bool(s['success']); steps += s['steps']; fires += s['overrides']
print(f'{ok}/200 docked, mean {steps/200:.1f} steps, {fires} interlock fires')"

# 6. the shipped policy's own report (§7.3)
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.eval.run_eval \
    --checkpoint checkpoints/ppo_spacecraft_final.zip --episodes 50

# 7. train a new policy (500 steps is the smoke test, 200k is what shipped)
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.training.train --total_timesteps 500
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.training.train --total_timesteps 200000 --seed 42

# 8. inspect single frames of a scenario without rendering a video
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.preview
```

Timings on this machine: tests 4 s; scenarios a few seconds; the full render (three acts + film)
about 8 minutes; the 200-seed sweep about 20 s.

---

## 9. What is still open

Listed so nobody has to guess which parts are finished.

1. **The policy is still broken.** Nothing here retrains it. §5.2 explains the mechanism (no terminal
   penalty for leaving, no bound on the escape, effort penalty decoupled from fuel), but the reward
   was not redesigned and the checkpoint was not retrained. The fix is the obvious next piece of
   work: bound the episode, punish leaving, tie the effort penalty to fuel actually burned.
2. **v2 is not wired into the environment.** It lives in `control/rta.py` and is reachable through
   the recorder (`safety="v2"`). The environment still ships v1 behaviour. Deciding whether v2
   replaces the interlock, or sits alongside it as a certified filter with its own certificate, is a
   design decision that has not been made.
3. **The CW tests still prove nothing.** §5.7. The independent-integrator test is missing, and the
   three shipped tests should be rewritten or clearly labelled as consistency checks.
4. **There is no CI.** The README's green badge is a static image. Nothing runs the tests
   automatically, which is how a missing environment module survived to a handover.
5. **The README still overclaims**, and has been left alone: it is the author's document, and
   rewriting it would erase the evidence that CRITIQUE.md is built on. If this is ever published, the
   README needs a Status section stating plainly that the policy scores 0% and that the architecture
   diagram is aspirational.
6. **The perception and planning layers do not exist.** No camera, no ROS, no planner. Out of scope
   so far, and should stay out of scope until the control problem is solved.
7. **The reconstruction caveat stands.** `env/spacecraft_env.py` is a faithful reimplementation
   against the tests and docs, not the deleted original. If the original ever turns up, diff it.
8. **One of the three safety rules cannot fire.** The fuel floor (§5.4) is unreachable inside a
   300-step episode, and so is the fuel-exhaustion truncation. Decide what to do about it: raise
   `fuel_rate_scale`, lengthen the budget, or remove the rule and stop claiming it as protection.
   A safety rule that cannot trigger is worse than no rule, because it reads as coverage.

### 9.1 The author's roadmap, as shipped

The original version of this document ended with a five-step plan from this prototype to their
production system. Kept here because it is the project's stated direction, with a note on where each
step actually stands.

1. **Gazebo Harmonic sim** — replace `cw_dynamics.py` with 3D physics, solar radiation pressure and
   C++ plugins. Not started, and deliberately low priority: the linear CW model is the right fidelity
   for a rendezvous control problem, and a 3D engine would not have surfaced any of the findings in
   §5.
2. **ROS 2 middleware** — wrap the environment in `sim_bridge`, `rl_env`, `mission_ctrl` nodes. Not
   started; nothing in this repo imports a ROS package.
3. **Perception** — cameras or LiDAR, YOLOv8 + `solvePnP` for pose estimation. Not started, and note
   the ordering problem: the controller cannot dock from the relative state it is *already given*, so
   feeding it a noisier estimate of that same state would not help yet.
4. **Cognition** — a LangChain planner issuing high-level goals. Not started; also the layer furthest
   from anything measured here.
5. **Web dashboard** — a React + Three.js live 3D view. Partly superseded: `viz/dashboard.html`
   already delivers the interactive replay as one dependency-free file, which is what a live demo
   actually needs (§6.7). A React rewrite would add a build step and buy nothing for this purpose.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Chaser / target** | The spacecraft doing the docking, and the object it docks with (at the frame origin). |
| **Hill / LVLH / CW frame** | The rotating coordinate frame that rides with the target: `x` radial (up), `y` in-track (forward), `z` cross-track. |
| **CW equations** | The linearised equations of relative orbital motion, valid for separations small compared to the orbit radius. |
| **STM** | State transition matrix: the closed-form matrix that propagates `(position, velocity)` forward one step under a held acceleration. The code uses this directly instead of integrating. |
| **`n` (mean motion)** | Orbital rate, 0.00113 rad/s here, i.e. a ~92-minute orbit. |
| **Δv** | Total velocity change expended — the standard cost metric for a manoeuvre, in m/s. |
| **Keep-out bubble** | The 5 m sphere around the target inside which the interlock forms an opinion. |
| **Capture zone** | The 0.5 m sphere inside which a slow-enough chaser counts as docked. |
| **Docking criterion** | Range < 0.5 m **and** relative speed < 0.1 m/s, simultaneously. |
| **Interlock v1** | The shipped safety rule engine: fuel floor, and inside 5 m a too-fast approach has its thrust **zeroed**. |
| **Interlock v2 / RTA** | The replacement response: on veto, substitute maximum braking opposing the velocity. Runtime assurance = a filter that guarantees a property by overriding an untrusted controller. |
| **Glideslope** | A guidance profile where closing speed is proportional to range, so range decays exponentially and the approach slows itself down. |
| **Feed-forward cancellation** | Pre-computing the known disturbing acceleration (`cw_accel`) and subtracting it from the command, so the controller only fights the error rather than the physics. |
| **Continuous minimum** | The true closest approach, recovered by replaying recorded commands at 10 ms sub-steps, because 1 Hz sampling can skip over the capture bubble. |
| **Truncation vs termination** | Gymnasium's distinction: *terminated* means the task genuinely ended (docked, or crashed); *truncated* means the time limit or fuel ran out. The environment uses both. |
