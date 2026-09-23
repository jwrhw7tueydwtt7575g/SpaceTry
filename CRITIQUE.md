# SpaceForge — engineering critique

Ordered by what actually blocks the project. Every claim below has a command next to it; run them
yourself before you quote me.

How I checked it: cloned the repo, tried to run the test suite, ran the shipped checkpoint in the
shipped evaluation harness, replayed the recorded command sequences through the physics at 10 ms
resolution, and compared the docs against the code line by line.

---

## BLOCKING

### 1. The environment module was never committed. Eleven of twenty tests could not even run.

```
$ python -m pytest -q
ModuleNotFoundError: No module named 'spaceforge_mvp.env'   # 5 collection errors, 11 tests never ran

$ git log --all --oneline -- spaceforge_mvp/env/spacecraft_env.py
(no output)

$ git check-ignore -v --no-index spaceforge_mvp/env/probe.py
.gitignore:32:env/    spaceforge_mvp/env/probe.py
```

The file that defines the environment (state, reward, step, reset, docking criterion) is not in a
single commit — but `.gitignore` line 32 says `env/`, a bare pattern that matches at any depth, so
it silently swallowed the project's own `spaceforge_mvp/env/` package. Git itself told the authors
this was happening: commit `b95bdad` ("feat(env): implement SpacecraftEnv gymnasium environment and
unit tests") contains a file literally named `env`, which is the tail of git's own refusal message
`.../spaceforge_mvp/env/` committed by accident.

So every commit that claims to add the environment only ever changed the tests. Anyone who cloned
the repo got a test suite written against a file that wasn't there.

**Fix applied:** anchored the ignore rules to the repo root (`/env/`, `/venv/`, `/ENV/`) and
reconstructed `spaceforge_mvp/env/spacecraft_env.py` to the contract the tests define. 20/20
original tests now pass.

---

## HIGH

### 2. The reward function cannot produce a docking policy.

Measured on the shipped checkpoint, 50 episodes:

```
Success rate                 0.00%
Average episode length     300.00 steps        (every episode, i.e. the full budget)
Final range                3.72 km, 31.2 m/s, still accelerating
Closest approach           5.27 m               (never enters the keep-out bubble)
Safety overrides          64 total              (only in the episodes that start inside 5 m)
```

Four things in the reward make this outcome the expected one, not a surprise:

- **No divergence penalty and no out-of-bounds termination.** `Reward = 5.0·(dist_prev − dist_curr)
  − 0.5·speed − 0.01·|u| + bonus`. Closing distance pays every step; flying *away* is punished only
  through that same progress term, and the episode keeps running for the full 300 steps no matter
  how far away the craft gets. The policy can score positive reward while leaving the scene.
- **The only success signal is a sparse terminal bonus.** +100 requires `dist < 0.5` **and**
  `speed < 0.1`. That's a small target in a 40 m-wide state space, hit exactly once at the end.
  Nothing else in the reward says "slow down near the target".
- **The velocity penalty is too weak to teach braking.** 0.5·speed at 1 m/s is 0.5; a single step of
  progress reward at 1 m residual closing is 5.0. Drifting fast is cheap compared to the payoff for
  closing.
- **Fuel is nearly free.** Full-thrust all three axes for the entire 300 s costs ≈ 0.9 of the tank
  (the shipped policy used 0.054). A real docking from 14 m costs 0.0014. The fuel-pressure term in
  the design can't influence the policy at all.

**Fix:** add a divergence penalty, terminate on leaving the work volume, and either densify the
terminal signal (a shaping term that rewards *low closing speed* inside the bubble) or move to a
curriculum. All four are cheap; the environment already supports them.

### 3. The safety interlock has no authority where the failures happen, its veto is not a brake, and one of its rules can never fire.

The interlock is three rules: fuel floor → emergency hold; `goal_type == "APPROACH"` **and**
`dist < 5 m` **and** closing above 0.2 m/s → reject the command; otherwise accept.

Three problems, all measurable:

**(a) Wrong domain.** Its authority is exactly 5 metres. The shipped policy's failure mode lives at
kilometre scale, so the interlock fires 0 times in a 300-step fly-away. A safety layer that only
watches the last 5 metres of a mission cannot catch a mission-level failure.

**(b) "Reject" means zero the command, and zeroing the command is not braking.** Three identical
spacecraft 6 m out, flying the same reckless law, differ only in what sits between the policy and
the thrusters:

| Protection | Closest pass | Speed there | Outcome |
|---|---|---|---|
| none | 1.35 cm | 6.48 m/s | penetrates — collision |
| veto → zero thrust (as shipped) | 2.08 cm | 2.00 m/s | penetrates — collision |
| veto → certified braking | 0.33 cm | 0.01 m/s | soft capture, docked in 8 s |

Zeroing cut the impact speed by 70% and did not prevent the collision, because the vehicle was
already moving when it entered the bubble. A veto has to hand control to something that can fly
the spacecraft, not just take the stick away.

**(c) The fuel-floor rule is unreachable, so the highest-priority safety rule is dead code.** The
tank is debited `0.001 × Σ|applied|` per step, the episode is at most 300 steps, and each axis is
capped at 1.0 m/s². The largest drain any episode can produce is therefore
`0.001 × 3 × 300 = 0.9` — the worst case leaves 10% of the tank, and the floor is 5%. Flying the most
wasteful controller that exists (full thrust on all three axes, every step) confirms it:

```
steps run              300
fuel remaining         0.1000  (10.00% of tank)
fuel used              0.9000
interlock decisions    none — the interlock never intervened
```

The fuel-exhaustion truncation path dies with it: the tank cannot be emptied inside the budget. So
one of the three safety rules and one of the two truncation conditions cannot occur. That also tells
you what the 64 "safety overrides" in the evaluation report are — they must all be the speed gate,
because there is nothing else that can fire.

**Fix applied:** none. This is a configuration decision, not a code bug: raise `fuel_rate_scale`,
lengthen the budget so the floor is reachable, or delete the rule and stop counting it as coverage.

**Fix applied:** `control/rta.py` — on veto, substitute maximum braking along the velocity vector
instead of a zero command, using the same CW model the envelope is defined in. 6 new tests.

### 4. The README describes a system that isn't in the repository.

- The headline architecture (5 layers, 7 nodes, ROS 2, Gazebo, YOLOv8 perception, LangChain
  reasoning) does not appear anywhere in the code. `requirements.txt` is numpy, gymnasium,
  stable-baselines3, pytest, matplotlib. There is no ROS, no Gazebo, no perception, no LLM call.
  It may be a roadmap — but it's written in the present tense and reads as implemented.
- A commit message (`6f8cc2b`) says "semi-implicit CW dynamics"; the shipped implementation is the
  exact discrete state transition matrix, with no integrator in between. Those are different
  methods with different error behaviour. One of the two is wrong.
- The green "tests: 20 passed" badge at the top of the README is a hard-coded shields.io image with
  an empty link target — not CI, not generated. The text beside it claims the codebase is "fully
  scaffolded, implemented, and verified across all core modules with 20 automated unit and
  integration tests (100% passing)". On a fresh clone, 11 of those 20 could not run (item 1).
- For balance: the README's reward formula *does* match `config.py` and the implementation exactly
  (weights 5.0 / 0.5 / 0.01, bonus 100, penalty on `sum(|u|)`). That part is accurate.

**Fix:** a "Status" section at the top distinguishing what runs today from what's planned, replace the
static badge with a real CI badge (or delete it), and fix the commit message that describes an
integrator the code doesn't use.

---

## MEDIUM

### 5. Some of the physics tests cannot fail.

`test_cw_dynamics.py` cases 2 and 3 compute their expected answer as
`x_expected = Phi @ x_0 + Gamma @ accel_cmd` using the *same* `cw_stm_matrices()` the implementation
just used, then assert the implementation's output equals it (lines 70–72 and 98–100). That checks
internal consistency, not correctness: a wrong-but-self-consistent STM passes both. (The
`cw_accel` assertions in those cases *are* real — they compare against hard-coded literals like
`[0.0038307, 0.0, -0.00063845]`.)

I hand-checked the STM against the textbook closed form and it is correct — so nothing is broken
today. But these two tests would not have told you if it were.

**Fix:** hard-code one reference vector per case (textbook values for n = 0.00113, dt = 1 s), and
keep the consistency check as a separate test.

### 6. At 1 Hz, the sampler can jump straight through the target.

The environment samples at 1 s. At 6.5 m/s that's 6.5 m of travel between observations — twelve
times the 0.5 m capture bubble. Success, failure and the safety rules are all evaluated only at
sample times, so a penetration can be invisible.

Measured on the reckless-slam trace: the closest *sampled* range is 1.50 m (looks like a clean miss),
while the continuous 10 ms replay shows it passing 1.35 cm from the target at 6.48 m/s. The
environment sees a miss; the spacecraft actually went through the docking port.

**Fix:** evaluate the safety envelope on a continuous/interpolated minimum range, and use a smaller
physics step inside the keep-out zone (the visualisation already does this — `record.py`,
`replay_continuous_min()`).

### 7. The safety rule is gated on a string.

`safety/mission_safety.py:62` requires `goal_type == "APPROACH"` before the proximity rule applies.
Any other label — `"DOCK"`, `"INSPECT"`, a typo, `None` — bypasses proximity protection entirely,
and one of the shipped tests asserts that `DOCK` bypasses it, so the behaviour is load-bearing.
A safety condition that can be disabled by renaming a phase is not a safety condition.

**Fix:** make the proximity rule depend on geometry only (range and closing rate), and let
`goal_type` at most add restrictions, never remove them.

---

## LOW (hygiene, but they cost you credibility)

### 8. Reproducibility
`requirements.txt` pins numpy 1.26.4 and matplotlib 3.6.3, which don't resolve on current Python —
on 3.13 the install fails outright. It also pulls the CUDA build of torch by default, which is
~2.5 GB of wheels for a CPU-only simulation. There's no lockfile, no CI, and no lint config.
**Fix:** an unpinned `requirements.txt`, an explicit CPU torch index URL, and a GitHub Actions job
that runs pytest on push.

### 9. Artifact policy
`checkpoints/` is committed while `*.png` is gitignored yet still tracked, and intermediate run
outputs live in the tree. Generated files should be regenerable; the repo should say how.
**Fix:** keep one checkpoint in releases, not in git history, and add a `make reproduce`.

### 10. Observability
`print` throughout instead of `logging`, and `eval/run_eval.py` seeds episodes as `seed + episode`,
so the evaluation isn't reproducible from a single seed argument.
**Fix:** a `logging` config, and a seeded RNG passed explicitly.

---

## What I'd do next, in order

1. **Reward + termination** (item 2). Divergence penalty, out-of-bounds termination, a
   low-closing-speed shaping term. Then retrain and report the success curve, honestly.
2. **Control rate** (item 6). 10–20 Hz physics inside the keep-out zone.
3. **Make certified braking the default** (item 3b) everywhere the interlock can fire.
4. **Curriculum.** 1-D radial approaches first, widen the initialisation only when the success
   rate holds at the current level.
5. **Domain randomisation** (mass, thrust magnitude, sensor noise) *before* any sim-to-real claim
   is made.

---

## What already works

Worth saying plainly, because it's what the project got right:

- The CW state transition matrix is implemented correctly — I verified the closed form by hand.
- The problem is well framed: a 6-D state, 3-axis bounded thrust, a clear success condition, and a
  safety layer that's separate from the policy. That separation is the right architecture.
- The environment's public contract is tight enough that it could be reconstructed from the tests
  alone — which is how I recovered it. That's a sign the tests encode real intent.
- The safety interlock, as far as it goes, is deterministic, side-effect free, and unit tested,
  including the fuel-floor precedence rule.
