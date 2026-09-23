# Presenting SpaceForge

Everything here is measured, not claimed. If someone asks "where does that number come from",
the command is in this file.

---

## 1. The 30-second version

> "This is a testbed for autonomous spacecraft docking. A chaser spacecraft has to fly up to a
> target and touch it gently enough to capture. There are two brains involved: a reinforcement
> learning policy that decides how to thrust, and a deterministic safety layer that can overrule
> it. The physics is the real Clohessy-Wiltshire relative motion model, so what you see is real
> orbital mechanics, not a toy simulation."

That is the whole idea. Everything else is detail.

---

## 2. What to show, in this order

**Play the film first.** `00_presentation.mp4` (45 s) is all three acts back to back. Each act opens
on a title card and closes on its result card, so it tells the story with no clicking and no file
switching. The three acts also exist as separate files if you'd rather stop and talk over one.

If you have time for a live segment afterwards, open the dashboard (below) and step through the same
episodes by hand.

### Shot 1 — the docking (act 1, video `01_docking.mp4`, 14 s)

What you're looking at: the target station sits at the centre. The dashed amber circle is the
5 m keep-out bubble. The small green dashed circle is the 0.5 m capture zone — the spacecraft
has to get inside that, and be slower than 0.1 m/s, to count as docked.

Say: *"The chaser starts 14 metres out with a 3-D offset. It closes on a glideslope — closing
speed proportional to range — with the natural orbital drift cancelled in feed-forward. 97
seconds, 1.4 m/s of total velocity change, and it arrives at 1.7 cm/s. That's a soft capture."*

Numbers on screen: range falls 14.33 m → 0.494 m, closing rate drops to almost zero, fuel bar
barely moves (0.14% used), interlock fires stay at 0.

Why it matters: this proves the environment, the physics and the docking criterion all work.
Nothing here is reinforcement learning — it's a classical guidance law — which is exactly the
point you make in Shot 2.

### Shot 2 — the shipped AI policy fails (act 2, video `02_rl_policy.mp4`, 18 s)

Say: *"This is the checkpoint that ships with the project — 200,000 timesteps of PPO, the
standard RL algorithm for this kind of continuous control. Same environment. It never docks.
It flies away: 3.72 km out, still accelerating, 31 m/s at the 300-second cutoff. Over 50
episodes it succeeds zero times."*

Then the interesting part: *"And the safety interlock never fires. Not once in 300 steps. Its closest
approach is 5.27 metres — it never even gets inside the 5-metre keep-out bubble, so the safety
layer never gets a vote. That's not a bug in the safety layer — it's the boundary of what it can
see. The interlock only has authority inside 5 metres, and the policy fails outside that."*

Why it matters: this is the difference between a demo and an engineering result. Say: *"The README
advertises twenty passing tests and a green build badge. On a fresh clone, eleven of those tests
don't run at all, and the badge is a static image — there's no CI behind it. Which is why everything
I'm about to show you is re-measured in front of you, live."*

### Shot 3 — runtime assurance showdown (act 3, video `03_runtime_assurance.mp4`, 12 s)

Three identical spacecraft, 6 m out, flying a deliberately reckless law: full thrust straight
at the target, no braking logic. The only difference is what sits between the policy and the
thrusters.

| Track | What it is | Closest pass | Speed there | Outcome |
|---|---|---|---|---|
| red | nothing between policy and thrusters | 1.35 cm | **6.48 m/s** | flies through — a collision |
| amber | interlock vetoes by **zeroing** thrust | 2.08 cm | **2.00 m/s** | still flies through |
| green | interlock vetoes by **substituting certified braking** | 0.33 cm | **0.01 m/s** | soft capture, docked in 8 s |

Say: *"Denying a command is not the same as applying the brakes. Zeroing the thrust cut the
impact speed by 70% — from 6.5 down to 2 metres per second — but the vehicle still went through
the target, because it was already moving when it entered the protected zone. Veto rules don't
slow anything down. So I changed the veto from 'do nothing' to 'the certified controller takes
the stick and brakes'. Same reckless policy, same physics: it arrives at 1 cm/s and captures."*

Why it matters: this is a real contribution with a measurable result, and it's the kind of
distinction (veto authority vs. control authority) that separates a safety filter that looks
right from one that works.

### Optional live segment — the dashboard (`spaceforge_mvp/viz/dashboard.html`)

`spaceforge_mvp/viz/dashboard.html`. One file, no server, no internet. Same recorded telemetry as
the videos, but you can drive it.

- Five scenarios as tabs: the certified docking, the shipped policy, then the three protection
  levels.
- Space bar plays and pauses, ← → step through, or drag the slider. 0.5× / 1× / 2× speed.
- The right-hand panel shows range, closing rate, relative speed, applied thrust split into radial /
  in-track / cross-track, fuel, and the interlock fire count.
- The lower chart is range on a log scale for every scenario at once, with a playhead, a ▼ at each
  scenario's closest approach, and ticks wherever the interlock fired.

The thrust bars are the clearest place to see the finding during act 3. In the zeroing column all
three axes collapse to zero the moment the craft crosses the keep-out — and it flies straight on
through, because a command of zero is not a brake (16 of the 17 vetoed frames are an exactly zero
command). In the braking column the same demand comes back with its sign flipped: the radial bar
stays saturated and fires against the velocity instead, which is what turns 6.5 m/s into 1 cm/s at
contact.

### Shot 4 — the numbers and the tests

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
# 26 passed
```

26 tests: 20 that came with the project plus 6 for the runtime-assurance filter. Mention that
when you first cloned the project, **11 of the 20 could not even run** — the environment module
had never been committed, because the `.gitignore` rule `env/` (a leftover from the standard
Python template, meant to ignore virtualenv folders) also matched the project's own
`spaceforge_mvp/env/` package. Git silently ate it. Every commit that claimed to add the
environment only changed the tests. You can show this in one command:

```bash
git log --all --oneline -- spaceforge_mvp/env/spacecraft_env.py   # no output: never committed
git check-ignore -v --no-index spaceforge_mvp/env/probe.py         # .gitignore:32:env/
```

Say: *"That's the kind of thing a green test badge hides. The tests were written against a file
that was never in the repository, so nobody who cloned it could run any of it."*

---

## 3. Numbers you can quote (all reproducible)

| Claim | Value | How it was measured |
|---|---|---|
| Certified glideslope docking rate | 200/200 seeds | `control/glideslope.py`, 300-step budget |
| Mean time to dock | 106 steps (106 s) | same run |
| Total Δv of a nominal docking | 1.40 m/s | recorded scenario |
| Shipped PPO policy success rate | **0%** over 50 episodes | `eval/run_eval.py` on the shipped checkpoint |
| Shipped PPO final range / speed | 3.72 km, 31.2 m/s | `viz/scenarios.py` |
| Reckless slam, no safety layer | passes at 1.35 cm, 6.48 m/s | continuous replay at 10 ms resolution |
| Reckless slam, veto = zero thrust | passes at 2.08 cm, 2.00 m/s | same |
| Reckless slam, veto = certified braking | 0.33 cm, 0.01 m/s, docked in 8 s | same |
| Test suite | 26 passed in ~6 s | `pytest -q` |

One thing worth explaining if asked: the closest-pass numbers (1.35 cm, 2.08 cm) come from
replaying the recorded commands with a 10 ms physics step. The environment samples at 1 Hz, and
at 6 m/s a spacecraft covers 6 metres between samples — so the 1 Hz sampler can jump clean over
the 0.5 m capture bubble and never notice. That's a real limitation of the discrete environment,
and it's also why "closest approach" is reported separately from "did the episode end".

---

## 4. Questions you will get

**"So the AI is flying the spacecraft?"**
No. The docking video is a classical guidance law. The RL policy that ships with the project
does not dock at all. What I did was make the failure measurable, then show what a working
controller looks like in the same environment. If the question is "does RL work here", the
honest answer today is "not with this reward function" — and I can say precisely why (below).

**"Why does the reinforcement learning fail?"**
Four reasons, in order of weight:
1. The reward pays for closing distance every step but never punishes flying away, and the
   episode doesn't end when the craft leaves the working volume. So drifting off forever still
   collects positive reward.
2. The only real success signal is a +100 bonus at the very end, and it's gated on a narrow
   condition (inside 0.5 m *and* slower than 0.1 m/s). That's a hard-to-hit, sparse signal.
3. Nothing tells the policy that it should slow down on final approach, so the natural gradient
   is "always thrust toward the target".
4. Fuel is nearly free: full thrust on all three axes for the whole 300 s episode costs 0.9 of
   the tank, and a real docking uses 0.14%. The fuel pressure in the design is decorative.

**"Is the safety layer real, or is it a diagram?"**
It's real code with unit tests: fuel floor wins over everything, and inside 5 m a command that
accelerates toward the target above 0.2 m/s closing speed is vetoed. But it only has *veto*
authority — it can stop the policy, it can't fly the spacecraft. That's the whole finding of
Shot 3, and it's why I added the certified-braking version.

**"What did you actually add?"**
Be straight about it. The environment module was missing from the repository, so I wrote it to
the contract the tests define. Then I added the visualisation stack (recorder, renderer,
dashboard), a deterministic reference controller so there's something honest to compare against,
the runtime-assurance filter and its 6 tests. The physics, the safety rules and the original 20
tests are the project's.

**"What would you fix next?"**
1. Reward: add a divergence penalty and terminate when the craft leaves the working volume.
2. Control rate: 1 Hz is far too slow for terminal docking. 10-20 Hz with the same physics.
3. Make certified braking the default veto everywhere, not an option.
4. Curriculum: start with 1-D radial approaches, widen only when the success rate holds.
5. Domain randomisation (mass, thrust noise, sensor noise) before any sim-to-real claim.

---

## 5. One-slide summary

> **SpaceForge — autonomous rendezvous and docking testbed**
> Real CW orbital dynamics. A PPO policy that flies away. A safety interlock that can veto but
> not fly. A certified controller that docks 200 times out of 200.
> The gap between "it runs" and "it works" — measured, with the tests to prove it.
