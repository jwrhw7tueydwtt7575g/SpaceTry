# INSTRUCTIONS — build the videos and run the dashboard yourself

Everything in this file has been run on the machine this repository was prepared on (Ubuntu, Python
3.13). Commands start from the repository root, and the expected output is shown after each one so
you can tell immediately whether a step worked.

Nothing here needs a GPU, a server, or an internet connection except the one-time package install.

**What you get when you finish:** four MP4s (three acts plus a stitched 45-second film) and
`dashboard.html`, an interactive replay of the same telemetry that runs straight from your browser.

**Time:** about 10 minutes, most of it waiting for the video render.

---

## 0. What you need

| Requirement | Why | Check |
|---|---|---|
| Python 3.13 | this is the verified interpreter; the author's older pins need 3.12 or lower | `python3 --version` |
| `ffmpeg` on PATH | the renderer pipes raw frames to libx264 | `ffmpeg -version` |
| ~2 GB of disk | mostly the torch wheel | `df -h .` |
| No GPU | the physics is numpy and torch runs on CPU | — |

If `ffmpeg` is missing:

```bash
sudo apt update && sudo apt install -y ffmpeg
```

---

## 1. Install

**Do not use the pinned `requirements.txt` on Python 3.13.** Those pins are the author's
(`numpy==1.26.4`, `matplotlib==3.6.3`) and predate Python 3.13, so pip will try to build them from
source and fail. Use the stack below instead — it is exactly what everything here was verified
against.

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# CPU-only torch, installed first and from the CPU index: the default Linux wheel
# drags in the whole CUDA runtime (~5 GB) which this project never uses.
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

.venv/bin/pip install gymnasium==1.3.0 stable-baselines3==2.9.0 numpy matplotlib pytest pillow
```

Expect roughly: `torch 2.14.0+cpu`, `stable_baselines3 2.9.0`, `gymnasium 1.3.0`, `numpy 2.5.3`,
`matplotlib 3.11.2`. Check with `.venv/bin/pip freeze`.

If you specifically want the author's pins, install an older Python first (they work on 3.12 and
below): `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

**About `PYTHONPATH`.** Every command below is prefixed with `PYTHONPATH=.` because the package is
not installed into the venv. You can instead run `.venv/bin/pip install -e .` once and drop the
prefix; both work.

---

## 2. Prove the install: run the tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

**Expected:** `84 passed` in about 10 seconds.

The suite covers the CW physics, the safety interlock, the environment, training and evaluation
smoke tests, and the runtime-assurance filter. If you see `ModuleNotFoundError: spaceforge_mvp`, you
left off `PYTHONPATH=.` or you are not in the repository root.

---

## 3. Record the telemetry — the source of every number

```bash
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.scenarios
```

This writes five JSON traces into `spaceforge_mvp/viz/scenarios/`:

| File | What it is |
|---|---|
| `docking.json` | the certified guidance law, docking (seed 42) |
| `rl_policy.json` | the shipped PPO checkpoint, flying away (seed 42) |
| `hot_open.json` | reckless slam with **no** protection (seed 7) |
| `hot_v1.json` | reckless slam with the shipped veto (seed 7) |
| `hot_v2.json` | reckless slam with the braking filter (seed 7) |

Takes a few seconds.

**These five files already ship with the repository (144 KB), so every later step works without this
one.** Re-run it whenever you change the physics, the reward or the interlock and want the videos and
the dashboard to reflect it — nothing is hand-typed anywhere, so re-recording and re-rendering is
always safe.

---

## 4. Build the videos

### 4a. One act first

```bash
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.make_videos 01_docking
```

**Expected:** `done: /.../spaceforge_mvp/viz/videos/01_docking.mp4 (2.3 MB, 14.3 s)`.

About 2.5 minutes at 1920×1080. The renderer is single-threaded, so a slower CPU takes proportionally
longer. It prints one line when it starts and one when it finishes — **silence in between is
normal, not a hang.**

The `videos/` directory is created if it does not exist, so a fresh clone works immediately.

### 4b. All three acts and the stitched film

```bash
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.make_videos --film
```

**Expected, in order:**

```
01_docking.mp4              2.3 MB   14.3 s     the certified docking
02_rl_policy.mp4            3.4 MB   18.0 s     the shipped policy flying away
03_runtime_assurance.mp4    2.5 MB   12.4 s     one reckless law, three protection levels
00_presentation.mp4         8.3 MB   44.8 s     all three acts back to back, with fades
```

Allow 6-9 minutes. The film is assembled by concatenating the three acts with a lossless stream copy,
so it contains exactly the three acts — 1343 frames = 430 + 541 + 372 — and adding fades afterwards
does not re-render anything.

**Then just watch them.** Any video player will do: `xdg-open spaceforge_mvp/viz/videos/00_presentation.mp4`.

### 4c. Flags worth knowing

| Flag | Effect |
|---|---|
| `--dpi 120` | default; gives 1920×1080 |
| `--dpi 110` | 1760×990 — lower resolution, faster render |
| `--dpi 150` | 2400×1350; render time scales with pixel count |
| `--no-fades` | skip the fade to/from black on each act (use this if you plan to cut the clips yourself) |
| `--film` | also build `00_presentation.mp4` |
| `01_docking 03_runtime_assurance` | render only the acts you name |

### 4d. Check a single frame instead of a whole video

Faster than a full render when you are only checking the layout:

```bash
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.preview docking 0 30 60
```

Writes `docking_000.png`, `docking_030.png`, `docking_060.png` into
`spaceforge_mvp/viz/preview/` (change the directory with `--out DIR`). Handy options: `--title`,
`--subtitle`, and any frame numbers you like — the docking trace is 97 steps long.

---

## 5. Build and open the dashboard

```bash
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.viz.dashboard
```

Writes `spaceforge_mvp/viz/dashboard.html` and prints
`dashboard written: /.../dashboard.html  (182 KB)`. It is completely self-contained — the telemetry
is inlined as JSON, there is no server, no CDN and no internet fetch — so you can also email the file
to someone and it will still work.

Open it:

```bash
xdg-open spaceforge_mvp/viz/dashboard.html        # or double-click it in your file manager
```

**Controls:** the five scenarios are tabs across the top; the space bar plays and pauses; `←` and `→`
step four frames at a time; the slider scrubs; the 0.5× / 1× / 2× buttons change playback speed. The
right-hand panel is the live HUD — range, closing rate, relative speed, thrust split into
radial / in-track / cross-track, fuel, and the interlock fire count. The lower chart plots range on a
log scale for every scenario at once, with a ▼ marking each closest approach and a tick wherever the
interlock fired.

If your browser refuses to run JavaScript from a local file (uncommon, but some hardened setups do),
serve the directory instead:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/spaceforge_mvp/viz/dashboard.html
```

---

## 6. Reproduce every number the docs quote

### The shipped policy — expect 100% on these seeds

```bash
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.eval.run_eval \
    --checkpoint checkpoints/ppo_spacecraft_final.zip --episodes 50
```

```
Success Rate               | 100.00%
Average Fuel Used          | 0.0241
Average Episode Length     | 205.84 steps
Safety Overrides Triggered | 3
Beats Glideslope Baseline  | False
Acceptance Passed          | True
```

A `<frozen runpy>` RuntimeWarning about the module being in `sys.modules` comes from the package's
`__init__` and is harmless.

### The certified baseline — expect 200 out of 200

```bash
PYTHONPATH=. .venv/bin/python -c "
from spaceforge_mvp.control.glideslope import glideslope_action
from spaceforge_mvp.viz.record import record_episode
ok = steps = fires = 0
for seed in range(200):
    s = record_episode(glideslope_action, seed=seed, safety='v1')['summary']
    ok += bool(s['success']); steps += s['steps']; fires += s['overrides']
print(f'{ok}/200 docked, mean {steps/200:.1f} steps, {fires} interlock fires')"
```

```
200/200 docked, mean 107.2 steps, 0 interlock fires
```

About 20 seconds.

### The Phase 6 validation matrix — expect all verdicts true

```bash
PYTHONPATH=. .venv/bin/python -m spaceforge_mvp.eval.validation_matrix \
    --episodes 120 --seed 0 --output /tmp/phase6/matrix_report.json
```

```
[   nominal] success 100.00%  fuel 0.0141 (max 0.0174)  dv 1.411  overrides 0.000  violations 0
[randomized] success 100.00%  fuel 0.0159 (max 0.0237)  dv 1.709  overrides 0.000  violations 0
[    faults] success 100.00%  fuel 0.0363 (max 0.0456)  dv 1.514  overrides 0.000  violations 0
[pose]      measured: 0.5 px noise -> mean 0.037 m, p95 0.139 m, max 0.194 m over 120 trials
[llm]       measured: valid 1.0  contained 1.0
[curriculum] measured: advance-in-50 True  rollback True
[sim2sim]   measured: vs independent RK4 at 0.01 s max div 4.5e-11 m; vs semi-implicit Euler @0.1 s max div 0.025 m
[transport] measured: max div 5.4e-09 m, agreement 1.0 (same simulator both sides)
[verdicts]  {"success_above_threshold": true, "fuel_within_budget": true, "no_invariant_violations": true}
```

120 episodes per domain level (360 total), 120 PnP trials, 20/25 LLM probes,
20 paired sim-to-sim rollouts. About 15 seconds. Full numbers and the explicit
unverified list live in `VALIDATION_RESULTS.md`. Sibling packages must be on
`PYTHONPATH` for the pose/LLM/curriculum/sim2sim probes (see the CI `python`
job); without them those probes report `unavailable` instead of guessing.

### The three protection levels, side by side

The numbers are already in the traces; this prints them from the recorded data rather than
re-deriving them:

```bash
PYTHONPATH=. .venv/bin/python -c "
import json, pathlib
d = pathlib.Path('spaceforge_mvp/viz/scenarios')
for k in ('hot_open', 'hot_v1', 'hot_v2'):
    s = json.loads((d / f'{k}.json').read_text())['summary']
    print(f\"{k:9s} closest {s['continuous_min_dist']*100:6.2f} cm at \"
          f\"{s['continuous_min_speed']:5.2f} m/s  fires {s['overrides']:2d}  \"
          f\"{'docked' if s['success'] else 'collision'}\")"
```

```
hot_open  closest   1.35 cm at  6.48 m/s  fires  0  collision
hot_v1    closest   2.08 cm at  2.00 m/s  fires 17  collision
hot_v2    closest   0.33 cm at  0.01 m/s  fires  4  docked
```

`continuous_min_dist` and `continuous_min_speed` are not frame samples. The environment ticks at
1 Hz, which can step straight over the 0.5 m capture bubble, so the recorder replays each recorded
command at 10 ms sub-steps to recover the true closest approach. `MVP_SUMMARY.md` §5.5 explains why
that matters.

---

## 7. What is generated and what is committed

| Generated by steps 3-5 (not committed, safe to delete) | Committed |
|---|---|
| `spaceforge_mvp/viz/videos/*.mp4` — re-render with step 4 | the source, the tests, the five scenario traces |
| `spaceforge_mvp/viz/preview/*.png` — re-run step 4d | `dashboard.html`, so it opens without building anything (regenerate it any time with step 5) |
| `spaceforge_mvp/viz/dashboard.html` — regenerated by step 5 | `MVP_SUMMARY.md`, `CRITIQUE.md`, `PRESENTATION.md`, `INSTRUCTIONS.md` |

The MP4s are deliberately not in the repository — 16 MB of binaries that you can rebuild in eight
minutes from the committed traces is not worth carrying in git history. If you want them versioned,
either commit them yourself or attach them to a GitHub Release (a release asset is the usual place
for a build product).

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ffmpeg not found on PATH` | install it (step 0). The renderer exits immediately rather than writing a broken file. |
| `ModuleNotFoundError: spaceforge_mvp` | add the `PYTHONPATH=.` prefix, and run from the repository root. |
| `FileNotFoundError` for `scenarios/*.json` | run step 3, or check you are in the repository root — all paths are relative to it. |
| A render prints nothing for several minutes | normal. The renderer writes to a temporary file and moves it into place at the end, so the finished MP4 appears all at once. Do not start a second render. |
| Render is much slower than quoted | the renderer is single-threaded; scale the estimate by your CPU, or use `--dpi 110`. |
| `pip install torch` starts downloading several gigabytes | you skipped the CPU index. Use `--index-url https://download.pytorch.org/whl/cpu`. |
| `pip install -r requirements.txt` fails to build numpy or matplotlib | the author's pins are too old for Python 3.13 — use step 1's command set, or Python 3.12. |
| Dashboard opens but nothing animates | click inside the page first (the space-bar shortcut needs focus), and check the browser console for a CSP error from a hardened local-file policy — then use the `http.server` method in step 5. |
| `pip` prints dependency-conflict warnings like "launch-ros requires pyyaml" while installing | your shell exports a `PYTHONPATH` (or has globally installed packages) that the venv can see | harmless. The project commands in this file set `PYTHONPATH=.` explicitly, which overrides it. |
| Dashboard numbers look stale after you changed the code | it is a generated file: re-run step 5. |

---

## 9. Where to read more

| File | What it is |
|---|---|
| `MVP_SUMMARY.md` | the source of truth — physics, environment, file-by-file map, findings, measured results, open items |
| `CRITIQUE.md` | the engineering critique, severity ordered, with the command that proves each claim |
| `PRESENTATION.md` | what to say over each clip, with the numbers and the questions you will get |
| `README.md` | the author's original project description, with a status note at the top |
