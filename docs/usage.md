# Usage

All commands run from the repository root.

## Install

Python 3.10-3.12. Install CPU torch first; the default Linux wheel pulls in several GB of CUDA
libraries that this project does not use.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
pytest -q                   # 91 passed
```

On Python 3.13 the pinned numpy 1.26.4 has no wheel. Use
`pip install gymnasium==1.3.0 stable-baselines3==2.9.0 numpy "matplotlib>=3.8" pytest pillow`
instead of the requirements file.

## Run a mission

Every mission value is a required flag. Positions and velocities are in the Hill frame; the
simulator subtracts the target state to get the relative one.

```bash
python3 -m spaceforge_mvp.simulate \
  --controller glideslope \
  --start-pos 12 -7 3.5 --target-pos 2 1 0 \
  --start-vel 0 0 0 --target-vel 0 0 0 \
  --fuel 0.8 --duration 120 --dt 1 --mean-motion 0.00113 \
  --safety v2 --seed 42 --output-dir runs/demo
```

| Flag | Values |
|---|---|
| `--controller` | `glideslope`, `rl` (needs `--checkpoint checkpoints/ppo_spacecraft_final.zip`), `hot` (reckless) |
| `--safety` | `off`, `v1`, `v2` |
| `--max-accel` | actuator authority in m/s² per unit command, default 0.1 |
| `--nondeterministic` | sample PPO actions instead of using the mean |

The run writes three files to `--output-dir`:

| File | Contents |
|---|---|
| `mission.json` | the resolved inputs |
| `telemetry.json` | one frame per step: `t`, `pos`, `vel`, `cmd`, `applied`, `executed`, `override`, `dist`, `speed`, `fuel`, `dv` |
| `metrics.json` | `success`, `steps`, `start_dist`, `end_dist`, `end_speed`, `fuel_used`, `dv`, `overrides`, `terminal_reason` |

## Train and evaluate

```bash
python3 -m spaceforge_mvp.training.train --total_timesteps 500                  # smoke test
python3 -m spaceforge_mvp.training.train --total_timesteps 300000 --save_freq 50000 --seed 1
python3 -m spaceforge_mvp.eval.run_eval --checkpoint checkpoints/ppo_spacecraft_final.zip --episodes 50
```

Checkpoints go to `runs/ppo/`. Training is unstable from snapshot to snapshot, so evaluate every
saved snapshot on held-out seeds and keep the best, rather than taking the last one.

## Browser simulator

Open `spaceforge_mvp/viz/dashboard.html`. It is a bundled copy of `spaceforge_mvp/viz/lab/`. After
changing anything in `lab/`:

```bash
python3 -m spaceforge_mvp.viz.dashboard
```

After changing the physics, either filter or the checkpoint, regenerate the fixtures and weights
the page uses, then check the port:

```bash
python3 -m spaceforge_mvp.viz.lab_export
node spaceforge_mvp/viz/lab/sim.test.js
```

While working on the page, serve the repository (`python3 -m http.server 8000`) and open
`http://localhost:8000/spaceforge_mvp/viz/lab/index.html`, which loads the unbundled files.

**Keys:**

| Key | Action |
|---|---|
| `1` `2` `3` | Fly, Compare, Monte Carlo |
| `space` | play or pause |
| `.` | one step |
| `R` | reset |
| `←` `→` | scrub (`shift` for 10 steps) |
| `V` | next view |
| `F` | fit the view |
| `W/S`, `A/D`, `Q/E` | thrust along R-bar, V-bar, H-bar when you are flying |

URL parameters open a specific state, for example `dashboard.html#tab=compare&scenario=oblique`.

## Videos

Needs `ffmpeg` on the path.

```bash
python3 -m spaceforge_mvp.viz.make_videos 01_docking      # one clip, about 2.5 min
python3 -m spaceforge_mvp.viz.make_videos --film          # all clips plus a joined film, 6-9 min
python3 -m spaceforge_mvp.viz.preview docking 0 30 60     # single frames, for checking layout
```

The clips go to `spaceforge_mvp/viz/videos/` and are not committed. `--dpi` changes the resolution
(120 gives 1920×1080) and `--no-fades` drops the fades. The renderer prints nothing between its
start and finish lines; that is normal.

## ROS 2

Needs ROS 2 Humble and Gazebo Harmonic. The `ros` job in `.github/workflows/ros.yml` is the
reference setup.

```bash
pip install -r requirements.txt pyyaml optuna opencv-python-headless==4.10.0.84 && pip install -e .
pip install setuptools==58.2.0          # newer setuptools breaks ament_python builds on Humble
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
colcon test && colcon test-result --verbose
ros2 launch spaceforge_gazebo leo_orbit.launch.py headless:=true
```

Each package's README covers its own interface.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: spaceforge_mvp` | run `pip install -e .`, or prefix commands with `PYTHONPATH=.` |
| `pip install -r requirements.txt` tries to build numpy | you are on Python 3.13; see Install |
| torch download is several GB | install it from the CPU index first |
| `ffmpeg not found on PATH` | `sudo apt install ffmpeg` |
| the page changed nothing after an edit to `lab/` | rebuild `dashboard.html` |
| `<frozen runpy>` RuntimeWarning from `train` or `run_eval` | harmless |
