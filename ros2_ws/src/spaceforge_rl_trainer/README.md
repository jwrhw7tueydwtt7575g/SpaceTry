# spaceforge_rl_trainer

The trainer defaults to Stable-Baselines3 PPO; SAC is an explicitly selectable
alternative via `algorithm: SAC`, `--algorithm SAC`, or `config/stage1_sac.yaml`.
Every setting is explicit in YAML/JSON and on the CLI (CLI wins; `--print-config`
dumps the resolved config without training): algorithm, env mode/count,
timesteps, output directory, scenario, seeds, max steps, evaluation settings,
checkpoint/evaluation frequencies, TensorBoard, resume path, actuator limits,
acceptance gates, and PPO/SAC hyperparameters.

Dependencies: `pip install stable-baselines3 pyyaml` (install CPU torch first, see
docs/usage.md). YAML configs require PyYAML.

Example YAML (`config/stage1_ppo.yaml`; SAC mirror in `config/stage1_sac.yaml`):
algorithm: PPO
env_mode: single
n_envs: 1
total_timesteps: 10000
output_dir: runs/rl_stage1
scenario: stage1_1d
seed: 42
eval_episodes: 20
eval_seed: 1000

Run with: PYTHONPATH=ros2_ws/src/spaceforge_rl_env:ros2_ws/src/spaceforge_rl_trainer:. python3 -m spaceforge_rl_trainer.cli --config trainer.yaml
SAC with:  PYTHONPATH=ros2_ws/src/spaceforge_rl_env:ros2_ws/src/spaceforge_rl_trainer:. python3 -m spaceforge_rl_trainer.cli --config ros2_ws/src/spaceforge_rl_trainer/config/stage1_sac.yaml
Entry point (installed): spaceforge-rl-train --config trainer.yaml

Seed scheme: train env rank r uses `seed+r`; the periodic-eval env uses
`seed+100000` (`EVAL_RANK_OFFSET`); final-report episode k uses `eval_seed+k`
for the policy, classical, and safety-filtered controllers alike.

Every run:
1. Proves the deterministic Stage-1 glideslope controller before learning (aborts otherwise).
2. Trains in `single` (DummyVecEnv) or `parallel` (SubprocVecEnv/spawn) mode with
   checkpoint + periodic-eval callbacks, TensorBoard logs (rewards native plus
   `rlenv/distance_m|speed_mps|stale_sensor|action_clipped`), and resume support
   (`--resume ckpt.zip`, recorded as `resumed_from`, timesteps not reset).
3. Writes `policy_final.zip`, `trainer_config.json` (exact resolved config), and
   `training_report.json` (`schema_version: 1`): policy vs classical glideslope
   vs safety-filtered hot-approach on identical seeds, plus acceptance verdict,
   stage-1 proof, timestamps, and the full config.

Live Gazebo training is not run by the smoke tests. It requires ROS 2, Gazebo,
and an installed bridge; the live test (`test_trainer_live_gazebo.py`) skips
unless the host advertises those dependencies.
