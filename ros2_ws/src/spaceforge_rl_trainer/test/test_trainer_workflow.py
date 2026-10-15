"""Workflow smoke tests for the Blueprint rl_trainer (mock env only).

Live Gazebo training is never exercised here; see test_trainer_live_gazebo.py,
which skips unless the host advertises ROS 2 + Gazebo dependencies.
"""
import json
from pathlib import Path

from spaceforge_rl_trainer.baselines import evaluate_controller, make_backend
from spaceforge_rl_trainer.cli import build_parser, resolve_config
from spaceforge_rl_trainer.config import TrainerConfig
from spaceforge_rl_trainer.trainer import build_vec_env, train


def _tiny(**overrides):
    values = dict(
        total_timesteps=64, ppo_n_steps=32, ppo_batch_size=16,
        eval_freq=0, checkpoint_freq=64, eval_episodes=1, max_steps=50,
        tensorboard=False,
    )
    values.update(overrides)
    return TrainerConfig(output_dir=str(overrides.get("output_dir", "runs/test_tiny")), **{k: v for k, v in values.items() if k != "output_dir"})


def test_backends_are_seeded_not_hardcoded():
    first = make_backend("stage1_1d", seed=7)
    second = make_backend("stage1_1d", seed=8)
    assert first.run_config.seed == 7
    assert second.run_config.seed == 8


def test_seed_propagation_is_reproducible():
    kwargs = dict(scenario="stage1_1d", episodes=1, seed=11, max_steps=50)
    run_a = evaluate_controller("classical", **kwargs)
    run_b = evaluate_controller("classical", **kwargs)
    assert run_a == run_b


def test_parallel_env_mode_builds_and_resets():
    config = TrainerConfig(env_mode="parallel", n_envs=2, total_timesteps=8, tensorboard=False)
    env = build_vec_env(config)
    obs = env.reset()
    assert obs.shape[0] == 2
    env.close()


def test_resume_continues_from_checkpoint(tmp_path):
    first_dir = tmp_path / "first"
    config = _tiny(output_dir=str(first_dir))
    _, _ = train(TrainerConfig(**{**config.to_dict(), "output_dir": str(first_dir)}))
    checkpoint = first_dir / "policy_final.zip"
    assert checkpoint.is_file()
    second_dir = tmp_path / "second"
    resumed, report = train(TrainerConfig(**{**config.to_dict(), "output_dir": str(second_dir), "resume": str(checkpoint)}))
    assert resumed is not None
    assert report["resumed_from"] == str(checkpoint)
    assert (second_dir / "policy_final.zip").is_file()


def test_report_schema_and_config_snapshot(tmp_path):
    out = tmp_path / "report"
    _, report = train(_tiny(output_dir=str(out)))
    assert report["schema_version"] == 1
    assert report["algorithm"] == "PPO"
    assert report["resumed_from"] is None
    assert set(report) >= {"policy", "classical", "safety_filtered", "acceptance",
                           "checkpoint", "config", "stage1_proof", "created_utc"}
    assert set(report["policy"]) >= {"success_rate", "avg_fuel_used", "beats_classical",
                                     "beats_safety_filtered", "acceptance_passed"}
    saved = json.loads((out / "training_report.json").read_text())
    assert saved["schema_version"] == 1
    snapshot = json.loads((out / "trainer_config.json").read_text())
    assert snapshot["algorithm"] == "PPO" and snapshot["seed"] == 42
    assert Path(report["checkpoint"]).is_file()


def test_cli_exposes_all_settings_without_training(tmp_path, capsys):
    parser = build_parser()
    args = parser.parse_args([
        "--algorithm", "SAC", "--env-mode", "single", "--timesteps", "10",
        "--output-dir", str(tmp_path), "--scenario", "stage1_1d", "--seed", "7",
        "--max-steps", "50", "--eval-episodes", "2", "--eval-seed", "9",
        "--eval-freq", "0", "--checkpoint-freq", "10", "--learning-rate", "0.001",
        "--gamma", "0.95", "--sac-buffer-size", "1000", "--sac-batch-size", "32",
        "--warmup-steps", "10", "--sensor-timeout", "0.1",
        "--force-limit", "1", "1", "1", "--torque-limit", "0.5", "0.5", "0.5",
        "--stochastic-eval", "--no-tensorboard", "--print-config",
    ])
    from spaceforge_rl_trainer.cli import main
    assert main([
        "--algorithm", "SAC", "--timesteps", "10", "--output-dir", str(tmp_path),
        "--max-steps", "50", "--print-config",
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["algorithm"] == "SAC" and printed["total_timesteps"] == 10
    config = resolve_config(args)
    assert config.algorithm == "SAC" and config.deterministic_eval is False
    assert config.force_limit_n == (1.0, 1.0, 1.0) and config.torque_limit_nm == (0.5, 0.5, 0.5)
    assert config.eval_episodes == 2 and config.eval_seed == 9


def test_yaml_config_loads_both_algorithms():
    here = Path(__file__).resolve().parent.parent / "config"
    ppo = TrainerConfig.from_file(str(here / "stage1_ppo.yaml"))
    sac = TrainerConfig.from_file(str(here / "stage1_sac.yaml"))
    assert ppo.algorithm == "PPO" and sac.algorithm == "SAC"
    assert ppo.total_timesteps == sac.total_timesteps == 10000
