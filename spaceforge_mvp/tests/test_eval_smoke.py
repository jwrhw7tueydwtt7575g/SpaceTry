"""Smoke test for policy evaluation script."""

import sys
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.eval.run_eval import run_eval
    from spaceforge_mvp.training.train import train
except ImportError:
    from eval.run_eval import run_eval
    from training.train import train


def test_run_eval_smoke_end_to_end(tmp_path, capsys):
    """
    Run run_eval with a 500-timestep smoke checkpoint trained into tmp_path over 3 episodes.
    Asserts:
      - completes end-to-end without error
      - report contains all 4 metrics: success_rate, avg_fuel_used,
        avg_episode_length, safety_overrides_count
      - plain table is printed to stdout
    """
    save_dir = str(tmp_path / "checkpoints")
    train(total_timesteps=500, save_freq=500, save_dir=save_dir, verbose=0)
    ckpt_path = Path(save_dir) / "ppo_spacecraft_final.zip"
    if not ckpt_path.exists():
        ckpt_path = Path(save_dir) / "ppo_spacecraft_500_steps.zip"

    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} could not be created in {save_dir}."

    # Run evaluation across 3 episodes
    report = run_eval(checkpoint_path=str(ckpt_path), num_episodes=3)

    # Assert all 4 metrics are present in the report
    assert "success_rate" in report
    assert "avg_fuel_used" in report
    assert "avg_episode_length" in report
    assert "safety_overrides_count" in report

    # Check metric values
    assert 0.0 <= report["success_rate"] <= 1.0
    assert report["avg_fuel_used"] >= 0.0
    assert report["avg_episode_length"] > 0.0
    assert isinstance(report["safety_overrides_count"], (int, np.integer))

    # Check printed report table in stdout
    captured = capsys.readouterr()
    assert "Evaluation Report" in captured.out
    assert "Success Rate" in captured.out
    assert "Average Fuel Used" in captured.out
    assert "Average Episode Length" in captured.out
    assert "Safety Overrides Triggered" in captured.out
