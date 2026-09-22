"""Full pipeline end-to-end integration test."""

import os
import sys
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.training.train import train
    from spaceforge_mvp.eval.run_eval import run_eval
    from spaceforge_mvp.eval.plot_trajectory import plot_trajectory
except ImportError:
    from training.train import train
    from eval.run_eval import run_eval
    from eval.plot_trajectory import plot_trajectory


def test_full_pipeline_integration_smoke(tmp_path):
    """
    End-to-end integration check:
      1. Train for 1000 timesteps (fresh)
      2. Run eval/run_eval.py against that checkpoint for 5 episodes
      3. Run eval/plot_trajectory.py once
      4. Assert:
         - no exceptions anywhere in the chain
         - the eval report has all 4 metrics
         - trajectory.png is produced and non-empty
    """
    save_dir = str(tmp_path / "integration_ckpts")
    output_png = str(tmp_path / "eval" / "trajectory.png")

    # 1. Train fresh model for 1000 timesteps
    model, reward_tracker = train(
        total_timesteps=1000,
        save_freq=1000,
        save_dir=save_dir,
        verbose=1,
    )
    assert model is not None, "Model failed to train"

    checkpoint_file = Path(save_dir) / "ppo_spacecraft_final.zip"
    if not checkpoint_file.exists():
        checkpoint_file = Path(save_dir) / "ppo_spacecraft_1000_steps.zip"
    assert checkpoint_file.exists(), f"Expected checkpoint file not found at {checkpoint_file}"

    # 2. Run evaluation against checkpoint for 5 episodes
    report = run_eval(
        checkpoint_path=str(checkpoint_file),
        num_episodes=5,
    )

    # Assert eval report has all 4 metrics
    assert "success_rate" in report, "Metric 'success_rate' missing from report"
    assert "avg_fuel_used" in report, "Metric 'avg_fuel_used' missing from report"
    assert "avg_episode_length" in report, "Metric 'avg_episode_length' missing from report"
    assert "safety_overrides_count" in report, "Metric 'safety_overrides_count' missing from report"

    assert 0.0 <= report["success_rate"] <= 1.0
    assert report["avg_fuel_used"] >= 0.0
    assert report["avg_episode_length"] > 0.0
    assert report["safety_overrides_count"] >= 0

    # 3. Run trajectory plotting once
    saved_png_path = plot_trajectory(
        checkpoint_path=str(checkpoint_file),
        output_path=output_png,
        seed=42,
    )

    # 4. Assert trajectory.png is produced and non-empty
    png_file = Path(saved_png_path)
    assert png_file.exists(), f"Expected output PNG {png_file} does not exist"
    assert png_file.stat().st_size > 0, f"Expected output PNG {png_file} to be non-empty"
