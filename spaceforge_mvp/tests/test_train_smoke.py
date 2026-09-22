"""Smoke test for PPO training script."""

from pathlib import Path
import numpy as np
from spaceforge_mvp.training.train import train


def test_train_smoke_500_steps(tmp_path):
    """
    Run training with total_timesteps=500 and verify:
      (a) completes without error
      (b) a checkpoint file exists in tmp_path
      (c) reward is never NaN/inf during training
    """
    save_dir = str(tmp_path / "checkpoints")
    model, reward_tracker = train(
        total_timesteps=500,
        save_freq=500,
        save_dir=save_dir,
        verbose=0,
    )

    # (a) completes without error
    assert model is not None

    # (b) a checkpoint file exists
    checkpoint_files = list(Path(save_dir).glob("*.zip"))
    assert len(checkpoint_files) > 0, f"No checkpoint file found in {save_dir}"

    # (c) reward is never NaN/inf during training
    assert len(reward_tracker.rewards) > 0, "No rewards were recorded during training"
    assert not reward_tracker.has_nan_or_inf, "Found NaN or Inf in tracked rewards"
    for r in reward_tracker.rewards:
        assert np.isfinite(r), f"Encountered non-finite reward value: {r}"
