"""Smoke test for PPO training script."""

import sys
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.training.train import train
except ImportError:
    from training.train import train


def test_train_smoke_500_steps():
    """
    Run training with total_timesteps=500 and verify:
      (a) completes without error
      (b) a checkpoint file exists
      (c) reward is never NaN/inf during training
    """
    save_dir = "checkpoints"
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
