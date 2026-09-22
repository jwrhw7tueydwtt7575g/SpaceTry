"""Training script for Spacecraft rendezvous PPO policy using Stable-Baselines3."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Ensure importability across different invocation working directories
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
except ImportError:
    from env.spacecraft_env import SpacecraftEnv


class RewardTrackerCallback(BaseCallback):
    """Callback that logs step rewards and tracks any NaN or Inf occurrences."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rewards: List[float] = []
        self.has_nan_or_inf: bool = False

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            for r in rewards:
                val = float(r)
                if not np.isfinite(val):
                    self.has_nan_or_inf = True
                self.rewards.append(val)
        return True


def make_env(env: Optional[SpacecraftEnv] = None) -> Monitor:
    """Instantiate and wrap SpacecraftEnv with SB3 Monitor."""
    if env is None:
        base_env = SpacecraftEnv()
    else:
        base_env = env
    return Monitor(base_env)


def train(
    total_timesteps: int = 200_000,
    save_freq: int = 1000,
    save_dir: str = "checkpoints",
    env: Optional[SpacecraftEnv] = None,
    verbose: int = 1,
    seed: int = 42,
) -> Tuple[PPO, RewardTrackerCallback]:
    """
    Train PPO policy on SpacecraftEnv.

    Parameters
    ----------
    total_timesteps : int
        Total training timesteps (default: 200,000).
    save_freq : int
        Checkpoint save frequency (default: 1000).
    save_dir : str
        Directory to save model checkpoints.
    env : SpacecraftEnv, optional
        Custom environment instance.
    verbose : int
        Verbosity level for PPO logging.
    seed : int
        Random seed for environment and PPO model (default: 42).

    Returns
    -------
    Tuple[PPO, RewardTrackerCallback]
        The trained PPO model and the callback containing reward telemetry.
    """
    os.makedirs(save_dir, exist_ok=True)
    monitored_env = make_env(env)

    # Ensure at least one periodic checkpoint triggers even for short test runs
    effective_save_freq = min(save_freq, total_timesteps)

    checkpoint_callback = CheckpointCallback(
        save_freq=effective_save_freq,
        save_path=save_dir,
        name_prefix="ppo_spacecraft",
    )
    reward_tracker = RewardTrackerCallback()
    callbacks = CallbackList([checkpoint_callback, reward_tracker])

    model = PPO(
        policy="MlpPolicy",
        env=monitored_env,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        verbose=verbose,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save final model checkpoint
    final_checkpoint = os.path.join(save_dir, "ppo_spacecraft_final.zip")
    model.save(final_checkpoint)

    return model, reward_tracker


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Spacecraft PPO Policy")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=200_000,
        help="Total environment timesteps to train (default: 200_000)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1000,
        help="Checkpoint save frequency (default: 1000)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        total_timesteps=args.total_timesteps,
        save_freq=args.save_freq,
        save_dir=args.save_dir,
        seed=args.seed,
    )
