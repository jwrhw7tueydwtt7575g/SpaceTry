"""Training script for Spacecraft rendezvous PPO policy using Stable-Baselines3."""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from spaceforge_mvp.config import MissionConfig
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv


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
    tensorboard_log: Optional[str] = None,
    env: Optional[SpacecraftEnv] = None,
    eval_freq: int = 5000,
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
    tensorboard_log : str, optional
        TensorBoard log directory path.
    env : SpacecraftEnv, optional
        Custom environment instance.
    eval_freq : int
        Evaluation callback frequency.
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
    eval_env = make_env()

    # Ensure at least one periodic checkpoint triggers even for short test runs
    effective_save_freq = min(save_freq, total_timesteps)

    checkpoint_callback = CheckpointCallback(
        save_freq=effective_save_freq,
        save_path=save_dir,
        name_prefix="ppo_spacecraft",
    )
    reward_tracker = RewardTrackerCallback()
    
    callbacks_list = [checkpoint_callback, reward_tracker]
    
    if total_timesteps >= 2000:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=save_dir,
            eval_freq=min(eval_freq, total_timesteps),
            deterministic=True,
            render=False,
        )
        callbacks_list.append(eval_callback)

    callbacks = CallbackList(callbacks_list)

    model = PPO(
        policy="MlpPolicy",
        env=monitored_env,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
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
        "--tensorboard_log",
        type=str,
        default=None,
        help="TensorBoard log directory (default: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training (default: 42)",
    )
    parser.add_argument("--start-pos", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-pos", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--start-vel", nargs=3, type=float, metavar=("VX", "VY", "VZ"))
    parser.add_argument("--target-vel", nargs=3, type=float, metavar=("VX", "VY", "VZ"))
    parser.add_argument("--fuel", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--mean-motion", type=float)
    parser.add_argument("--safety", choices=("off", "v1", "v2"), default="v1")
    parser.add_argument("--goal", default="APPROACH")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mission_values = (args.start_pos, args.target_pos, args.start_vel, args.target_vel,
                      args.fuel, args.duration, args.dt, args.mean_motion)
    if any(value is not None for value in mission_values) and not all(value is not None for value in mission_values):
        raise SystemExit("When configuring a fixed training mission, provide all position, velocity, fuel, duration, dt, and mean-motion flags.")
    training_env = None
    if all(value is not None for value in mission_values):
        if not 0.0 <= args.fuel <= 1.0 or args.duration <= 0 or args.dt <= 0 or args.mean_motion <= 0:
            raise SystemExit("fuel must be in [0, 1]; duration, dt, and mean-motion must be positive.")
        mission = MissionConfig(
            start_pos=tuple(args.start_pos), target_pos=tuple(args.target_pos),
            start_vel=tuple(args.start_vel), target_vel=tuple(args.target_vel),
            initial_fuel=args.fuel, duration=args.duration, dt=args.dt,
            mean_motion=args.mean_motion, safety_mode=args.safety, goal_type=args.goal,
        )
        training_env = SpacecraftEnv(mission=mission)
    train(
        total_timesteps=args.total_timesteps,
        save_freq=args.save_freq,
        save_dir=args.save_dir,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
        env=training_env,
    )
