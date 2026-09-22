"""Evaluation script for trained Spacecraft PPO checkpoint."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from stable_baselines3 import PPO

# Ensure importability across directories
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
except ImportError:
    from env.spacecraft_env import SpacecraftEnv


def run_eval(
    checkpoint_path: str,
    num_episodes: int = 20,
    deterministic: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a trained PPO policy checkpoint on SpacecraftEnv.

    Parameters
    ----------
    checkpoint_path : str
        Path to the trained SB3 model checkpoint (.zip).
    num_episodes : int
        Number of test episodes to run (default: 20).
    deterministic : bool
        Whether to use deterministic actions (default: True).
    seed : int
        Base random seed for reproducible evaluation.

    Returns
    -------
    dict
        Dictionary containing success rate, average fuel used,
        average episode length, and safety overrides count.
    """
    model = PPO.load(checkpoint_path, seed=seed)
    env = SpacecraftEnv()

    success_count = 0
    total_fuel_consumed = 0.0
    total_steps_elapsed = 0
    total_safety_overrides = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_steps = 0
        done = False
        last_info = info
        initial_fuel = float(env.fuel)

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_steps += 1

            if last_info.get("safety_override") is not None:
                total_safety_overrides += 1

            done = terminated or truncated

        # Episode succeeds if dist < 0.5 reached before timeout or fuel depletion
        if last_info.get("is_success", False):
            success_count += 1

        fuel_used = max(0.0, initial_fuel - float(env.fuel))
        total_fuel_consumed += fuel_used
        total_steps_elapsed += ep_steps

    success_rate = float(success_count / num_episodes) if num_episodes > 0 else 0.0
    avg_fuel_used = float(total_fuel_consumed / num_episodes) if num_episodes > 0 else 0.0
    avg_episode_length = float(total_steps_elapsed / num_episodes) if num_episodes > 0 else 0.0

    report = {
        "success_rate": success_rate,
        "avg_fuel_used": avg_fuel_used,
        "average_fuel_used": avg_fuel_used,
        "avg_episode_length": avg_episode_length,
        "average_episode_length": avg_episode_length,
        "safety_overrides_count": total_safety_overrides,
        "safety_overrides": total_safety_overrides,
        "episodes_evaluated": num_episodes,
    }

    print_report_table(report, checkpoint_path)
    return report


def print_report_table(report: Dict[str, Any], checkpoint_path: str) -> None:
    """Print plain tabular report of evaluation metrics."""
    header = f"Evaluation Report for: {checkpoint_path}"
    divider = "-" * 55
    print("\n" + divider)
    print(header)
    print(divider)
    print(f"{'Metric':<35} | {'Value':<15}")
    print(divider)
    print(f"{'Success Rate':<35} | {report['success_rate'] * 100:.2f}%")
    print(f"{'Average Fuel Used':<35} | {report['avg_fuel_used']:.4f}")
    print(f"{'Average Episode Length':<35} | {report['avg_episode_length']:.2f} steps")
    print(f"{'Safety Overrides Triggered':<35} | {report['safety_overrides_count']}")
    print(divider + "\n")


def parse_args() -> Tuple[str, int, int]:
    """Parse CLI arguments supporting both positional and optional flags."""
    parser = argparse.ArgumentParser(description="Evaluate Spacecraft PPO Checkpoint")
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Path to trained model .zip checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model .zip checkpoint",
    )
    parser.add_argument(
        "--episodes",
        "--num_episodes",
        dest="episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes to run (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation (default: 42)",
    )

    args = parser.parse_args()
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        parser.error("A checkpoint path must be provided via --checkpoint or positional argument.")
    return ckpt, args.episodes, args.seed


if __name__ == "__main__":
    ckpt_path, n_episodes, eval_seed = parse_args()
    run_eval(checkpoint_path=ckpt_path, num_episodes=n_episodes, seed=eval_seed)
