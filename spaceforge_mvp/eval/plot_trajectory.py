"""Plot 2D trajectory of spacecraft relative rendezvous motion."""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

# Ensure importability across execution contexts
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
except ImportError:
    from env.spacecraft_env import SpacecraftEnv


def plot_trajectory(
    checkpoint_path: str = "checkpoints/ppo_spacecraft_final.zip",
    output_path: str = "eval/trajectory.png",
    seed: int = 42,
    deterministic: bool = True,
) -> str:
    """
    Run one episode with a given checkpoint, record rel_pos at every step,
    and save a matplotlib 2D plot (x vs y) with start/end markers and a dashed
    circle at the 0.5m success radius.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved SB3 .zip checkpoint.
    output_path : str
        Destination path for saving the PNG plot.
    seed : int
        Environment initial state random seed.
    deterministic : bool
        Whether to predict actions deterministically.

    Returns
    -------
    str
        Path to the saved plot image.
    """
    model = PPO.load(checkpoint_path)
    env = SpacecraftEnv()

    obs, info = env.reset(seed=seed)
    trajectory: List[np.ndarray] = [env.rel_pos.copy()]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(env.rel_pos.copy())
        done = terminated or truncated

    traj_arr = np.array(trajectory)
    xs = traj_arr[:, 0]
    ys = traj_arr[:, 1]

    # Create 2D plot (x vs y)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Trajectory path
    ax.plot(xs, ys, color="#1f77b4", linewidth=1.8, label="Trajectory (x vs y)")

    # Start and End markers
    ax.scatter(xs[0], ys[0], color="#2ca02c", marker="o", s=90, zorder=5, label=f"Start ({xs[0]:.1f}, {ys[0]:.1f})m")
    ax.scatter(xs[-1], ys[-1], color="#d62728", marker="X", s=100, zorder=5, label=f"End ({xs[-1]:.1f}, {ys[-1]:.1f})m")

    # Target marker at relative origin (0, 0)
    ax.scatter(0, 0, color="black", marker="*", s=140, zorder=6, label="Target (0, 0)")

    # Dashed circle at the 0.5m success radius
    success_circle = plt.Circle(
        (0.0, 0.0),
        0.5,
        color="#2ca02c",
        fill=False,
        linestyle="--",
        linewidth=2.0,
        label="Success Radius (0.5m)",
    )
    ax.add_patch(success_circle)

    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("Relative X - Radial (m)")
    ax.set_ylabel("Relative Y - In-Track (m)")
    ax.set_title("Spacecraft Rendezvous Trajectory (CW Relative Frame)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    # Ensure target output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Trajectory plot saved to: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Spacecraft Rendezvous Trajectory")
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ppo_spacecraft_final.zip",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/trajectory.png",
        help="Output path for trajectory plot (default: eval/trajectory.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Environment initial seed (default: 42)",
    )
    args = parser.parse_args()
    ckpt = args.checkpoint_pos or args.checkpoint
    return ckpt, args.output, args.seed


if __name__ == "__main__":
    ckpt_file, out_file, rand_seed = parse_args()
    plot_trajectory(checkpoint_path=ckpt_file, output_path=out_file, seed=rand_seed)
