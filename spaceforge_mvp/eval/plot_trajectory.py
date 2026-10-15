"""Trajectory visualization script for Spacecraft Env policy evaluation."""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from spaceforge_mvp.config import ENV_CONFIG, RunConfig
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv

# Set matplotlib non-interactive backend for headless execution
plt.switch_backend("Agg")


def generate_trajectory_data(
    checkpoint_path: str,
    seed: int = 42,
    max_steps: int = ENV_CONFIG.max_steps,
    run_config: RunConfig | None = None,
) -> Tuple[List[np.ndarray], bool, float, float]:
    """
    Run single episode evaluation recording relative 2D positions (X-radial vs Y-in-track).

    Returns
    -------
    Tuple[List[np.ndarray], bool, float, float]
        (trajectory_positions, is_success, final_dist, final_fuel)
    """
    model = PPO.load(checkpoint_path, seed=seed)
    env = SpacecraftEnv(run_config=run_config) if run_config else SpacecraftEnv(max_steps=max_steps)

    obs, info = env.reset(seed=seed)
    trajectory: List[np.ndarray] = [env.rel_pos[0:2].copy()]
    done = False
    last_info = info

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, last_info = env.step(action)
        trajectory.append(env.rel_pos[0:2].copy())
        done = terminated or truncated

    final_dist = float(np.linalg.norm(env.rel_pos))
    final_fuel = float(env.fuel)
    is_success = bool(last_info.get("is_success", False))

    return trajectory, is_success, final_dist, final_fuel


def plot_trajectory(
    checkpoint_path: str,
    output_path: str = "eval/trajectory.png",
    seed: int = 42,
    run_config: RunConfig | None = None,
) -> None:
    """Generate 2D matplotlib relative trajectory plot and save to disk."""
    trajectory, is_success, final_dist, final_fuel = generate_trajectory_data(
        checkpoint_path=checkpoint_path,
        seed=seed,
        run_config=run_config,
    )

    traj_arr = np.array(trajectory)  # shape (N, 2)
    x_coords = traj_arr[:, 0]
    y_coords = traj_arr[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # Plot 0.5m success radius target circle
    target_circle = plt.Circle(
        (0.0, 0.0),
        run_config.environment.docking_success_dist if run_config else ENV_CONFIG.docking_success_dist,
        color="green",
        fill=False,
        linestyle="--",
        linewidth=2,
        label=f"Success Radius ({run_config.environment.docking_success_dist if run_config else ENV_CONFIG.docking_success_dist}m)",
    )
    ax.add_patch(target_circle)

    # Target origin marker
    ax.plot(0.0, 0.0, "k*", markersize=14, label="Target (0, 0)")

    # Trajectory path line
    ax.plot(
        x_coords,
        y_coords,
        color="#1f77b4",
        linewidth=2.0,
        label="Trajectory (x vs y)",
    )

    # Start position marker
    ax.plot(
        x_coords[0],
        y_coords[0],
        "go",
        markersize=10,
        label=f"Start ({x_coords[0]:.1f}, {y_coords[0]:.1f})m",
    )

    # End position marker
    ax.plot(
        x_coords[-1],
        y_coords[-1],
        "rX",
        markersize=12,
        label=f"End ({x_coords[-1]:.1f}, {y_coords[-1]:.1f})m",
    )

    ax.set_xlabel("Relative X - Radial (m)", fontsize=11)
    ax.set_ylabel("Relative Y - In-Track (m)", fontsize=11)
    ax.set_title("Spacecraft Rendezvous Trajectory (CW Relative Frame)", fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    if str(output_dir) and str(output_dir) != ".":
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Trajectory plot saved to: {output_path}")


def parse_args() -> Tuple[str, str, int]:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Spacecraft Trajectory")
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
        "--output",
        type=str,
        default="eval/trajectory.png",
        help="Output image path (default: eval/trajectory.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trajectory episode (default: 42)",
    )

    args = parser.parse_args()
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        parser.error("A checkpoint path must be provided via --checkpoint or positional argument.")
    return ckpt, args.output, args.seed


if __name__ == "__main__":
    ckpt_path, out_path, traj_seed = parse_args()
    plot_trajectory(checkpoint_path=ckpt_path, output_path=out_path, seed=traj_seed)
