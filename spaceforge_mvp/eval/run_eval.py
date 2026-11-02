"""Evaluation script for trained Spacecraft PPO checkpoint."""

import argparse
from typing import Any, Dict, Tuple
from stable_baselines3 import PPO

from spaceforge_mvp.control.glideslope import glideslope_action

from spaceforge_mvp.config import RunConfig
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv


def _metrics_for_controller(controller, env, seeds):
    success_count = 0
    fuel_total = 0.0
    steps_total = 0
    overrides = 0
    for episode_seed in seeds:
        obs, _ = env.reset(seed=episode_seed)
        initial_fuel = float(env.fuel)
        done = False
        steps = 0
        last_info = {}
        while not done:
            action = controller(obs, env)
            obs, _, terminated, truncated, last_info = env.step(action)
            steps += 1
            overrides += int(last_info.get("safety_override") is not None)
            done = terminated or truncated
        success_count += int(last_info.get("is_success", False))
        fuel_total += max(0.0, initial_fuel - float(env.fuel))
        steps_total += steps
    count = max(1, len(seeds))
    return {"success_rate": success_count / count, "avg_fuel_used": fuel_total / count, "avg_episode_length": steps_total / count, "safety_overrides_count": overrides}


def run_eval(
    checkpoint_path: str,
    num_episodes: int = 20,
    deterministic: bool = True,
    seed: int = 42,
    run_config: RunConfig | None = None,
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
    env = SpacecraftEnv(run_config=run_config) if run_config else SpacecraftEnv()
    seeds = [(run_config.seed if run_config else seed) + ep for ep in range(num_episodes)]
    policy_metrics = _metrics_for_controller(
        lambda obs, _env: model.predict(obs, deterministic=deterministic)[0], env, seeds
    )
    baseline_env = SpacecraftEnv(run_config=run_config) if run_config else SpacecraftEnv()
    baseline_metrics = _metrics_for_controller(
        lambda _obs, baseline: glideslope_action(
            baseline.rel_pos, baseline.rel_vel, n=baseline.physics_config.n,
            dt=baseline.physics_config.dt, max_accel=baseline.actuator_config.max_cmd_accel,
        ),
        baseline_env,
        seeds,
    )
    report = dict(policy_metrics)
    report.update({
        "average_fuel_used": report["avg_fuel_used"],
        "average_episode_length": report["avg_episode_length"],
        "safety_overrides": report["safety_overrides_count"],
        "episodes_evaluated": num_episodes,
        "fixed_seeds": seeds,
        "baseline": baseline_metrics,
        "policy": policy_metrics,
        "beats_baseline": bool(
            policy_metrics["success_rate"] > baseline_metrics["success_rate"]
            or (policy_metrics["success_rate"] == baseline_metrics["success_rate"]
                and policy_metrics["avg_fuel_used"] < baseline_metrics["avg_fuel_used"])
        ),
        "acceptance": {
            "min_success_rate": 0.85, "max_avg_fuel_used": 0.35,
            "max_safety_overrides_per_episode": 1.0,
        },
    })
    report["acceptance_passed"] = bool(
        report["success_rate"] >= report["acceptance"]["min_success_rate"]
        and report["avg_fuel_used"] <= report["acceptance"]["max_avg_fuel_used"]
        and report["safety_overrides_count"] / max(1, num_episodes) <= report["acceptance"]["max_safety_overrides_per_episode"]
    )
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
    print(f"{'Glideslope Success Rate':<35} | {report['baseline']['success_rate'] * 100:.2f}%")
    print(f"{'Beats Glideslope Baseline':<35} | {report['beats_baseline']}")
    print(f"{'Acceptance Passed':<35} | {report['acceptance_passed']}")
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
