"""Command-line mission simulator.

Example::

    python3 -m spaceforge_mvp.simulate --controller glideslope \
      --start-pos 12 -7 3.5 --target-pos 2 1 0 \
      --start-vel 0 0 0 --target-vel 0 0 0 --fuel 0.8 \
      --duration 120 --dt 1 --mean-motion 0.00113 --safety v2
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from spaceforge_mvp.config import MissionConfig
from spaceforge_mvp.control.glideslope import glideslope_action, hot_approach_action
from spaceforge_mvp.viz.record import record_episode, rl_controller


def _vec(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} requires exactly three values")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one SpaceForge mission simulation")
    parser.add_argument("--controller", choices=("rl", "glideslope", "hot"), required=True)
    parser.add_argument("--checkpoint", help="PPO .zip checkpoint; required for --controller rl")
    parser.add_argument("--start-pos", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-pos", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--start-vel", nargs=3, type=float, required=True, metavar=("VX", "VY", "VZ"))
    parser.add_argument("--target-vel", nargs=3, type=float, required=True, metavar=("VX", "VY", "VZ"))
    parser.add_argument("--fuel", type=float, required=True, help="Initial fuel fraction [0, 1]")
    parser.add_argument("--duration", type=float, required=True, help="Mission duration in seconds")
    parser.add_argument("--dt", type=float, required=True, help="Simulation step in seconds")
    parser.add_argument("--mean-motion", type=float, required=True, help="CW mean motion in rad/s")
    parser.add_argument("--safety", choices=("off", "v1", "v2"), required=True)
    parser.add_argument("--goal", default="APPROACH", help="Mission safety goal type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nondeterministic", action="store_true", help="Use stochastic RL actions")
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.controller == "rl" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --controller rl is selected")
    if not 0.0 <= args.fuel <= 1.0:
        raise ValueError("--fuel must be between 0 and 1")
    if args.duration <= 0 or args.dt <= 0 or args.mean_motion <= 0:
        raise ValueError("--duration, --dt, and --mean-motion must be positive")

    mission = MissionConfig(
        start_pos=_vec(args.start_pos, "--start-pos"),
        target_pos=_vec(args.target_pos, "--target-pos"),
        start_vel=_vec(args.start_vel, "--start-vel"),
        target_vel=_vec(args.target_vel, "--target-vel"),
        initial_fuel=float(args.fuel),
        duration=float(args.duration),
        dt=float(args.dt),
        mean_motion=float(args.mean_motion),
        safety_mode=args.safety,
        goal_type=args.goal,
    )

    if args.controller == "rl":
        controller: Callable = rl_controller(args.checkpoint, deterministic=not args.nondeterministic)
    elif args.controller == "hot":
        controller = hot_approach_action
    else:
        controller = glideslope_action

    episode = record_episode(
        controller,
        seed=args.seed,
        label=args.controller,
        mission=mission,
        safety=args.safety,
        annotate_continuous=True,
        goal_type=args.goal,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mission.json").write_text(json.dumps({
        "controller": args.controller,
        "checkpoint": args.checkpoint,
        "start_pos": mission.start_pos,
        "target_pos": mission.target_pos,
        "start_vel": mission.start_vel,
        "target_vel": mission.target_vel,
        "relative_pos": mission.relative_pos,
        "relative_vel": mission.relative_vel,
        "initial_fuel": mission.initial_fuel,
        "duration": mission.duration,
        "dt": mission.dt,
        "mean_motion": mission.mean_motion,
        "safety": mission.safety_mode,
        "goal": mission.goal_type,
        "seed": args.seed,
    }, indent=2))
    (args.output_dir / "telemetry.json").write_text(json.dumps(episode, indent=2))
    (args.output_dir / "metrics.json").write_text(json.dumps(episode["summary"], indent=2))
    print(json.dumps(episode["summary"], indent=2))
    return episode


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
