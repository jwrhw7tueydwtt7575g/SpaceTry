"""CLI for the SpaceForge PPO/SAC trainer.

Every TrainerConfig field is settable via ``--config`` (YAML/JSON) and/or an
explicit CLI flag; CLI flags win. There are no hidden constants: run
``--print-config`` to dump the resolved configuration without training.
"""
import argparse
import json
from .config import TrainerConfig
from .trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceForge PPO/SAC trainer (default: PPO)")
    parser.add_argument("--config", help="YAML or JSON trainer configuration file")
    parser.add_argument("--algorithm", choices=("PPO", "SAC", "ppo", "sac"),
                        help="RL algorithm; PPO is the default")
    parser.add_argument("--env-mode", choices=("single", "parallel"),
                        help="single (DummyVecEnv) or parallel (SubprocVecEnv) training")
    parser.add_argument("--n-envs", type=int, help="Number of vectorized environments")
    parser.add_argument("--timesteps", type=int, help="Total training timesteps")
    parser.add_argument("--output-dir", help="Output directory for checkpoints/report")
    parser.add_argument("--scenario", help="Benchmark scenario (e.g. stage1_1d)")
    parser.add_argument("--seed", type=int, help="Base seed: train env rank r uses seed+r")
    parser.add_argument("--max-steps", type=int, help="Max steps per episode")
    parser.add_argument("--eval-episodes", type=int, help="Evaluation episodes per controller")
    parser.add_argument("--eval-seed", type=int, help="Base seed: eval episode k uses eval_seed+k")
    parser.add_argument("--eval-freq", type=int, help="Periodic EvalCallback frequency (0 disables)")
    parser.add_argument("--checkpoint-freq", type=int, help="CheckpointCallback frequency")
    parser.add_argument("--resume", help="Checkpoint .zip to resume training from")
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor in (0, 1]")
    parser.add_argument("--ppo-n-steps", type=int, help="PPO rollout steps per env")
    parser.add_argument("--ppo-batch-size", type=int, help="PPO minibatch size")
    parser.add_argument("--sac-buffer-size", type=int, help="SAC replay buffer size")
    parser.add_argument("--sac-batch-size", type=int, help="SAC minibatch size")
    parser.add_argument("--warmup-steps", type=int, help="SAC learning-starts warmup steps")
    parser.add_argument("--sensor-timeout", type=float, dest="sensor_timeout_s",
                        help="Sensor wait timeout in seconds")
    parser.add_argument("--force-limit", nargs=3, type=float, metavar=("FX", "FY", "FZ"),
                        help="Force limits in N (3 values)")
    parser.add_argument("--torque-limit", nargs=3, type=float, metavar=("TX", "TY", "TZ"),
                        help="Torque limits in N-m (3 values)")
    parser.add_argument("--acceptance-min-success", type=float,
                        help="Acceptance gate: minimum policy success rate")
    parser.add_argument("--acceptance-max-fuel", type=float,
                        help="Acceptance gate: maximum average fuel used")
    deterministic = parser.add_mutually_exclusive_group()
    deterministic.add_argument("--deterministic-eval", dest="deterministic_eval",
                               action="store_true", default=None,
                               help="Evaluate deterministically (default)")
    deterministic.add_argument("--stochastic-eval", dest="deterministic_eval",
                               action="store_false",
                               help="Evaluate stochastically")
    tensorboard = parser.add_mutually_exclusive_group()
    tensorboard.add_argument("--tensorboard", dest="tensorboard", action="store_true",
                             default=None, help="Enable TensorBoard logging")
    tensorboard.add_argument("--no-tensorboard", dest="tensorboard", action="store_false",
                             help="Disable TensorBoard logging")
    parser.add_argument("--print-config", action="store_true",
                        help="Print the resolved config and exit without training")
    return parser


def resolve_config(args: argparse.Namespace, base: dict | None = None) -> TrainerConfig:
    values: dict = dict(base or {})
    if args.config:
        values.update(TrainerConfig.from_file(args.config).to_dict())
    mapping = {
        "algorithm": args.algorithm, "env_mode": args.env_mode, "n_envs": args.n_envs,
        "total_timesteps": args.timesteps, "output_dir": args.output_dir,
        "scenario": args.scenario, "seed": args.seed, "max_steps": args.max_steps,
        "eval_episodes": args.eval_episodes, "eval_seed": args.eval_seed,
        "eval_freq": args.eval_freq, "checkpoint_freq": args.checkpoint_freq,
        "resume": args.resume, "learning_rate": args.learning_rate, "gamma": args.gamma,
        "ppo_n_steps": args.ppo_n_steps, "ppo_batch_size": args.ppo_batch_size,
        "sac_buffer_size": args.sac_buffer_size, "sac_batch_size": args.sac_batch_size,
        "warmup_steps": args.warmup_steps, "sensor_timeout_s": args.sensor_timeout_s,
        "deterministic_eval": args.deterministic_eval, "tensorboard": args.tensorboard,
        "acceptance_min_success": args.acceptance_min_success,
        "acceptance_max_fuel": args.acceptance_max_fuel,
    }
    values.update({key: value for key, value in mapping.items() if value is not None})
    if args.force_limit is not None:
        values["force_limit_n"] = tuple(float(v) for v in args.force_limit)
    if args.torque_limit is not None:
        values["torque_limit_nm"] = tuple(float(v) for v in args.torque_limit)
    return TrainerConfig.from_mapping(values)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0
    _, report = train(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
