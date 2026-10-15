"""Stable-Baselines3 PPO/SAC training orchestration.

Default algorithm is PPO; SAC is an explicitly selectable alternative via
``TrainerConfig(algorithm="SAC")`` / ``--algorithm SAC`` / YAML.

Seed scheme (no hidden constants):
  train env rank r -> seed = config.seed + r        (vec-env construction + SB3 model seed)
  eval env          -> seed = config.seed + EVAL_RANK_OFFSET
  final report ep k -> seed = config.eval_seed + k (policy, classical, safety-filtered alike)
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import json
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from spaceforge_rl_env.classical import prove_stage1_approach
from spaceforge_rl_env.contract import ActionLimits
from spaceforge_rl_env.env import SpacecraftRLEnv
from .baselines import compare_policy, make_backend
from .config import TrainerConfig

#: Rank offset for the periodic-evaluation env so its RNG stream never overlaps
#: the training envs (ranks 0..n_envs-1).
EVAL_RANK_OFFSET = 100_000

#: Machine-readable report schema version.
REPORT_SCHEMA_VERSION = 1


def action_limits_from_config(config: TrainerConfig) -> ActionLimits:
    """Build actuator limits from explicit config (never module literals)."""
    return ActionLimits(force_n=tuple(config.force_limit_n), torque_nm=tuple(config.torque_limit_nm))


def make_env(config: TrainerConfig, rank: int = 0):
    limits = action_limits_from_config(config)
    seed = config.seed + rank

    def factory():
        env = SpacecraftRLEnv(
            backend=make_backend(config.scenario, seed=seed),
            max_steps=config.max_steps,
            action_limits=limits,
            sensor_timeout_s=config.sensor_timeout_s,
        )
        env.reset(seed=seed)
        return env

    return factory


def build_vec_env(config: TrainerConfig):
    factories = [make_env(config, rank) for rank in range(config.n_envs)]
    if config.env_mode == "parallel" and config.n_envs > 1:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def _algorithm(config: TrainerConfig, env):
    common = dict(
        policy="MlpPolicy", env=env, learning_rate=config.learning_rate,
        gamma=config.gamma, seed=config.seed, verbose=0,
        tensorboard_log=(str(Path(config.output_dir) / "tensorboard") if config.tensorboard else None),
    )
    if config.algorithm == "PPO":
        return PPO(n_steps=config.ppo_n_steps, batch_size=config.ppo_batch_size, **common)
    return SAC(
        batch_size=config.sac_batch_size, buffer_size=config.sac_buffer_size,
        learning_starts=config.warmup_steps, **common,
    )


class RLEnvMetricsCallback(BaseCallback):
    """Log transport-level telemetry to TensorBoard (rewards are SB3-native)."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict):
                if "distance_m" in info:
                    self.logger.record("rlenv/distance_m", float(info["distance_m"]))
                if "speed_mps" in info:
                    self.logger.record("rlenv/speed_mps", float(info["speed_mps"]))
                self.logger.record("rlenv/stale_sensor", float(bool(info.get("stale_sensor", False))))
                self.logger.record("rlenv/action_clipped", float(bool(info.get("action_clipped", False))))
        return True


def train(config: TrainerConfig):
    # Gate 1: the deterministic classical proof must pass before any learning.
    proof = prove_stage1_approach(config.seed)
    if not proof.success:
        raise RuntimeError(f"Stage-1 classical proof failed: {proof}")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "trainer_config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    env = build_vec_env(config)
    eval_env = DummyVecEnv([make_env(config, EVAL_RANK_OFFSET)])
    resumed_from: Optional[str] = None
    if config.resume:
        model_cls = PPO if config.algorithm == "PPO" else SAC
        model = model_cls.load(config.resume, env=env, seed=config.seed)
        resumed_from = str(config.resume)
    else:
        model = _algorithm(config, env)
    callbacks: List = [
        CheckpointCallback(
            save_freq=max(1, config.checkpoint_freq // config.n_envs),
            save_path=str(output / "checkpoints"),
            name_prefix=config.algorithm.lower(),
        ),
        RLEnvMetricsCallback(),
    ]
    if config.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(output / "best"),
                log_path=str(output / "eval"),
                eval_freq=max(1, config.eval_freq // config.n_envs),
                n_eval_episodes=config.eval_episodes,
                deterministic=config.deterministic_eval,
                render=False,
            )
        )
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=not bool(config.resume),
    )
    checkpoint = output / "policy_final.zip"
    model.save(str(checkpoint))
    # Gate 2: every trained policy is compared against the classical baseline
    # AND the safety-filtered baseline on identical seeds.
    limits = action_limits_from_config(config)
    report = compare_policy(
        model, config.scenario, config.eval_episodes, config.eval_seed, config.max_steps,
        {"min_success_rate": config.acceptance_min_success, "max_fuel_used": config.acceptance_max_fuel},
        action_limits=limits, sensor_timeout_s=config.sensor_timeout_s,
    )
    report.update({
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": config.algorithm,
        "env_mode": config.env_mode,
        "n_envs": config.n_envs,
        "total_timesteps": int(config.total_timesteps),
        "seed": int(config.seed),
        "resumed_from": resumed_from,
        "checkpoint": str(checkpoint),
        "config": config.to_dict(),
        "stage1_proof": proof.__dict__,
    })
    (output / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    env.close()
    eval_env.close()
    return model, report
