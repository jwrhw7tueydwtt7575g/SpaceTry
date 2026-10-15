"""Explicit, serializable rl_trainer configuration."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import json

@dataclass
class TrainerConfig:
    algorithm: str = "PPO"
    env_mode: str = "single"
    n_envs: int = 1
    total_timesteps: int = 10000
    output_dir: str = "runs/rl_trainer"
    scenario: str = "stage1_1d"
    seed: int = 42
    eval_episodes: int = 20
    eval_seed: int = 1000
    deterministic_eval: bool = True
    eval_freq: int = 2000
    checkpoint_freq: int = 2000
    tensorboard: bool = True
    resume: str = ""
    max_steps: int = 300
    learning_rate: float = 0.0003
    gamma: float = 0.99
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    sac_buffer_size: int = 100000
    sac_batch_size: int = 256
    warmup_steps: int = 1000
    sensor_timeout_s: float = 0.25
    force_limit_n: tuple[float, float, float] = (1.0, 1.0, 1.0)
    torque_limit_nm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    acceptance_min_success: float = 0.85
    acceptance_max_fuel: float = 0.35

    def __post_init__(self):
        self.algorithm = self.algorithm.upper()
        if self.algorithm not in {"PPO", "SAC"}: raise ValueError("algorithm must be PPO or SAC")
        if self.env_mode not in {"single", "parallel"}: raise ValueError("env_mode must be single or parallel")
        if self.env_mode == "single": self.n_envs = 1
        if min(self.n_envs, self.total_timesteps, self.eval_episodes, self.max_steps, self.checkpoint_freq) < 1 or self.eval_freq < 0: raise ValueError("counts must be positive and eval_freq may be zero")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1: raise ValueError("learning_rate positive and gamma in (0,1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]):
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(mapping) - allowed
        if unknown: raise ValueError("unknown trainer configuration keys: " + ", ".join(sorted(unknown)))
        values = {key: value for key, value in mapping.items() if key in allowed}
        for key in ("force_limit_n", "torque_limit_nm"):
            if key in values: values[key] = tuple(float(v) for v in values[key])
        return cls(**values)

    @classmethod
    def from_file(cls, path: str):
        text = Path(path).read_text()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc: raise RuntimeError("PyYAML is required for YAML trainer configs") from exc
            data = yaml.safe_load(text) or {}
        else: data = json.loads(text)
        return cls.from_mapping(data)
