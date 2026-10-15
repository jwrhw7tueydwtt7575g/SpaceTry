"""Stage-1 deterministic baseline proof on the Python oracle."""
from dataclasses import dataclass
import numpy as np
from spaceforge_mvp.config import MissionConfig, PhysicsConfig, RunConfig
from spaceforge_mvp.control.glideslope import glideslope_action
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from .contract import ActionLimits

@dataclass(frozen=True)
class Stage1Result:
    success: bool
    steps: int
    distance_m: float
    speed_mps: float
    fuel_used: float

def prove_stage1_approach(seed: int = 7) -> Stage1Result:
    """Run a fixed 1-D, 5 m approach; this must pass before RL training."""
    mission = MissionConfig((5.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, 300.0, 1.0, 0.00113, "v1", "APPROACH")
    run = RunConfig(mission=mission, physics=PhysicsConfig(n=mission.mean_motion, dt=mission.dt), seed=seed)
    backend = SpacecraftEnv(run_config=run)
    env = __import__("spaceforge_rl_env.env", fromlist=["SpacecraftRLEnv"]).SpacecraftRLEnv(backend=backend, action_limits=ActionLimits())
    _, _ = env.reset(seed=seed)
    initial_fuel = backend.fuel
    info = {}
    for step in range(1, env.max_steps + 1):
        acceleration = glideslope_action(backend.rel_pos, backend.rel_vel)
        action = np.concatenate((acceleration, np.zeros(3, dtype=np.float64)))
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return Stage1Result(bool(info.get("is_success", False)), step, float(info.get("distance_m", 999.0)), float(info.get("speed_mps", 999.0)), float(initial_fuel - backend.fuel))

if __name__ == "__main__":
    result = prove_stage1_approach()
    print(result)
    raise SystemExit(0 if result.success else 1)
