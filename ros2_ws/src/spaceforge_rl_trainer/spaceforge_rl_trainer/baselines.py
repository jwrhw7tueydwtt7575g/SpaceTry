"""Policy and classical-controller evaluation on identical mock episodes.

Seed scheme (explicit, no hidden constants):
  training env rank r  -> seed = config.seed + r
  evaluation episode k -> seed = config.eval_seed + k
  stage-1 backend      -> built with the same evaluation seed, never a fixed literal.
"""
from typing import Any, Dict, Optional
import numpy as np
from spaceforge_mvp.config import MissionConfig, PhysicsConfig, RunConfig
from spaceforge_mvp.control.glideslope import glideslope_action, hot_approach_action
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from spaceforge_rl_env.contract import ActionLimits
from spaceforge_rl_env.env import SpacecraftRLEnv

# Stage-1 benchmark scenario: 5 m 1-D approach at rest, 300 s budget, 1 s step,
# LEO mean motion. Named here so YAML/CLI scenario "stage1_1d" has one definition.
STAGE1_START_POS = (5.0, 0.0, 0.0)
STAGE1_TARGET_POS = (0.0, 0.0, 0.0)
STAGE1_START_VEL = (0.0, 0.0, 0.0)
STAGE1_TARGET_VEL = (0.0, 0.0, 0.0)
STAGE1_FUEL = 1.0
STAGE1_DURATION_S = 300.0
STAGE1_DT_S = 1.0
STAGE1_MEAN_MOTION = 0.00113

#: Safety mode used for the classical baseline (glideslope never trips the gate).
CLASSICAL_SAFETY_MODE = "v1"
#: Safety mode used for the safety-filtered baseline (hot approach + active braking).
SAFETY_FILTERED_SAFETY_MODE = "v2"


def make_backend(scenario: str, safety_mode: str = "v1", seed: int = 42):
    """Build the pure-Python backend for a named scenario.

    seed is threaded into RunConfig (no fixed literal) so evaluation episodes
    are reproducible from the caller's seed.
    """
    if scenario == "stage1_1d":
        mission = MissionConfig(
            STAGE1_START_POS, STAGE1_TARGET_POS,
            STAGE1_START_VEL, STAGE1_TARGET_VEL,
            STAGE1_FUEL, STAGE1_DURATION_S, STAGE1_DT_S, STAGE1_MEAN_MOTION,
            safety_mode, "APPROACH",
        )
        run = RunConfig(mission=mission, physics=PhysicsConfig(n=mission.mean_motion, dt=mission.dt), seed=seed)
        return SpacecraftEnv(run_config=run)
    return SpacecraftEnv(safety_mode=safety_mode)


def _backend_override(transport) -> Optional[str]:
    """Latest safety-interlock verdict from the mock backend, if surfaced."""
    info = getattr(transport, "last_backend_info", None) or {}
    return info.get("safety_override")


def evaluate_controller(
    controller,
    scenario: str,
    episodes: int,
    seed: int,
    max_steps: int,
    action_limits: Optional[ActionLimits] = None,
    sensor_timeout_s: float = 0.25,
):
    """Roll out one controller over identical seeded episodes.

    controller: "classical" (glideslope), "safety_filtered" (hot approach under
    the v2 braking filter), or an SB3 model with .predict().
    """
    limits = action_limits or ActionLimits()
    successes = 0; fuel_total = 0.0; steps_total = 0; overrides = 0
    for episode in range(episodes):
        safety_mode = SAFETY_FILTERED_SAFETY_MODE if controller == "safety_filtered" else CLASSICAL_SAFETY_MODE
        backend = make_backend(scenario, safety_mode, seed=seed + episode)
        env = SpacecraftRLEnv(
            backend=backend, max_steps=max_steps,
            action_limits=limits, sensor_timeout_s=sensor_timeout_s,
        )
        obs, _ = env.reset(seed=seed + episode)
        initial_fuel = backend.fuel; done = False; last_info = {}; steps = 0
        while not done:
            if controller == "classical":
                action = np.concatenate((glideslope_action(backend.rel_pos, backend.rel_vel), np.zeros(3)))
            elif controller == "safety_filtered":
                action = np.concatenate((hot_approach_action(backend.rel_pos, backend.rel_vel), np.zeros(3)))
            else:
                action = controller.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, last_info = env.step(action)
            steps += 1; done = terminated or truncated
            if last_info.get("safety_override") is not None or _backend_override(env.transport) is not None:
                overrides += 1
        successes += int(last_info.get("is_success", False))
        fuel_total += max(0.0, initial_fuel - backend.fuel); steps_total += steps
    count = max(1, episodes)
    return {"success_rate": successes / count, "avg_fuel_used": fuel_total / count, "avg_episode_length": steps_total / count, "safety_overrides": overrides, "episodes": episodes, "seed": seed}


def compare_policy(
    model,
    scenario: str,
    episodes: int,
    seed: int,
    max_steps: int,
    acceptance: Dict[str, Any],
    action_limits: Optional[ActionLimits] = None,
    sensor_timeout_s: float = 0.25,
):
    """Evaluate the learned policy, the classical baseline, and the
    safety-filtered baseline on identical seeds; never train without this."""
    kwargs = {"action_limits": action_limits, "sensor_timeout_s": sensor_timeout_s}
    policy = evaluate_controller(model, scenario, episodes, seed, max_steps, **kwargs)
    classical = evaluate_controller("classical", scenario, episodes, seed, max_steps, **kwargs)
    safety = evaluate_controller("safety_filtered", scenario, episodes, seed, max_steps, **kwargs)
    policy["beats_classical"] = policy["success_rate"] > classical["success_rate"] or (policy["success_rate"] == classical["success_rate"] and policy["avg_fuel_used"] < classical["avg_fuel_used"])
    policy["beats_safety_filtered"] = policy["success_rate"] > safety["success_rate"] or (policy["success_rate"] == safety["success_rate"] and policy["avg_fuel_used"] < safety["avg_fuel_used"])
    policy["acceptance_passed"] = policy["success_rate"] >= acceptance["min_success_rate"] and policy["avg_fuel_used"] <= acceptance["max_fuel_used"]
    return {"policy": policy, "classical": classical, "safety_filtered": safety, "acceptance": acceptance}
