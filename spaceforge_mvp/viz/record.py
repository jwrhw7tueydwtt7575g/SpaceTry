"""Episode recorder — turns any controller into a replayable telemetry trace.

Everything the visualisation layer needs comes from here, so the videos and the
dashboard are driven by the same numbers a test would assert on: no hand-drawn
motion, no mocked telemetry.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from spaceforge_mvp.config import ENV_CONFIG, PHYSICS_CONFIG, SAFETY_CONFIG, MissionConfig, RunConfig
from spaceforge_mvp.core.cw_dynamics import cw_step
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv

Controller = Callable[[np.ndarray, np.ndarray], np.ndarray]

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


def _frame(
    t: float,
    pos: np.ndarray,
    vel: np.ndarray,
    cmd: np.ndarray,
    applied: np.ndarray,
    override: Optional[str],
    dist: float,
    speed: float,
    fuel: float,
    dv: float,
) -> Dict[str, Any]:
    return {
        "t": float(t),
        "pos": [round(float(v), 6) for v in pos],
        "vel": [round(float(v), 6) for v in vel],
        "cmd": [round(float(v), 4) for v in cmd],
        "applied": [round(float(v), 4) for v in applied],
        "override": override,
        "dist": round(float(dist), 6),
        "speed": round(float(speed), 6),
        "fuel": round(float(fuel), 6),
        "dv": round(float(dv), 5),
    }


def replay_continuous_min(
    frames: List[Dict[str, Any]],
    sub_dt: float = 0.01,
    interval_dt: float = PHYSICS_CONFIG.dt,
    mean_motion: float = PHYSICS_CONFIG.n,
    max_cmd_accel: float = 1.0,
) -> Tuple[float, float]:
    """True minimum range, replaying the recorded commands at sub-sample resolution.

    The environment samples at 1 Hz, so a metre-per-second fly-through can step
    straight over the 0.5 m capture bubble without the sampler ever registering
    it. Holding each recorded (post-interlock) command over its interval and
    propagating at ``sub_dt`` recovers the path the vehicle actually flew and
    reports the closest range and the speed at that moment.
    """
    pos = np.asarray(frames[0]["pos"], dtype=np.float64).copy()
    vel = np.asarray(frames[0]["vel"], dtype=np.float64).copy()
    best_dist = float(np.linalg.norm(pos))
    best_speed = float(np.linalg.norm(vel))
    n_sub = max(1, int(round(interval_dt / sub_dt)))

    for f in frames[1:]:
        applied = np.asarray(f["applied"], dtype=np.float64)
        for _ in range(n_sub):
            pos, vel = cw_step(pos, vel, applied * max_cmd_accel, dt=sub_dt, n=mean_motion)
            d = float(np.linalg.norm(pos))
            if d < best_dist:
                best_dist, best_speed = d, float(np.linalg.norm(vel))
    return best_dist, best_speed


def record_episode(
    controller: Controller,
    seed: int = 42,
    label: str = "episode",
    initial: Optional[Tuple[Any, Any]] = None,
    safety: str = "v1",
    max_steps: int = ENV_CONFIG.max_steps,
    annotate_continuous: bool = False,
    goal_type: str = "APPROACH",
    mission: Optional[MissionConfig] = None,
    run_config: Optional[RunConfig] = None,
) -> Dict[str, Any]:
    """Roll out one episode and return a full telemetry trace.

    Parameters
    ----------
    controller : callable(rel_pos, rel_vel) -> action
        Any guidance law (classical or a wrapped SB3 policy).
    initial : (pos, vel), optional
        Override the random reset with a fixed starting state.
    safety : {"v1", "v2", "off"}
        "v1"  — interlock vetoes by zeroing the thrust (the MVP behaviour).
        "v2"  — interlock vetoes by substituting certified braking (see control.rta).
        "off" — command goes straight to the thrusters, the counterfactual.
    annotate_continuous : bool
        Also compute the continuous-time closest approach (fine sub-stepping).
    """
    if run_config is not None:
        mission = run_config.mission
        env = SpacecraftEnv(run_config=run_config, safety_mode=safety)
    elif mission is None:
        if initial is None:
            env = SpacecraftEnv(max_steps=max_steps, safety_mode=safety)
        else:
            mission = MissionConfig(
                start_pos=tuple(np.asarray(initial[0], dtype=float)),
                target_pos=(0.0, 0.0, 0.0),
                start_vel=tuple(np.asarray(initial[1], dtype=float)),
                target_vel=(0.0, 0.0, 0.0),
                initial_fuel=1.0,
                duration=max_steps * PHYSICS_CONFIG.dt,
                dt=PHYSICS_CONFIG.dt,
                mean_motion=PHYSICS_CONFIG.n,
                safety_mode=safety,
                goal_type=goal_type,
            )
            env = SpacecraftEnv(mission=mission)
    elif run_config is None:
        env = SpacecraftEnv(mission=mission, safety_mode=safety)
    env.reset(seed=seed)
    # Resync wrapped-RL obs tracking (fuel/time) at episode start when available.
    reset_tracking = getattr(controller, "reset_tracking", None)
    if callable(reset_tracking):
        try:
            reset_tracking(float(env.fuel))
        except TypeError:
            reset_tracking()

    start_dist = float(np.linalg.norm(env.rel_pos))
    frames: List[Dict[str, Any]] = []
    dv = 0.0
    overrides = 0
    min_dist = start_dist
    terminal_reason = "timeout"

    def snap(t: float, cmd: np.ndarray, applied: np.ndarray, override: Optional[str]) -> None:
        dist = float(np.linalg.norm(env.rel_pos))
        speed = float(np.linalg.norm(env.rel_vel))
        frames.append(
            _frame(t, env.rel_pos, env.rel_vel, cmd, applied, override, dist, speed, env.fuel, dv)
        )

    snap(0.0, np.zeros(3), np.zeros(3), None)

    step = 0
    while step < env.max_steps:
        cmd = np.clip(np.asarray(controller(env.rel_pos, env.rel_vel), dtype=np.float64).reshape(3), -1.0, 1.0)
        _, _, terminated, truncated, info = env.step(cmd)
        applied = np.asarray(info["applied_action"], dtype=np.float64)
        override = info.get("safety_override")
        if override is not None:
            overrides += 1
        dv += float(np.sum(np.abs(applied)))
        step += 1

        dist = float(np.linalg.norm(env.rel_pos))
        min_dist = min(min_dist, dist)
        step_dt = run_config.physics.dt if run_config else (mission.dt if mission else PHYSICS_CONFIG.dt)
        snap(step * step_dt, cmd, applied, override)

        if terminated:
            terminal_reason = "docked"
            break
        if truncated and env.fuel <= 0.0:
            terminal_reason = "fuel"
            break

    summary: Dict[str, Any] = {
        "label": label,
        "seed": seed,
        "safety": safety,
        "start_dist": round(start_dist, 4),
        "end_dist": round(float(np.linalg.norm(env.rel_pos)), 4),
        "min_dist": round(min_dist, 4),
        "end_speed": round(float(np.linalg.norm(env.rel_vel)), 4),
        "success": terminal_reason == "docked",
        "steps": step,
        "duration": round(step * (run_config.physics.dt if run_config else (mission.dt if mission else PHYSICS_CONFIG.dt)), 2),
        "fuel_used": round((mission.initial_fuel if mission else 1.0) - float(env.fuel), 5),
        "dv": round(dv, 4),
        "overrides": overrides,
        "terminal_reason": terminal_reason,
    }

    if annotate_continuous:
        best_d, best_v = replay_continuous_min(
            frames,
            interval_dt=run_config.physics.dt if run_config else (mission.dt if mission else PHYSICS_CONFIG.dt),
            mean_motion=run_config.physics.n if run_config else (mission.mean_motion if mission else PHYSICS_CONFIG.n),
            max_cmd_accel=run_config.actuator.max_cmd_accel if run_config else 1.0
        )
        summary["continuous_min_dist"] = round(best_d, 4)
        summary["continuous_min_speed"] = round(best_v, 4)
        summary["penetrated"] = bool(
            best_d < ENV_CONFIG.docking_success_dist and best_v >= SAFETY_CONFIG.max_docking_speed
        )

    return {"summary": summary, "frames": frames}


def rl_controller(checkpoint: str, deterministic: bool = True) -> Controller:
    """Wrap an SB3 checkpoint as a plain controller callable.

    The dummy env only builds the observation vector; time/fuel fractions are
    tracked locally so the policy does not see a frozen step_count=0 and
    fuel=1.0 for the whole episode (previous fidelity bug). The controller
    signature stays (pos, vel) -> action for backward compatibility.
    """
    from stable_baselines3 import PPO

    from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv as _Env

    model = PPO.load(checkpoint)
    env = _Env()
    env.reset(seed=0)
    state = {"steps": 0, "fuel": 1.0, "last_applied_norm": 0.0}

    def act(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
        env.rel_pos = np.asarray(rel_pos, dtype=np.float64)
        env.rel_vel = np.asarray(rel_vel, dtype=np.float64)
        # Advance local time/fuel estimates using the previous step's spend so
        # obs[6:8] evolve instead of staying frozen. Clamped, never negative.
        env.step_count = int(state["steps"])
        env.fuel = float(max(0.0, min(1.0, state["fuel"])))
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float64).reshape(3)
        # Mirror env fuel dynamics: fuel -= fuel_rate * sum|applied|.
        spend = float(np.sum(np.abs(np.clip(action, -1.0, 1.0)))) * env.actuator_config.fuel_rate_scale
        state["fuel"] = max(0.0, state["fuel"] - spend)
        state["steps"] += 1
        return action

    # Exposed for tests/callers that want to resync (e.g. new episode).
    def reset_tracking(fuel: float = 1.0) -> None:
        state["steps"] = 0
        state["fuel"] = float(fuel)

    act.reset_tracking = reset_tracking  # type: ignore[attr-defined]
    return act


def write_episode(episode: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode))
    return path
