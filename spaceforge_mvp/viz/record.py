"""Episode recorder — turns any controller into a replayable telemetry trace.

Everything the visualisation layer needs comes from here, so the videos and the
dashboard are driven by the same numbers a test would assert on: no hand-drawn
motion, no mocked telemetry.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from spaceforge_mvp.config import ENV_CONFIG, PHYSICS_CONFIG, SAFETY_CONFIG
from spaceforge_mvp.control.rta import apply_safety_filter
from spaceforge_mvp.core.cw_dynamics import cw_step
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from spaceforge_mvp.safety.mission_safety import evaluate_goal

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
    n_sub = max(1, int(round(PHYSICS_CONFIG.dt / sub_dt)))

    for f in frames[1:]:
        applied = np.asarray(f["applied"], dtype=np.float64)
        for _ in range(n_sub):
            pos, vel = cw_step(pos, vel, applied, dt=sub_dt)
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
    env = SpacecraftEnv(max_steps=max_steps)
    env.reset(seed=seed)
    if initial is not None:
        env.rel_pos = np.asarray(initial[0], dtype=np.float64).copy()
        env.rel_vel = np.asarray(initial[1], dtype=np.float64).copy()

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
    while step < max_steps:
        cmd = np.clip(np.asarray(controller(env.rel_pos, env.rel_vel), dtype=np.float64).reshape(3), -1.0, 1.0)

        if safety == "off":
            applied, override = cmd, None
        elif safety == "v2":
            decision = apply_safety_filter(
                commanded_action=cmd,
                rel_pos=env.rel_pos,
                rel_vel=env.rel_vel,
                fuel_remaining=float(env.fuel),
                goal_type=goal_type,
            )
            applied = np.asarray(decision["applied"], dtype=np.float64)
            override = decision["override"]
            if override is not None:
                overrides += 1
        else:
            verdict = evaluate_goal(
                goal_type=goal_type,
                current_dist=float(np.linalg.norm(env.rel_pos)),
                fuel_remaining=float(env.fuel),
                current_speed=float(np.linalg.norm(env.rel_vel)),
                commanded_action=cmd,
                rel_vel=np.asarray(env.rel_vel, dtype=np.float64),
            )
            status = str(verdict["status"])
            if status == "accepted":
                applied, override = cmd, None
            else:
                applied, override = np.zeros(3), status
                overrides += 1

        next_pos, next_vel = cw_step(env.rel_pos, env.rel_vel, applied, dt=PHYSICS_CONFIG.dt)
        env.rel_pos = np.asarray(next_pos, dtype=np.float64)
        env.rel_vel = np.asarray(next_vel, dtype=np.float64)
        mag = float(np.sum(np.abs(applied)))
        dv += mag
        env.fuel = max(0.0, float(env.fuel) - ENV_CONFIG.fuel_rate_scale * mag)
        step += 1

        dist = float(np.linalg.norm(env.rel_pos))
        speed = float(np.linalg.norm(env.rel_vel))
        min_dist = min(min_dist, dist)
        snap(step * PHYSICS_CONFIG.dt, cmd, applied, override)

        if dist < ENV_CONFIG.docking_success_dist and speed < SAFETY_CONFIG.max_docking_speed:
            terminal_reason = "docked"
            break
        if env.fuel <= 0.0:
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
        "duration": round(step * PHYSICS_CONFIG.dt, 2),
        "fuel_used": round(1.0 - float(env.fuel), 5),
        "dv": round(dv, 4),
        "overrides": overrides,
        "terminal_reason": terminal_reason,
    }

    if annotate_continuous:
        best_d, best_v = replay_continuous_min(frames)
        summary["continuous_min_dist"] = round(best_d, 4)
        summary["continuous_min_speed"] = round(best_v, 4)
        summary["penetrated"] = bool(
            best_d < ENV_CONFIG.docking_success_dist and best_v >= SAFETY_CONFIG.max_docking_speed
        )

    return {"summary": summary, "frames": frames}


def rl_controller(checkpoint: str, deterministic: bool = True) -> Controller:
    """Wrap an SB3 checkpoint as a plain controller callable."""
    from stable_baselines3 import PPO

    from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv as _Env

    model = PPO.load(checkpoint)
    env = _Env()
    env.reset(seed=0)

    def act(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
        env.rel_pos = np.asarray(rel_pos, dtype=np.float64)
        env.rel_vel = np.asarray(rel_vel, dtype=np.float64)
        env.step_count = 0
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float64).reshape(3)

    return act


def write_episode(episode: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode))
    return path
