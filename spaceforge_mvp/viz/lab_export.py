"""Export what the browser simulator needs from the Python reference stack.

    python -m spaceforge_mvp.viz.lab_export

Writes two files into ``spaceforge_mvp/viz/lab/``:

``policy.js``
    The shipped PPO checkpoint's actor and critic weights (float32, base64),
    so the browser can run the exact same network.
``fixtures.json``
    Parity fixtures: unit-level cases for every safety and guidance function
    plus whole episodes stepped through ``SpacecraftEnv``. ``lab/sim.test.js``
    replays them against the JavaScript port, and
    ``tests/test_lab_export.py`` fails if they drift from the Python code.

Python is the source of truth; the browser port is only trusted because it
reproduces these numbers.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from spaceforge_mvp.config import ACTUATOR_CONFIG, ENV_CONFIG, PHYSICS_CONFIG, SAFETY_CONFIG, FaultConfig, MissionConfig
from spaceforge_mvp.control.glideslope import glideslope_action, hot_approach_action
from spaceforge_mvp.control.rta import apply_safety_filter, certified_brake_action, envelope_correction
from spaceforge_mvp.core.cw_dynamics import cw_step
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from spaceforge_mvp.safety.mission_safety import closing_speed_limit, evaluate_goal, predict_unsafe

ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = Path(__file__).resolve().parent / "lab"
CHECKPOINT = ROOT / "checkpoints" / "ppo_spacecraft_final.zip"

N = PHYSICS_CONFIG.n
DT = PHYSICS_CONFIG.dt
A = ACTUATOR_CONFIG.max_cmd_accel


def _f(values: Any) -> List[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


# ----------------------------------------------------------------- policy
def _b64(tensor: Any) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f4"))
    return base64.b64encode(array.tobytes()).decode("ascii")


def export_policy(checkpoint: Path = CHECKPOINT) -> Dict[str, Any]:
    """Actor/critic weights of an SB3 ``MlpPolicy`` with two tanh hidden layers."""
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device="cpu")
    state = model.policy.state_dict()

    def layer(prefix: str) -> Dict[str, Any]:
        weight = state[f"{prefix}.weight"]
        return {"out": int(weight.shape[0]), "in": int(weight.shape[1]),
                "w": _b64(weight), "b": _b64(state[f"{prefix}.bias"])}

    meta_path = checkpoint.with_suffix(".metadata.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "format": "sb3-mlp-tanh-v1",
        "activation": "tanh",
        "actor": [layer("mlp_extractor.policy_net.0"), layer("mlp_extractor.policy_net.2"), layer("action_net")],
        "critic": [layer("mlp_extractor.value_net.0"), layer("mlp_extractor.value_net.2"), layer("value_net")],
        "log_std": _b64(state["log_std"]),
        "source": checkpoint.name,
        "selected_snapshot": meta.get("selected_snapshot"),
    }


# --------------------------------------------------------------- fixtures
def _unit_cases(rng: np.random.Generator, count: int = 160) -> Dict[str, List[Dict[str, Any]]]:
    """Random states concentrated where the safety logic branches (inside ~8 m)."""
    cases: Dict[str, List[Dict[str, Any]]] = {
        "cw_step": [], "glideslope": [], "filter": [], "v1": [], "predict": [], "brake": [], "limit": [],
    }
    for i in range(count):
        scale = [0.4, 2.0, 6.0, 25.0][i % 4]
        pos = rng.normal(0.0, scale, 3)
        vel = rng.normal(0.0, [0.05, 0.2, 0.6, 1.2][i % 4], 3)
        cmd = np.clip(rng.normal(0.0, 0.8, 3), -1.0, 1.0)
        fuel = float(rng.choice([1.0, 0.5, 0.06, 0.04, 0.0]))
        dt = float(rng.choice([1.0, 1.0, 0.5, 2.0]))
        accel = float(rng.choice([A, A, 1.0, 0.03]))

        next_pos, next_vel = cw_step(pos, vel, cmd * accel, dt=dt, n=N)
        cases["cw_step"].append({"pos": _f(pos), "vel": _f(vel), "acc": _f(cmd * accel), "dt": dt,
                                 "next_pos": _f(next_pos), "next_vel": _f(next_vel)})
        cases["glideslope"].append({"pos": _f(pos), "vel": _f(vel), "dt": dt, "max_accel": accel,
                                    "out": _f(glideslope_action(pos, vel, n=N, dt=dt, max_accel=accel))})
        decision = apply_safety_filter(cmd, pos, vel, fuel, "APPROACH", SAFETY_CONFIG, dt=dt, max_accel=accel, n=N)
        cases["filter"].append({"cmd": _f(cmd), "pos": _f(pos), "vel": _f(vel), "fuel": fuel, "dt": dt,
                                "max_accel": accel, "applied": _f(decision["applied"]),
                                "override": decision["override"]})
        verdict = evaluate_goal("APPROACH", float(np.linalg.norm(pos)), fuel, float(np.linalg.norm(vel)),
                                cmd, vel, SAFETY_CONFIG, dt=dt, max_accel=accel)
        cases["v1"].append({"cmd": _f(cmd), "pos": _f(pos), "vel": _f(vel), "fuel": fuel, "dt": dt,
                            "max_accel": accel, "status": verdict["status"]})
        cases["predict"].append({"pos": _f(pos), "vel": _f(vel), "acc": _f(cmd * accel), "dt": dt, "max_accel": accel,
                                 "unsafe": bool(predict_unsafe(pos, vel, cmd * accel, dt, SAFETY_CONFIG, max_accel=accel))})
        cases["brake"].append({"pos": _f(pos), "vel": _f(vel), "dt": dt, "max_accel": accel,
                               "brake": _f(certified_brake_action(vel, max_accel=accel, dt=dt)),
                               "envelope": _f(envelope_correction(pos, vel, SAFETY_CONFIG, accel, dt))})
    for d in np.concatenate([np.linspace(0.0, 7.0, 29), np.linspace(8.0, 60.0, 14)]):
        for accel in (A, 1.0):
            cases["limit"].append({"d": float(d), "max_accel": accel,
                                   "limit": closing_speed_limit(float(d), SAFETY_CONFIG, accel)})
    return cases


Controller = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _episode(
    name: str,
    controller: str,
    start_pos: List[float],
    start_vel: List[float],
    safety: str,
    max_steps: int = 300,
    fuel: float = 1.0,
    faults: Optional[FaultConfig] = None,
    policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    mission = MissionConfig(tuple(start_pos), (0, 0, 0), tuple(start_vel), (0, 0, 0), fuel,
                            float(max_steps) * DT, DT, N, safety)
    env = SpacecraftEnv(mission=mission, faults=faults)
    obs, _ = env.reset(seed=0)
    steps: List[Dict[str, Any]] = []
    outcome = "timeout"
    for _ in range(max_steps):
        if controller == "glideslope":
            cmd = glideslope_action(env.rel_pos, env.rel_vel, n=N, dt=DT, max_accel=A)
        elif controller == "hot":
            cmd = hot_approach_action(env.rel_pos, env.rel_vel)
        elif controller == "away":
            cmd = env.rel_pos / max(float(np.linalg.norm(env.rel_pos)), 1e-9)
        elif controller == "ppo":
            assert policy is not None
            cmd = policy(obs)
        else:
            raise ValueError(controller)
        obs, _, terminated, truncated, info = env.step(cmd)
        record = {
            "pos": _f(env.rel_pos), "vel": _f(env.rel_vel), "fuel": float(env.fuel),
            "override": info["safety_override"],
        }
        if controller == "ppo":
            record["action"] = _f(np.clip(cmd, -1.0, 1.0))
        steps.append(record)
        if terminated:
            outcome = "collision" if info["collision"] else "docked"
            break
        if truncated:
            outcome = "fuel" if env.fuel <= 0.0 else ("out_of_bounds" if info["out_of_bounds"] else "timeout")
            break
    case: Dict[str, Any] = {
        "name": name, "controller": controller, "safety": safety, "start_pos": start_pos,
        "start_vel": start_vel, "fuel": fuel, "max_steps": max_steps, "outcome": outcome, "steps": steps,
    }
    if faults is not None:
        case["faults"] = {"stuck_axis": faults.stuck_axis, "stuck_value": faults.stuck_value,
                          "bias": list(faults.bias), "fuel_leak_rate": faults.fuel_leak_rate}
    return case


def build_fixtures(include_policy: bool = True) -> Dict[str, Any]:
    rng = np.random.default_rng(20261102)
    episodes = [
        _episode("glideslope-v1", "glideslope", [12.0, -7.0, 3.5], [0, 0, 0], "v1"),
        _episode("glideslope-v2-moving", "glideslope", [-15.0, 9.0, -4.0], [0.01, -0.004, 0.002], "v2"),
        _episode("hot-off", "hot", [6.0, 0.0, 0.0], [0, 0, 0], "off", max_steps=120),
        _episode("hot-v1", "hot", [6.0, 0.0, 0.0], [0, 0, 0], "v1", max_steps=120),
        _episode("hot-v2", "hot", [6.0, 0.0, 0.0], [0, 0, 0], "v2", max_steps=120),
        _episode("hot-v2-oblique", "hot", [3.0, -8.0, 2.0], [0, 0, 0], "v2", max_steps=200),
        _episode("hot-v2-stuck", "hot", [8.0, 4.0, 0.0], [0, 0, 0], "v2", max_steps=200,
                 faults=FaultConfig(stuck_axis=1, stuck_value=0.6)),
        _episode("glideslope-v2-bias", "glideslope", [10.0, 10.0, 0.0], [0, 0, 0], "v2",
                 faults=FaultConfig(bias=(0.05, -0.03, 0.0))),
        _episode("hot-v2-low-fuel", "hot", [7.0, -2.0, 1.0], [0, 0, 0], "v2", max_steps=150, fuel=0.07,
                 faults=FaultConfig(fuel_leak_rate=0.0005)),
        _episode("away-off", "away", [10.0, 0.0, 0.0], [0, 0, 0], "off", max_steps=200),
    ]
    if include_policy:
        from stable_baselines3 import PPO

        model = PPO.load(str(CHECKPOINT), device="cpu")
        policy = lambda obs: np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.float64)  # noqa: E731
        episodes.append(_episode("ppo-v1", "ppo", [12.0, -7.0, 3.5], [0, 0, 0], "v1", policy=policy))
        episodes.append(_episode("ppo-v2", "ppo", [-14.0, 11.0, 6.0], [0.003, 0.0, -0.002], "v2", policy=policy))
    return {
        "schema_version": 1,
        "config": {
            "n": N, "dt": DT, "max_accel": A, "fuel_rate": ACTUATOR_CONFIG.fuel_rate_scale,
            "safe_dist": SAFETY_CONFIG.safe_approach_dist, "fuel_floor": SAFETY_CONFIG.fuel_floor,
            "max_safe_closing": SAFETY_CONFIG.max_safe_closing_speed,
            "max_dock_speed": SAFETY_CONFIG.max_docking_speed, "capture_radius": SAFETY_CONFIG.capture_radius,
            "capture_frac": SAFETY_CONFIG.capture_speed_fraction, "dock_dist": ENV_CONFIG.docking_success_dist,
            "outer_braking_fraction": SAFETY_CONFIG.outer_braking_fraction,
            "work_volume": ENV_CONFIG.work_volume_bound, "pos_norm": ENV_CONFIG.pos_norm_scale,
            "vel_norm": ENV_CONFIG.vel_norm_scale,
        },
        "unit": _unit_cases(rng),
        "episodes": episodes,
    }


def policy_js(policy: Dict[str, Any]) -> str:
    body = json.dumps(policy, separators=(",", ":"))
    return (
        "// Generated by `python -m spaceforge_mvp.viz.lab_export` -- do not edit.\n"
        f"(function(root){{var P={body};"
        "if(typeof module==='object'&&module.exports)module.exports=P;else root.SF_POLICY=P;"
        "})(typeof self!=='undefined'?self:this);\n"
    )


def fixtures_text(fixtures: Dict[str, Any]) -> str:
    return json.dumps(fixtures, separators=(",", ":")) + "\n"


def write(lab_dir: Path = LAB_DIR) -> Dict[str, Path]:
    lab_dir.mkdir(parents=True, exist_ok=True)
    policy_path = lab_dir / "policy.js"
    fixtures_path = lab_dir / "fixtures.json"
    policy_path.write_text(policy_js(export_policy()))
    fixtures_path.write_text(fixtures_text(build_fixtures()))
    return {"policy": policy_path, "fixtures": fixtures_path}


if __name__ == "__main__":
    for kind, path in write().items():
        print(f"{kind:<9} {path}  ({path.stat().st_size / 1024:.0f} KB)")
