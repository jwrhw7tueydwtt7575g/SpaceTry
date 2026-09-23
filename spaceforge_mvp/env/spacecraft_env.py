"""Gymnasium environment for autonomous spacecraft rendezvous and docking.

The chaser's relative state is propagated with the exact Clohessy-Wiltshire
state transition matrix (see ``spaceforge_mvp.core.cw_dynamics``). Every
commanded thrust vector passes through the deterministic mission safety
interlock (``spaceforge_mvp.safety.mission_safety``) before it reaches the
physics, exactly as the real flight stack would do.

Observation (10,) float32:
    [0:3]  relative position  [x, y, z] / pos_norm_scale
    [3:6]  relative velocity  [vx, vy, vz] / vel_norm_scale
    [6]    fuel fraction      [0.0, 1.0]
    [7]    mission time fraction step / max_steps
    [8:10] 2D unit bearing vector pointing from the chaser at the target

Action (3,) float32 in [-1, 1]:
    commanded acceleration [ax, ay, az] in m/s^2 (scaled by MAX_CMD_ACCEL)

Reward:
    5.0 * (dist_prev - dist_curr) - 0.5 * speed - 0.01 * sum(|u_applied|)
    plus ENV_CONFIG.docking_bonus on a successful dock.

Episode ends:
    terminated=True  -> dist < docking_success_dist AND speed < max_docking_speed
    truncated=True   -> fuel exhausted or max_steps reached
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from spaceforge_mvp.config import ENV_CONFIG, PHYSICS_CONFIG, SAFETY_CONFIG, MissionConfig
from spaceforge_mvp.control.rta import apply_safety_filter
from spaceforge_mvp.core.cw_dynamics import cw_step
from spaceforge_mvp.safety.mission_safety import evaluate_goal

# Action is bounded to [-1, 1]; this converts it to a commanded acceleration in m/s^2.
MAX_CMD_ACCEL: float = 1.0

# Goal string handed to the safety interlock. The MVP only ever flies APPROACH,
# which is the goal type the proximity speed gate is keyed on.
APPROACH_GOAL: str = "APPROACH"

_EPS: float = 1e-9


class SpacecraftEnv(gym.Env):
    """Relative-motion rendezvous environment for a chaser spacecraft."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        mission: Optional[MissionConfig] = None,
        safety_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.mission = mission
        self.max_steps = int(max_steps or (mission.max_steps if mission else ENV_CONFIG.max_steps))
        self.safety_mode = safety_mode or (mission.safety_mode if mission else "v1")
        if self.safety_mode not in {"off", "v1", "v2"}:
            raise ValueError("safety_mode must be one of: off, v1, v2")

        # Observation values are unbounded in principle (a diverging policy can
        # fly arbitrarily far), so declare the box accordingly instead of
        # clipping real state into a fake range.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # --- mutable episode state ---
        self.rel_pos: np.ndarray = np.zeros(3, dtype=np.float64)
        self.rel_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self.fuel: float = float(mission.initial_fuel if mission else 1.0)
        self.step_count: int = 0

    # ------------------------------------------------------------------ helpers
    def _get_obs(self) -> np.ndarray:
        """Build the normalized 10-D observation vector."""
        pos_norm = self.rel_pos / ENV_CONFIG.pos_norm_scale
        vel_norm = self.rel_vel / ENV_CONFIG.vel_norm_scale
        fuel_frac = float(self.fuel)
        time_frac = float(self.step_count) / float(self.max_steps)

        planar = np.asarray(self.rel_pos[0:2], dtype=np.float64)
        planar_norm = float(np.linalg.norm(planar))
        if planar_norm > _EPS:
            bearing = -planar / planar_norm
        else:
            bearing = np.zeros(2, dtype=np.float64)

        obs = np.concatenate(
            [
                np.asarray(pos_norm, dtype=np.float64),
                np.asarray(vel_norm, dtype=np.float64),
                np.asarray([fuel_frac, time_frac], dtype=np.float64),
                np.asarray(bearing, dtype=np.float64),
            ]
        )
        return obs.astype(np.float32)

    def _info(
        self,
        dist: float,
        speed: float,
        is_success: bool,
        safety_override: Optional[str],
        applied_action: np.ndarray,
        safety_reason: str,
    ) -> Dict[str, Any]:
        return {
            "distance": float(dist),
            "speed": float(speed),
            "is_success": bool(is_success),
            "safety_override": safety_override,
            "applied_action": np.asarray(applied_action, dtype=np.float64),
            "safety_reason": safety_reason,
            "fuel": float(self.fuel),
            "step_count": int(self.step_count),
        }

    # -------------------------------------------------------------------- gym API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Draw a fresh random relative state and return (obs, info)."""
        super().reset(seed=seed)

        if self.mission is None:
            pos_bound = ENV_CONFIG.initial_pos_bound
            vel_bound = ENV_CONFIG.initial_vel_bound
            self.rel_pos = self.np_random.uniform(-pos_bound, pos_bound, size=3).astype(np.float64)
            self.rel_vel = self.np_random.uniform(-vel_bound, vel_bound, size=3).astype(np.float64)
            self.fuel = 1.0
        else:
            self.rel_pos = np.asarray(self.mission.relative_pos, dtype=np.float64)
            self.rel_vel = np.asarray(self.mission.relative_vel, dtype=np.float64)
            self.fuel = float(self.mission.initial_fuel)
        self.step_count = 0

        dist = float(np.linalg.norm(self.rel_pos))
        speed = float(np.linalg.norm(self.rel_vel))
        is_success = bool(
            dist < ENV_CONFIG.docking_success_dist
            and speed < SAFETY_CONFIG.max_docking_speed
        )
        info = self._info(
            dist=dist,
            speed=speed,
            is_success=is_success,
            safety_override=None,
            applied_action=np.zeros(3, dtype=np.float64),
            safety_reason="Episode reset.",
        )
        return self._get_obs(), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply the safety interlock to `action`, propagate the physics one step."""
        cmd = np.asarray(action, dtype=np.float64).reshape(3)
        cmd = np.clip(cmd, -1.0, 1.0)

        dist_prev = float(np.linalg.norm(self.rel_pos))
        speed_prev = float(np.linalg.norm(self.rel_vel))

        goal_type = self.mission.goal_type if self.mission else APPROACH_GOAL
        if self.safety_mode == "off":
            applied = cmd
            override = None
            safety_reason = "Safety interlock disabled."
        elif self.safety_mode == "v2":
            decision = apply_safety_filter(cmd, self.rel_pos, self.rel_vel, self.fuel, goal_type)
            applied = np.asarray(decision["applied"], dtype=np.float64)
            override = decision["override"]
            safety_reason = str(decision["reason"])
        else:
            verdict = evaluate_goal(
                goal_type=goal_type,
                current_dist=dist_prev,
                fuel_remaining=float(self.fuel),
                current_speed=speed_prev,
                commanded_action=cmd,
                rel_vel=np.asarray(self.rel_vel, dtype=np.float64),
            )
            status = str(verdict["status"])
            applied = cmd if status == "accepted" else np.zeros(3, dtype=np.float64)
            override = None if status == "accepted" else status
            safety_reason = str(verdict["reason"])

        # --- physics: exact CW state transition matrix ---
        next_pos, next_vel = cw_step(
            self.rel_pos,
            self.rel_vel,
            applied * MAX_CMD_ACCEL,
            dt=self.mission.dt if self.mission else PHYSICS_CONFIG.dt,
            n=self.mission.mean_motion if self.mission else PHYSICS_CONFIG.n,
        )
        self.rel_pos = np.asarray(next_pos, dtype=np.float64)
        self.rel_vel = np.asarray(next_vel, dtype=np.float64)

        thrust_mag = float(np.sum(np.abs(applied)))
        self.fuel = max(0.0, float(self.fuel) - ENV_CONFIG.fuel_rate_scale * thrust_mag)
        self.step_count += 1

        dist = float(np.linalg.norm(self.rel_pos))
        speed = float(np.linalg.norm(self.rel_vel))

        reward = (
            ENV_CONFIG.progress_reward_weight * (dist_prev - dist)
            - ENV_CONFIG.velocity_penalty_weight * speed
            - ENV_CONFIG.action_penalty_weight * thrust_mag
        )

        is_success = bool(
            dist < ENV_CONFIG.docking_success_dist
            and speed < SAFETY_CONFIG.max_docking_speed
        )
        terminated = False
        truncated = False
        if is_success:
            reward += ENV_CONFIG.docking_bonus
            terminated = True
        elif self.fuel <= 0.0 or self.step_count >= self.max_steps:
            truncated = True

        info = self._info(
            dist=dist,
            speed=speed,
            is_success=is_success,
            safety_override=override,
            applied_action=applied,
            safety_reason=safety_reason,
        )
        return self._get_obs(), float(reward), terminated, truncated, info
