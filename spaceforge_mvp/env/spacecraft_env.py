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
    normalized command; physical acceleration = action * ActuatorConfig.max_cmd_accel (m/s^2)

Reward components are reported in ``info["reward_breakdown"]``.  Position terms
    use metres, velocity uses metres/sec, effort uses normalized actuator command,
    fuel uses consumed fuel fraction, and collision risk is dimensionless.  The
    configured weights are progress, divergence, velocity, fuel, effort, risk,
    and docking success respectively; their sum is the scalar reward.

Episode ends:
    terminated=True  -> docked: dist < docking_success_dist AND speed < max_docking_speed
    terminated=True  -> collision: the step swept through the capture sphere at or
                        above max_docking_speed (penalty ``collision_penalty``)
    truncated=True   -> fuel exhausted, left the work volume, or max_steps reached
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from spaceforge_mvp.config import ENV_CONFIG, PHYSICS_CONFIG, SAFETY_CONFIG, ActuatorConfig, DomainConfig, FaultConfig, MissionConfig, RunConfig
from spaceforge_mvp.control.rta import apply_safety_filter
from spaceforge_mvp.core.cw_dynamics import cw_step
from spaceforge_mvp.safety.mission_safety import evaluate_goal

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
        run_config: Optional[RunConfig] = None,
        domain: Optional[DomainConfig] = None,
        faults: Optional[FaultConfig] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.run_config = run_config
        self.mission = run_config.mission if run_config else mission
        self.max_steps = int(max_steps or (run_config.max_steps if run_config else (mission.max_steps if mission else ENV_CONFIG.max_steps)))
        self.safety_mode = safety_mode or (run_config.safety_mode if run_config else (mission.safety_mode if mission else "v1"))
        self.env_config = run_config.environment if run_config else ENV_CONFIG
        self.physics_config = run_config.physics if run_config else PHYSICS_CONFIG
        self.actuator_config = run_config.actuator if run_config else ActuatorConfig()
        self.safety_config = run_config.safety if run_config else SAFETY_CONFIG
        self.reset_config = run_config.reset if run_config else None
        self.domain_config = run_config.domain if run_config and run_config.domain else domain
        self.fault_config = run_config.faults if run_config and run_config.faults else faults
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
        self.fuel: float = float(self.mission.initial_fuel if self.mission else 1.0)
        self.step_count: int = 0
        # --- per-episode domain draws (identity unless configured) ---
        self.episode_mass_kg: float = 1.0
        self.episode_effectiveness: np.ndarray = np.ones(3, dtype=np.float64)
        self._last_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ helpers
    def _draw_domain(self) -> None:
        """Draw per-episode mass/effectiveness from the seeded RNG (identity by default)."""
        domain = self.domain_config
        if domain is None or domain.is_identity:
            self.episode_mass_kg = 1.0
            self.episode_effectiveness = np.ones(3, dtype=np.float64)
            return
        self.episode_mass_kg = float(
            self.np_random.uniform(domain.mass_low_kg, domain.mass_high_kg)
        )
        tol = float(domain.effectiveness_tol)
        self.episode_effectiveness = self.np_random.uniform(1.0 - tol, 1.0 + tol, size=3).astype(
            np.float64
        )

    def _get_obs(self) -> np.ndarray:
        """Build the normalized 10-D observation vector.

        With domain randomization, camera/IMU noise perturbs the observed
        position/velocity (and the bearing derived from them). The true state
        in rel_pos/rel_vel — used for rewards and docking judgment — is exact.
        """
        pos = np.asarray(self.rel_pos, dtype=np.float64)
        vel = np.asarray(self.rel_vel, dtype=np.float64)
        domain = self.domain_config
        if domain is not None and (
            domain.obs_pos_noise_std_m > 0.0 or domain.obs_vel_noise_std_mps > 0.0
        ):
            pos = pos + self.np_random.normal(0.0, domain.obs_pos_noise_std_m, size=3)
            vel = vel + self.np_random.normal(0.0, domain.obs_vel_noise_std_mps, size=3)
        pos_norm = pos / self.env_config.pos_norm_scale
        vel_norm = vel / self.env_config.vel_norm_scale
        fuel_frac = float(self.fuel)
        time_frac = float(self.step_count) / float(self.max_steps)

        planar = np.asarray(pos[0:2], dtype=np.float64)
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
        ).astype(np.float32)
        faults = self.fault_config
        if faults is not None and faults.sensor_freeze_prob > 0.0 and self._last_obs is not None:
            if float(self.np_random.uniform()) < faults.sensor_freeze_prob:
                return self._last_obs
        self._last_obs = obs
        return obs

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
            "reward_breakdown": {},
        }

    def _swept_collision(self, pos0: np.ndarray, vel0: np.ndarray, accel: np.ndarray, dt: float, samples: int = 10) -> bool:
        """Did the step pass through the capture sphere at or above docking speed?

        The environment samples once per step, so a fast fly-through can jump
        over the 0.5 m capture sphere between samples. The step is re-sampled
        with constant-acceleration kinematics (CW coupling over one step is
        negligible against the actuator authority) to catch it.
        """
        taus = np.linspace(float(dt) / samples, float(dt), samples)[:, None]
        p = pos0 + vel0 * taus + 0.5 * accel * taus ** 2
        v = vel0 + accel * taus
        inside = np.linalg.norm(p, axis=1) < self.env_config.docking_success_dist
        fast = np.linalg.norm(v, axis=1) >= self.safety_config.max_docking_speed
        return bool(np.any(inside & fast))

    # -------------------------------------------------------------------- gym API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Draw a fresh random relative state and return (obs, info)."""
        super().reset(seed=seed)

        if self.mission is None or (self.reset_config is not None and not self.reset_config.use_mission_state):
            reset = self.reset_config
            distribution = reset.distribution if reset else "uniform"
            pos_bound = reset.position_bound if reset else self.env_config.initial_pos_bound
            vel_bound = reset.velocity_bound if reset else self.env_config.initial_vel_bound
            if distribution == "curriculum":
                pos_bounds = reset.curriculum_position_bounds
                vel_bounds = reset.curriculum_velocity_bounds
                stage = min(reset.curriculum_stage, len(pos_bounds) - 1, len(vel_bounds) - 1)
                pos_bound, vel_bound = pos_bounds[stage], vel_bounds[stage]
            if distribution == "radial":
                self.rel_pos = np.array([self.np_random.uniform(pos_bound * 0.5, pos_bound), 0.0, 0.0], dtype=np.float64)
                self.rel_vel = self.np_random.uniform(-vel_bound, vel_bound, size=3).astype(np.float64)
            else:
                self.rel_pos = self.np_random.uniform(-pos_bound, pos_bound, size=3).astype(np.float64)
                self.rel_vel = self.np_random.uniform(-vel_bound, vel_bound, size=3).astype(np.float64)
            domain = self.domain_config
            if domain is not None and (domain.fuel_low, domain.fuel_high) != (1.0, 1.0):
                self.fuel = float(self.np_random.uniform(domain.fuel_low, domain.fuel_high))
            else:
                self.fuel = 1.0
        else:
            self.rel_pos = np.asarray(self.mission.relative_pos, dtype=np.float64)
            self.rel_vel = np.asarray(self.mission.relative_vel, dtype=np.float64)
            self.fuel = float(self.mission.initial_fuel)
        self.step_count = 0
        self._last_obs = None
        self._draw_domain()

        dist = float(np.linalg.norm(self.rel_pos))
        speed = float(np.linalg.norm(self.rel_vel))
        is_success = bool(
            dist < self.env_config.docking_success_dist
            and speed < self.safety_config.max_docking_speed
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
        dt = float(self.physics_config.dt)
        max_accel = float(self.actuator_config.max_cmd_accel)
        if self.safety_mode == "off":
            applied = cmd.copy()
            override = None
            safety_reason = "Safety interlock disabled."
        elif self.safety_mode == "v2":
            decision = apply_safety_filter(
                cmd, self.rel_pos, self.rel_vel, self.fuel, goal_type, self.safety_config,
                dt=dt, max_accel=max_accel, n=float(self.physics_config.n),
            )
            applied = np.asarray(decision["applied"], dtype=np.float64).copy()
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
                safety_config=self.safety_config,
                dt=dt,
                max_accel=max_accel,
            )
            status = str(verdict["status"])
            applied = cmd.copy() if status == "accepted" else np.zeros(3, dtype=np.float64)
            override = None if status == "accepted" else status
            safety_reason = str(verdict["reason"])

        # --- faults (post-interlock, pre-physics): stuck -> bias -> dropout ---
        # Work on a copy: faults change what the thrusters do, not what the
        # interlock decided, and telemetry must keep the two apart.
        faults = self.fault_config
        fault_active: Optional[str] = None
        disturbed = np.array(applied, dtype=np.float64, copy=True)
        if faults is not None and not faults.is_identity:
            if faults.stuck_axis is not None:
                disturbed[faults.stuck_axis] = faults.stuck_value
                fault_active = "stuck_thruster"
            disturbed = disturbed + np.asarray(faults.bias, dtype=np.float64)
            if faults.dropout_prob > 0.0 and float(self.np_random.uniform()) < faults.dropout_prob:
                disturbed = np.zeros(3, dtype=np.float64)
                fault_active = "command_dropout"
        # Thruster throttle actually fired (normalized, saturated): this is what
        # consumes propellant, including faulty firings.
        throttle = np.clip(disturbed, -1.0, 1.0)

        # --- domain: mass/inertia scaling + thruster noise, then clip ---
        domain = self.domain_config
        mass_scale = 1.0
        if domain is not None and not domain.is_identity:
            mass_scale = float(domain.mass_nominal_kg / self.episode_mass_kg)
            disturbed = disturbed * self.episode_effectiveness
            if domain.thrust_noise_std > 0.0:
                disturbed = disturbed + self.np_random.normal(0.0, domain.thrust_noise_std, size=3)
        accel_cmd = np.clip(
            disturbed * mass_scale * max_accel,
            -max_accel,
            max_accel,
        )

        # --- physics: exact CW state transition matrix ---
        prev_pos = np.array(self.rel_pos, dtype=np.float64, copy=True)
        prev_vel = np.array(self.rel_vel, dtype=np.float64, copy=True)
        next_pos, next_vel = cw_step(
            self.rel_pos,
            self.rel_vel,
            accel_cmd,
            dt=dt,
            n=self.physics_config.n,
        )
        self.rel_pos = np.asarray(next_pos, dtype=np.float64)
        self.rel_vel = np.asarray(next_vel, dtype=np.float64)

        thrust_mag = float(np.sum(np.abs(throttle)))
        fuel_used = max(0.0, float(self.actuator_config.fuel_rate_scale * thrust_mag))
        fuel_debit = fuel_used
        if faults is not None and faults.fuel_leak_rate > 0.0:
            fuel_debit += faults.fuel_leak_rate
            fault_active = fault_active or "fuel_leak"
        self.fuel = max(0.0, float(self.fuel) - fuel_debit)
        self.step_count += 1

        dist = float(np.linalg.norm(self.rel_pos))
        speed = float(np.linalg.norm(self.rel_vel))

        progress = self.env_config.progress_reward_weight * (dist_prev - dist)
        divergence = -self.env_config.divergence_penalty_weight * max(0.0, dist - dist_prev)
        if self.env_config.velocity_penalty_mode == "approach":
            if dist > _EPS:
                ref_speed = min(self.env_config.approach_closing_cap, dist / self.env_config.approach_glide_time)
                v_ref = -self.rel_pos / dist * ref_speed
            else:
                v_ref = np.zeros(3, dtype=np.float64)
            velocity = -self.env_config.velocity_penalty_weight * float(np.linalg.norm(self.rel_vel - v_ref))
        else:
            velocity = -self.env_config.velocity_penalty_weight * speed
        fuel_penalty = -self.env_config.fuel_penalty_weight * fuel_used
        effort = -self.env_config.action_penalty_weight * thrust_mag
        risk = 0.0
        if dist < self.safety_config.safe_approach_dist:
            risk = max(0.0, (speed - self.safety_config.max_safe_closing_speed) / self.safety_config.max_safe_closing_speed)
        collision_risk = -self.env_config.collision_risk_penalty_weight * risk
        reward = progress + divergence + velocity + fuel_penalty + effort + collision_risk

        is_success = bool(
            dist < self.env_config.docking_success_dist
            and speed < self.safety_config.max_docking_speed
        )
        collision = False
        if not is_success and self.env_config.terminate_on_collision:
            collision = self._swept_collision(prev_pos, prev_vel, accel_cmd, dt)
        terminated = False
        truncated = False
        docking = 0.0
        collision_term = 0.0
        if is_success:
            docking = self.env_config.docking_bonus
            reward += docking
            terminated = True
        elif collision:
            collision_term = -self.env_config.collision_penalty
            reward += collision_term
            terminated = True
        elif self.fuel <= 0.0 or self.step_count >= self.max_steps or dist > self.env_config.work_volume_bound:
            truncated = True

        info = self._info(
            dist=dist,
            speed=speed,
            is_success=is_success,
            safety_override=override,
            applied_action=applied,
            safety_reason=safety_reason,
        )
        info["collision"] = bool(collision)
        info["out_of_bounds"] = bool(truncated and dist > self.env_config.work_volume_bound)
        info["collision_risk"] = float(risk)
        info["fuel_used"] = float(fuel_used)
        info["domain"] = {
            "mass_kg": float(self.episode_mass_kg),
            "effectiveness": [float(v) for v in self.episode_effectiveness],
        }
        info["fault"] = fault_active
        info["thruster_command"] = np.asarray(throttle, dtype=np.float64)
        info["executed_action"] = np.asarray(accel_cmd, dtype=np.float64)
        info["reward_breakdown"] = {
            "progress": float(progress), "divergence": float(divergence),
            "velocity": float(velocity), "fuel": float(fuel_penalty),
            "action_effort": float(effort), "collision_risk": float(collision_risk),
            "docking_success": float(docking), "collision": float(collision_term),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
