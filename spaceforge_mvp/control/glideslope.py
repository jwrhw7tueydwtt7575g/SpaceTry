"""Deterministic reference guidance laws for Clohessy-Wiltshire rendezvous.

``glideslope_action`` is the honest baseline: what a flight-dynamics team would
fly before handing the problem to reinforcement learning. It is used to bound
what the environment actually allows, and to give the RL policy something real
to be measured against.

Control law
-----------
1. Desired velocity points at the target with magnitude
   ``min(closing_cap, d / glide_time, braking_margin * sqrt(2 * a_max * d))``
   -> an exponential glideslope, capped so the vehicle can always stop within
   the remaining range using the available authority.
2. First-order velocity tracking ``a = k * (v_des - v)`` with
   ``k = min(vel_gain, 1 / dt)``: at most a dead-beat correction per step, so
   the discrete loop stays stable and non-oscillatory for any step size.
3. Feed-forward cancellation of the natural CW drift (``cw_accel``) at the
   mission's own mean motion.
4. Convert to the environment's normalized command (1.0 == ``max_accel``
   m/s^2) and saturate to [-1, 1].
"""

from functools import partial
from typing import Callable, Optional

import numpy as np

from spaceforge_mvp.config import PHYSICS_CONFIG
from spaceforge_mvp.core.cw_dynamics import cw_accel


def glideslope_action(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    glide_time: float = 30.0,
    closing_cap: float = 0.5,
    vel_gain: float = 0.7,
    cw_compensation: bool = True,
    n: float = PHYSICS_CONFIG.n,
    dt: float = PHYSICS_CONFIG.dt,
    max_accel: float = 1.0,
    braking_margin: float = 0.5,
) -> np.ndarray:
    """Return the normalized command for a safe approach (1.0 == ``max_accel`` m/s^2).

    Parameters
    ----------
    rel_pos, rel_vel : array-like
        Relative state in the Hill frame (target at the origin).
    glide_time : float
        Glideslope time constant in seconds (closing speed = range / glide_time).
    closing_cap : float
        Hard cap on the commanded closing speed (m/s). With glide_time=30 the
        commanded speed at 5 m is 0.167 m/s, inside the 0.2 m/s interlock gate.
    vel_gain : float
        Requested velocity-loop gain in 1/s; limited to 1/dt for stability.
    n : float
        Mean motion (rad/s) used for the CW feed-forward. Pass the mission's value.
    dt : float
        Control step (s).
    max_accel : float
        Actuator authority (m/s^2) that a unit command produces.
    braking_margin : float
        Fraction of the stopping-distance-limited speed sqrt(2 a d) to allow.
    """
    pos = np.asarray(rel_pos, dtype=np.float64)
    vel = np.asarray(rel_vel, dtype=np.float64)

    dist = float(np.linalg.norm(pos))
    if dist < 1e-9:
        v_des = np.zeros(3, dtype=np.float64)
    else:
        stoppable = float(braking_margin) * float(np.sqrt(2.0 * float(max_accel) * dist))
        closing_speed = min(float(closing_cap), dist / float(glide_time), stoppable)
        v_des = (-pos / dist) * closing_speed

    gain = min(float(vel_gain), 1.0 / float(dt))
    accel = gain * (v_des - vel)
    if cw_compensation:
        accel = accel - cw_accel(pos, vel, n=float(n))

    return np.clip(accel / float(max_accel), -1.0, 1.0).astype(np.float64)


def glideslope_controller(run_config: Optional[object] = None, **overrides) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Bind the glideslope law to a run's physics and actuator configuration.

    Returns a ``(rel_pos, rel_vel) -> command`` callable that uses the run's
    mean motion, step size and actuator authority instead of module defaults.
    """
    params = {}
    if run_config is not None:
        params = {
            "n": run_config.physics.n,
            "dt": run_config.physics.dt,
            "max_accel": run_config.actuator.max_cmd_accel,
        }
    params.update(overrides)
    return partial(glideslope_action, **params)


def hot_approach_action(
    rel_pos: np.ndarray,
    rel_vel: Optional[np.ndarray] = None,
    max_accel: float = 1.0,
) -> np.ndarray:
    """Deliberately unsafe guidance: full thrust straight at the target, never brakes.

    This is the adversarial case for the mission safety interlock. Left unchecked
    it drives the chaser through the target at metres per second.
    """
    pos = np.asarray(rel_pos, dtype=np.float64)
    dist = float(np.linalg.norm(pos))
    if dist < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return ((-pos / dist) * float(max_accel)).astype(np.float64)
