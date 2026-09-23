"""Deterministic reference guidance laws for Clohessy-Wiltshire rendezvous.

``glideslope_action`` is the honest baseline: what a flight-dynamics team would
fly before handing the problem to reinforcement learning. It is used to bound
what the environment actually allows, and to give the RL policy something real
to be measured against.

Control law
-----------
1. Desired velocity points at the target with magnitude ``min(closing_cap, d / glide_time)``
   -> an exponential glideslope with time constant ``glide_time``.
2. First-order velocity tracking: ``a = vel_gain * (v_des - v)``.
3. Feed-forward cancellation of the natural CW drift (``cw_accel``), which turns
   the relative motion into a near double integrator and removes the
   Coriolis-like coupling that makes naive pursuit guidance spiral.
4. Saturate to the actuator limit [-1, 1] m/s^2.
"""

from typing import Optional

import numpy as np

from spaceforge_mvp.core.cw_dynamics import cw_accel


def glideslope_action(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    glide_time: float = 30.0,
    closing_cap: float = 0.5,
    vel_gain: float = 0.7,
    cw_compensation: bool = True,
) -> np.ndarray:
    """Return the commanded acceleration [ax, ay, az] in m/s^2 for a safe approach.

    Parameters
    ----------
    rel_pos : array-like
        Relative position [x, y, z] in meters (Hill frame, target at the origin).
    rel_vel : array-like
        Relative velocity [vx, vy, vz] in m/s.
    glide_time : float
        Glideslope time constant in seconds. Closing speed is distance / glide_time,
        so the range decays exponentially with this constant.
    closing_cap : float
        Hard cap on the commanded closing speed (m/s). Keep it below the 0.2 m/s
        safety-interlock gate at 5 m: with glide_time=30 the commanded speed at
        5 m is 0.167 m/s, so the interlock never has to speak.
    vel_gain : float
        Velocity-loop gain in 1/s. Must stay below 2/dt for discrete stability;
        below 1/dt to be non-oscillatory.
    cw_compensation : bool
        Cancel the natural CW drift acceleration (feed-forward).
    """
    pos = np.asarray(rel_pos, dtype=np.float64)
    vel = np.asarray(rel_vel, dtype=np.float64)

    dist = float(np.linalg.norm(pos))
    if dist < 1e-9:
        v_des = np.zeros(3, dtype=np.float64)
    else:
        closing_speed = min(float(closing_cap), dist / float(glide_time))
        v_des = (-pos / dist) * closing_speed

    accel = float(vel_gain) * (v_des - vel)
    if cw_compensation:
        accel = accel - cw_accel(pos, vel)

    return np.clip(accel, -1.0, 1.0).astype(np.float64)


def hot_approach_action(
    rel_pos: np.ndarray,
    rel_vel: Optional[np.ndarray] = None,
    max_accel: float = 1.0,
) -> np.ndarray:
    """Deliberately unsafe guidance: full thrust straight at the target, never brakes.

    This is the adversarial case for the mission safety interlock. Left unchecked
    it drives the chaser through the target at metres per second; with the
    interlock active, its accelerating thrust is rejected inside the 5 m bubble.
    """
    pos = np.asarray(rel_pos, dtype=np.float64)
    dist = float(np.linalg.norm(pos))
    if dist < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return ((-pos / dist) * float(max_accel)).astype(np.float64)
