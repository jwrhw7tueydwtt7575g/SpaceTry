"""Clohessy-Wiltshire (CW / Hill) relative orbital dynamics."""

import numpy as np


def cw_accel(pos, vel, n: float = 0.00113) -> np.ndarray:
    """
    Calculate orbital relative acceleration using Clohessy-Wiltshire equations.

    Equations:
        ax = 2*n*vy + 3*n^2*x
        ay = -2*n*vx
        az = -n^2*z

    Parameters
    ----------
    pos : array-like
        Relative position [x, y, z] in meters (radial, in-track, cross-track).
    vel : array-like
        Relative velocity [vx, vy, vz] in m/s.
    n : float, optional
        Mean motion in rad/s (default is 0.00113 rad/s, typical for ~400 km LEO).

    Returns
    -------
    np.ndarray
        Relative acceleration [ax, ay, az] in m/s^2.
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)

    x = pos_arr[0]
    z = pos_arr[2]
    vx = vel_arr[0]
    vy = vel_arr[1]

    ax = 2.0 * n * vy + 3.0 * (n ** 2) * x
    ay = -2.0 * n * vx
    az = -(n ** 2) * z

    return np.array([ax, ay, az], dtype=np.float64)


def cw_step(pos, vel, accel_cmd, dt: float, n: float = 0.00113):
    """
    Integrate relative spacecraft state over a single timestep using semi-implicit (symplectic) Euler integration.

    Parameters
    ----------
    pos : array-like
        Current relative position [x, y, z] in meters.
    vel : array-like
        Current relative velocity [vx, vy, vz] in m/s.
    accel_cmd : array-like
        External commanded acceleration [ux, uy, uz] in m/s^2.
    dt : float
        Integration timestep in seconds.
    n : float, optional
        Mean motion in rad/s (default is 0.00113 rad/s).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (next_pos, next_vel) updated state vectors.
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)
    cmd_arr = np.asarray(accel_cmd, dtype=np.float64)

    a_cw = cw_accel(pos_arr, vel_arr, n=n)
    a_total = a_cw + cmd_arr

    # Symplectic / Semi-Implicit Euler: update velocity first, then position using new velocity
    next_vel = vel_arr + a_total * dt
    next_pos = pos_arr + next_vel * dt

    return next_pos, next_vel
