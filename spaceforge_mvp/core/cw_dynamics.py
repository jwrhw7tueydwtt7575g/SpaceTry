"""Clohessy-Wiltshire (CW / Hill) relative orbital dynamics with analytical State Transition Matrix (STM) propagator."""

import numpy as np
from spaceforge_mvp.config import PHYSICS_CONFIG


def cw_accel(pos, vel, n: float = PHYSICS_CONFIG.n) -> np.ndarray:
    """
    Calculate instantaneous orbital relative acceleration using Clohessy-Wiltshire equations.

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

    x, _, z = pos_arr[0], pos_arr[1], pos_arr[2]
    vx, vy, _ = vel_arr[0], vel_arr[1], vel_arr[2]

    ax = 2.0 * n * vy + 3.0 * (n ** 2) * x
    ay = -2.0 * n * vx
    az = -(n ** 2) * z

    return np.array([ax, ay, az], dtype=np.float64)


def cw_stm_matrices(dt: float, n: float = PHYSICS_CONFIG.n):
    """
    Construct the 6x6 analytical Clohessy-Wiltshire State Transition Matrix (Phi)
    and the 6x3 forced response input matrix (Gamma) over interval dt.

    Returns
    -------
    Phi : np.ndarray (6, 6)
        Unforced state transition matrix.
    Gamma : np.ndarray (6, 3)
        Forced response control input matrix.
    """
    nt = n * dt
    sin_nt = np.sin(nt)
    cos_nt = np.cos(nt)

    Phi_rr = np.array([
        [4.0 - 3.0 * cos_nt, 0.0, 0.0],
        [6.0 * (sin_nt - nt), 1.0, 0.0],
        [0.0, 0.0, cos_nt],
    ], dtype=np.float64)

    Phi_rv = np.array([
        [sin_nt / n, (2.0 / n) * (1.0 - cos_nt), 0.0],
        [(2.0 / n) * (cos_nt - 1.0), (1.0 / n) * (4.0 * sin_nt - 3.0 * nt), 0.0],
        [0.0, 0.0, sin_nt / n],
    ], dtype=np.float64)

    Phi_vr = np.array([
        [3.0 * n * sin_nt, 0.0, 0.0],
        [6.0 * n * (cos_nt - 1.0), 0.0, 0.0],
        [0.0, 0.0, -n * sin_nt],
    ], dtype=np.float64)

    Phi_vv = np.array([
        [cos_nt, 2.0 * sin_nt, 0.0],
        [-2.0 * sin_nt, 4.0 * cos_nt - 3.0, 0.0],
        [0.0, 0.0, cos_nt],
    ], dtype=np.float64)

    Phi = np.block([
        [Phi_rr, Phi_rv],
        [Phi_vr, Phi_vv],
    ])

    # Gamma_r is integral of Phi_rv over [0, dt]
    Gamma_r = np.array([
        [(1.0 - cos_nt) / (n ** 2), (2.0 / (n ** 2)) * (nt - sin_nt), 0.0],
        [(2.0 / (n ** 2)) * (sin_nt - nt), (4.0 * (1.0 - cos_nt) - 1.5 * (nt ** 2)) / (n ** 2), 0.0],
        [0.0, 0.0, (1.0 - cos_nt) / (n ** 2)],
    ], dtype=np.float64)

    # Gamma_v is integral of Phi_vv over [0, dt] (which equals Phi_rv)
    Gamma_v = Phi_rv.copy()

    Gamma = np.vstack([Gamma_r, Gamma_v])

    return Phi, Gamma


def cw_step(pos, vel, accel_cmd, dt: float = PHYSICS_CONFIG.dt, n: float = PHYSICS_CONFIG.n):
    """
    Propagate spacecraft relative state over timestep dt using exact Clohessy-Wiltshire State Transition Matrix (STM).

    Parameters
    ----------
    pos : array-like
        Current relative position [x, y, z] in meters.
    vel : array-like
        Current relative velocity [vx, vy, vz] in m/s.
    accel_cmd : array-like
        External commanded acceleration [ax, ay, az] in m/s^2.
    dt : float, optional
        Integration timestep in seconds (default from PHYSICS_CONFIG).
    n : float, optional
        Mean motion in rad/s (default from PHYSICS_CONFIG).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (next_pos, next_vel) exact analytical updated state vectors.
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)
    cmd_arr = np.asarray(accel_cmd, dtype=np.float64)

    x_0 = np.concatenate([pos_arr, vel_arr])
    Phi, Gamma = cw_stm_matrices(dt=dt, n=n)

    x_next = Phi @ x_0 + Gamma @ cmd_arr
    next_pos = x_next[0:3]
    next_vel = x_next[3:6]

    return next_pos, next_vel
