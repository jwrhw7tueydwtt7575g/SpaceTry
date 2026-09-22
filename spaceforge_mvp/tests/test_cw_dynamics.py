"""Tests for Clohessy-Wiltshire (CW) relative dynamics."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure spaceforge_mvp is importable from various pytest working directories
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.core.cw_dynamics import cw_accel, cw_step
except ImportError:
    from core.cw_dynamics import cw_accel, cw_step


def test_case_1_origin_at_rest_with_thrust():
    """
    Case 1: Origin at rest with pure external thrust.
    Input:
      pos = [0.0, 0.0, 0.0]
      vel = [0.0, 0.0, 0.0]
      accel_cmd = [0.1, -0.05, 0.02]
      dt = 10.0, n = 0.00113

    Hand-calculated expected output:
      ax = 2*n*vy + 3*n^2*x = 2*(0.00113)*0 + 3*(0.00113^2)*0 = 0.0
      ay = -2*n*vx = -2*(0.00113)*0 = 0.0
      az = -n^2*z = -(0.00113^2)*0 = 0.0
      expected_accel = [0.0, 0.0, 0.0]

      a_total = expected_accel + accel_cmd = [0.1, -0.05, 0.02]
      next_pos = pos + vel * dt = [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0]*10.0 = [0.0, 0.0, 0.0]
      next_vel = vel + a_total * dt = [0.0, 0.0, 0.0] + [0.1, -0.05, 0.02]*10.0 = [1.0, -0.5, 0.2]
    """
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    accel_cmd = np.array([0.1, -0.05, 0.02])
    dt = 10.0
    n = 0.00113

    expected_accel = np.array([0.0, 0.0, 0.0])
    expected_pos = np.array([0.0, 0.0, 0.0])
    expected_vel = np.array([1.0, -0.5, 0.2])

    accel = cw_accel(pos, vel, n=n)
    next_pos, next_vel = cw_step(pos, vel, accel_cmd, dt, n=n)

    np.testing.assert_allclose(accel, expected_accel, atol=1e-9)
    np.testing.assert_allclose(next_pos, expected_pos, atol=1e-9)
    np.testing.assert_allclose(next_vel, expected_vel, atol=1e-9)


def test_case_2_radial_and_cross_track_displacement_no_thrust():
    """
    Case 2: Pure displacement along radial (x) and cross-track (z) axes with zero initial velocity.
    Input:
      pos = [1000.0, 0.0, 500.0]
      vel = [0.0, 0.0, 0.0]
      accel_cmd = [0.0, 0.0, 0.0]
      dt = 5.0, n = 0.00113

    Hand-calculated expected output:
      n = 0.00113 => n^2 = 1.2769e-6, 3*n^2 = 3.8307e-6
      ax = 2*n*vy + 3*n^2*x = 2*(0.00113)*0.0 + 3*(1.2769e-6)*1000.0
         = 3.8307e-6 * 1000.0 = 0.0038307
      ay = -2*n*vx = -2*(0.00113)*0.0 = 0.0
      az = -n^2*z = -1.2769e-6 * 500.0 = -0.00063845
      expected_accel = [0.0038307, 0.0, -0.00063845]

      a_total = expected_accel + [0.0, 0.0, 0.0] = [0.0038307, 0.0, -0.00063845]
      next_pos = pos + vel * dt = [1000.0, 0.0, 500.0] + [0.0, 0.0, 0.0]*5.0 = [1000.0, 0.0, 500.0]
      next_vel = vel + a_total * dt:
        vx = 0.0 + 0.0038307 * 5.0 = 0.0191535
        vy = 0.0
        vz = 0.0 + (-0.00063845) * 5.0 = -0.00319225
      expected_next_pos = [1000.0, 0.0, 500.0]
      expected_next_vel = [0.0191535, 0.0, -0.00319225]
    """
    pos = np.array([1000.0, 0.0, 500.0])
    vel = np.array([0.0, 0.0, 0.0])
    accel_cmd = np.array([0.0, 0.0, 0.0])
    dt = 5.0
    n = 0.00113

    expected_accel = np.array([0.0038307, 0.0, -0.00063845])
    expected_pos = np.array([1000.0, 0.0, 500.0])
    expected_vel = np.array([0.0191535, 0.0, -0.00319225])

    accel = cw_accel(pos, vel, n=n)
    next_pos, next_vel = cw_step(pos, vel, accel_cmd, dt, n=n)

    np.testing.assert_allclose(accel, expected_accel, atol=1e-9)
    np.testing.assert_allclose(next_pos, expected_pos, atol=1e-9)
    np.testing.assert_allclose(next_vel, expected_vel, atol=1e-9)


def test_case_3_coupled_state_with_commanded_acceleration():
    """
    Case 3: Coupled 3D position and velocity with active commanded control acceleration.
    Input:
      pos = [100.0, -200.0, 50.0]
      vel = [1.0, 2.0, -0.5]
      accel_cmd = [0.01, -0.02, 0.005]
      dt = 2.0, n = 0.00113

    Hand-calculated expected output:
      2*n = 0.00226, n^2 = 1.2769e-6, 3*n^2 = 3.8307e-6
      ax = 2*n*vy + 3*n^2*x = 0.00226*2.0 + 3.8307e-6*100.0
         = 0.00452 + 0.00038307 = 0.00490307
      ay = -2*n*vx = -0.00226 * 1.0 = -0.00226
      az = -n^2*z = -1.2769e-6 * 50.0 = -0.000063845
      expected_accel = [0.00490307, -0.00226, -0.000063845]

      a_total:
        ax_tot = 0.00490307 + 0.01 = 0.01490307
        ay_tot = -0.00226 + (-0.02) = -0.02226
        az_tot = -0.000063845 + 0.005 = 0.004936155

      next_pos = pos + vel * dt:
        x = 100.0 + 1.0 * 2.0 = 102.0
        y = -200.0 + 2.0 * 2.0 = -196.0
        z = 50.0 + (-0.5) * 2.0 = 49.0
      expected_next_pos = [102.0, -196.0, 49.0]

      next_vel = vel + a_total * dt:
        vx = 1.0 + 0.01490307 * 2.0 = 1.02980614
        vy = 2.0 + (-0.02226) * 2.0 = 1.95548
        vz = -0.5 + 0.004936155 * 2.0 = -0.49012769
      expected_next_vel = [1.02980614, 1.95548, -0.49012769]
    """
    pos = np.array([100.0, -200.0, 50.0])
    vel = np.array([1.0, 2.0, -0.5])
    accel_cmd = np.array([0.01, -0.02, 0.005])
    dt = 2.0
    n = 0.00113

    expected_accel = np.array([0.00490307, -0.00226, -0.000063845])
    expected_pos = np.array([102.0, -196.0, 49.0])
    expected_vel = np.array([1.02980614, 1.95548, -0.49012769])

    accel = cw_accel(pos, vel, n=n)
    next_pos, next_vel = cw_step(pos, vel, accel_cmd, dt, n=n)

    np.testing.assert_allclose(accel, expected_accel, atol=1e-9)
    np.testing.assert_allclose(next_pos, expected_pos, atol=1e-9)
    np.testing.assert_allclose(next_vel, expected_vel, atol=1e-9)
