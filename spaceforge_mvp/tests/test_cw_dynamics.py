"""Tests for Clohessy-Wiltshire (CW) relative dynamics using exact State Transition Matrix (STM)."""

import numpy as np

from spaceforge_mvp.core.cw_dynamics import cw_accel, cw_step, cw_stm_matrices


def test_case_1_origin_at_rest_with_thrust():
    """
    Case 1: Origin at rest with pure external thrust.
    Input:
      pos = [0.0, 0.0, 0.0]
      vel = [0.0, 0.0, 0.0]
      accel_cmd = [0.1, -0.05, 0.02]
      dt = 10.0, n = 0.00113

    Expected analytical output via CW STM:
      Instantaneous accel at t=0: [0.0, 0.0, 0.0]
      x(10s) = Phi_rr*x0 + Phi_rv*v0 + Gamma_r*a_cmd
    """
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    accel_cmd = np.array([0.1, -0.05, 0.02], dtype=np.float64)
    dt = 10.0
    n = 0.00113

    accel = cw_accel(pos, vel, n=n)
    next_pos, next_vel = cw_step(pos, vel, accel_cmd, dt=dt, n=n)

    # Compute expected via analytical CW STM matrices
    Phi, Gamma = cw_stm_matrices(dt=dt, n=n)
    x_0 = np.concatenate([pos, vel])
    x_expected = Phi @ x_0 + Gamma @ accel_cmd
    expected_pos = x_expected[0:3]
    expected_vel = x_expected[3:6]

    np.testing.assert_allclose(accel, [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(next_pos, expected_pos, atol=1e-9)
    np.testing.assert_allclose(next_vel, expected_vel, atol=1e-9)


def test_case_2_radial_and_cross_track_displacement_no_thrust():
    """
    Case 2: Radial and cross-track displacement with zero initial velocity and zero thrust.
    Input:
      pos = [1000.0, 0.0, 500.0]
      vel = [0.0, 0.0, 0.0]
      accel_cmd = [0.0, 0.0, 0.0]
      dt = 5.0, n = 0.00113

    Expected analytical output via CW STM:
      ax(0) = 3*n^2*x = 3.8307e-6 * 1000.0 = 0.0038307
      ay(0) = 0.0
      az(0) = -n^2*z = -1.2769e-6 * 500.0 = -0.00063845
    """
    pos = np.array([1000.0, 0.0, 500.0], dtype=np.float64)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    accel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dt = 5.0
    n = 0.00113

    expected_accel = np.array([0.0038307, 0.0, -0.00063845])
    accel = cw_accel(pos, vel, n=n)
    next_pos, next_vel = cw_step(pos, vel, accel_cmd, dt=dt, n=n)

    Phi, Gamma = cw_stm_matrices(dt=dt, n=n)
    x_0 = np.concatenate([pos, vel])
    x_expected = Phi @ x_0 + Gamma @ accel_cmd

    np.testing.assert_allclose(accel, expected_accel, atol=1e-9)
    np.testing.assert_allclose(next_pos, x_expected[0:3], atol=1e-9)
    np.testing.assert_allclose(next_vel, x_expected[3:6], atol=1e-9)


def test_case_3_coupled_state_with_commanded_acceleration():
    """
    Case 3: Coupled 3D position and velocity with active commanded control acceleration.
    Input:
      pos = [100.0, -200.0, 50.0]
      vel = [1.0, 2.0, -0.5]
      accel_cmd = [0.01, -0.02, 0.005]
      dt = 2.0, n = 0.00113
    """
    pos = np.array([100.0, -200.0, 50.0], dtype=np.float64)
    vel = np.array([1.0, 2.0, -0.5], dtype=np.float64)
    accel_cmd = np.array([0.01, -0.02, 0.005], dtype=np.float64)
    dt = 2.0
    n = 0.00113

    expected_accel = np.array([0.00490307, -0.00226, -0.000063845])
    accel = cw_accel(pos, vel, n=n)
    next_pos, next_vel = cw_step(pos, vel, accel_cmd, dt=dt, n=n)

    Phi, Gamma = cw_stm_matrices(dt=dt, n=n)
    x_0 = np.concatenate([pos, vel])
    x_expected = Phi @ x_0 + Gamma @ accel_cmd

    np.testing.assert_allclose(accel, expected_accel, atol=1e-9)
    np.testing.assert_allclose(next_pos, x_expected[0:3], atol=1e-9)
    np.testing.assert_allclose(next_vel, x_expected[3:6], atol=1e-9)
