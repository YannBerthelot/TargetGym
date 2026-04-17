"""Tests for the 3D plane dynamics module."""

import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane.dynamics import aero_coefficients
from target_gym.plane3d.dynamics import (
    compute_acceleration_3d,
    compute_alpha_3d,
    compute_gamma_3d,
    compute_next_aileron,
    compute_psi,
    compute_velocity_3d,
    newton_second_law_3d,
)
from target_gym.plane3d.env import PlaneParams3D


class TestDerivedAngles:
    def test_gamma_level_flight(self):
        """gamma = 0 for level flight."""
        gamma = compute_gamma_3d(200.0, 0.0, 0.0)
        assert gamma == pytest.approx(0.0, abs=1e-6)

    def test_gamma_climbing(self):
        """gamma > 0 when z_dot > 0."""
        gamma = compute_gamma_3d(200.0, 0.0, 10.0)
        assert float(gamma) > 0

    def test_psi_along_x(self):
        """psi = 0 when flying along +x."""
        psi = compute_psi(200.0, 0.0)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_psi_along_y(self):
        """psi = pi/2 when flying along +y."""
        psi = compute_psi(0.0, 200.0)
        assert psi == pytest.approx(jnp.pi / 2, abs=1e-6)

    def test_psi_negative_y(self):
        """psi = -pi/2 when flying along -y."""
        psi = compute_psi(0.0, -200.0)
        assert psi == pytest.approx(-jnp.pi / 2, abs=1e-6)

    def test_alpha_level_zero_pitch(self):
        """alpha = 0 for level flight with zero pitch."""
        alpha, gamma = compute_alpha_3d(0.0, 200.0, 0.0, 0.0)
        assert alpha == pytest.approx(0.0, abs=1e-6)

    def test_velocity_3d(self):
        """Total velocity from components."""
        V = compute_velocity_3d(3.0, 4.0, 0.0)
        assert V == pytest.approx(5.0, abs=1e-6)


class TestNewtonSecondLaw3D:
    """Test the 3D force decomposition."""

    def test_matches_2d_when_no_bank_no_lateral(self):
        """With phi=0, psi=0, the 3D result should match the 2D forces in x and z."""
        from target_gym.plane.dynamics import newton_second_law

        thrust, lift, drag, P = 100_000.0, 300_000.0, 50_000.0, 720_000.0
        gamma, theta = 0.05, 0.08

        F_x_2d, F_z_2d = newton_second_law(thrust, lift, drag, P, gamma, theta)
        F_x_3d, F_y_3d, F_z_3d = newton_second_law_3d(
            thrust, lift, drag, P, gamma, theta, phi=0.0, psi=0.0
        )

        assert float(F_x_3d) == pytest.approx(float(F_x_2d), rel=1e-4)
        assert float(F_z_3d) == pytest.approx(float(F_z_2d), rel=1e-4)
        assert float(F_y_3d) == pytest.approx(0.0, abs=1e-2)

    def test_bank_creates_lateral_force(self):
        """Banking should create a lateral force component."""
        thrust, lift, drag, P = 100_000.0, 300_000.0, 50_000.0, 720_000.0
        gamma, theta = 0.0, 0.0

        _, F_y_level, F_z_level = newton_second_law_3d(
            thrust, lift, drag, P, gamma, theta, phi=0.0, psi=0.0
        )
        _, F_y_banked, F_z_banked = newton_second_law_3d(
            thrust, lift, drag, P, gamma, theta, phi=0.3, psi=0.0
        )

        # Banking should create lateral force
        assert abs(float(F_y_banked)) > abs(float(F_y_level))
        # Vertical lift decreases with bank
        assert float(F_z_banked) < float(F_z_level)

    def test_right_bank_positive_y_force(self):
        """Right bank (phi>0) with heading along +x should push in +y."""
        thrust, lift, drag, P = 0.0, 300_000.0, 0.0, 0.0
        _, F_y, _ = newton_second_law_3d(
            thrust, lift, drag, P, gamma=0.0, theta=0.0, phi=0.3, psi=0.0
        )
        assert float(F_y) > 0

    def test_left_bank_negative_y_force(self):
        """Left bank (phi<0) with heading along +x should push in -y."""
        thrust, lift, drag, P = 0.0, 300_000.0, 0.0, 0.0
        _, F_y, _ = newton_second_law_3d(
            thrust, lift, drag, P, gamma=0.0, theta=0.0, phi=-0.3, psi=0.0
        )
        assert float(F_y) < 0

    def test_weight_only_acts_downward(self):
        """With no aero forces, only weight acts, pulling down in z."""
        F_x, F_y, F_z = newton_second_law_3d(
            thrust=0.0,
            lift=0.0,
            drag=0.0,
            P=100_000.0,
            gamma=0.0,
            theta=0.0,
            phi=0.0,
            psi=0.0,
        )
        assert float(F_x) == pytest.approx(0.0, abs=1e-6)
        assert float(F_y) == pytest.approx(0.0, abs=1e-6)
        assert float(F_z) == pytest.approx(-100_000.0, abs=1e-2)


class TestAileron:
    def test_compute_next_aileron_approaches_target(self):
        """Aileron should approach requested value."""
        current = 0.0
        requested = 0.3
        for _ in range(100):
            current = compute_next_aileron(requested, current, 1.0)
        assert float(current) == pytest.approx(requested, abs=0.01)

    def test_compute_next_aileron_smooth(self):
        """Aileron should not jump instantly."""
        result = compute_next_aileron(1.0, 0.0, 1.0)
        assert 0.0 < float(result) < 1.0


class TestComputeAcceleration3D:
    def test_level_flight_reasonable(self):
        """Accelerations should be finite and reasonable for level flight."""
        params = PlaneParams3D()
        velocities = jnp.array([200.0, 0.0, 0.0, 0.0, 0.0])
        positions = jnp.array([0.0, 0.0, 5000.0, 0.02, 0.0])
        thrust = 50_000.0

        accel, metrics = compute_acceleration_3d(
            velocities=velocities,
            positions=positions,
            action=(thrust, 0.0, 0.0),
            params=params,
        )
        a_x, a_y, a_z, alpha_pitch, alpha_roll = accel
        assert jnp.isfinite(a_x)
        assert jnp.isfinite(a_y)
        assert jnp.isfinite(a_z)
        assert jnp.isfinite(alpha_pitch)
        assert jnp.isfinite(alpha_roll)
        assert -50 <= float(a_x) <= 50
        assert -50 <= float(a_z) <= 50

    def test_no_lateral_accel_without_bank(self):
        """With phi=0 and y_dot=0, lateral acceleration should be near zero."""
        params = PlaneParams3D()
        velocities = jnp.array([200.0, 0.0, 0.0, 0.0, 0.0])
        positions = jnp.array([0.0, 0.0, 5000.0, 0.02, 0.0])

        accel, _ = compute_acceleration_3d(
            velocities=velocities,
            positions=positions,
            action=(50_000.0, 0.0, 0.0),
            params=params,
        )
        assert abs(float(accel[1])) < 0.5  # a_y near zero

    def test_aileron_creates_roll_moment(self):
        """Positive aileron should create a roll acceleration."""
        params = PlaneParams3D()
        velocities = jnp.array([200.0, 0.0, 0.0, 0.0, 0.0])
        positions = jnp.array([0.0, 0.0, 5000.0, 0.02, 0.0])

        accel_no_ail, _ = compute_acceleration_3d(
            velocities=velocities,
            positions=positions,
            action=(50_000.0, 0.0, 0.0),
            params=params,
        )
        accel_ail, _ = compute_acceleration_3d(
            velocities=velocities,
            positions=positions,
            action=(50_000.0, 0.0, 0.2),
            params=params,
        )
        # Aileron deflection should change roll acceleration
        assert float(accel_ail[4]) != pytest.approx(float(accel_no_ail[4]), abs=0.001)

    def test_roll_damping(self):
        """Roll rate should create a damping moment opposing the rotation."""
        params = PlaneParams3D()
        # Flying level with a positive roll rate
        velocities_rolling = jnp.array([200.0, 0.0, 0.0, 0.0, 0.3])
        positions = jnp.array([0.0, 0.0, 5000.0, 0.02, 0.0])

        accel_rolling, _ = compute_acceleration_3d(
            velocities=velocities_rolling,
            positions=positions,
            action=(50_000.0, 0.0, 0.0),
            params=params,
        )
        velocities_still = jnp.array([200.0, 0.0, 0.0, 0.0, 0.0])
        accel_still, _ = compute_acceleration_3d(
            velocities=velocities_still,
            positions=positions,
            action=(50_000.0, 0.0, 0.0),
            params=params,
        )
        # Roll damping should reduce roll acceleration when rolling
        assert float(accel_rolling[4]) < float(accel_still[4])
