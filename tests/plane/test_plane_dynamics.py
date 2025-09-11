import jax.numpy as jnp
import pytest

from plane_env.plane.dynamics import (
    aero_coefficients,
    clamp_altitude,
    compute_acceleration,
    compute_speed_and_pos_from_acceleration,
)

# Assume your functions are imported:
# from aircraft_model import aero_coefficients, compute_acceleration, compute_speed_and_pos_from_acceleration, clamp_altitude


def test_aero_coefficients_cl_cd_ranges():
    """Ensure CL and CD are reasonable for typical AoA."""
    aoas = [-10, 0, 5, 10, 15]
    machs = [0.0, 0.5, 0.8, 0.85]

    for aoa in aoas:
        for M in machs:
            CL, CD = aero_coefficients(aoa, M)
            # CL should be within physical limits
            assert -1.0 <= CL <= 2.0, f"CL out of range for AoA {aoa}, M={M}, got {CL}"
            # CD should be positive
            assert CD > 0, f"CD negative for AoA {aoa}, M={M}"


def test_compute_acceleration_consistency():
    """Check that accelerations are reasonable and moments are finite."""
    F_x, F_z, alpha_y, metrics = compute_acceleration(
        thrust=50000,
        stick=0.0,
        m=70000,
        gravity=9.81,
        x_dot=200.0,
        z_dot=0.0,
        frontal_surface=20.0,
        wings_surface=122.6,
        alpha=jnp.deg2rad(2.0),
        M=0.6,
        M_crit=0.82,
        C_x0=0.02,
        C_z0=0.3,
        gamma=0.01,
        theta=0.02,
        rho=1.225,
    )
    # Linear accelerations should be within reason
    assert -50 <= F_x <= 50, f"F_x acceleration unreasonable: {F_x}"
    assert -50 <= F_z <= 50, f"F_z acceleration unreasonable: {F_z}"
    # Angular acceleration should be finite
    assert jnp.isfinite(alpha_y), "Angular acceleration is not finite"


def test_clamp_altitude():
    """Ensure aircraft does not go below ground."""
    z_clamped, z_dot_clamped = clamp_altitude(-10, -5)
    assert z_clamped == 0, "Altitude not clamped at ground"
    assert z_dot_clamped == 0, "Descending velocity not zeroed at ground"

    z_clamped, z_dot_clamped = clamp_altitude(100, -10)
    assert z_clamped == 100, "Altitude incorrectly clamped above ground"
    assert z_dot_clamped == -10, "Velocity incorrectly modified above ground"


def test_compute_speed_and_pos_integration():
    """Check semi-implicit Euler integration produces reasonable outputs."""
    V_x, V_z, theta_dot, x, z, theta = compute_speed_and_pos_from_acceleration(
        V_x=200.0,
        V_z=0.0,
        theta_dot=0.01,
        x=0.0,
        z=1000.0,
        theta=0.05,
        a_x=1.0,
        a_z=-9.8,
        alpha_y=0.001,
        delta_t=0.1,
    )
    # velocities updated
    assert 200.0 <= V_x <= 201.0, f"V_x integration unexpected: {V_x}"
    assert -1.0 <= V_z <= 0.0, f"V_z integration unexpected: {V_z}"
    # positions updated
    assert x > 0, "X position did not increase"
    assert z > 0, "Z position should remain above ground"
    # angle finite
    assert jnp.isfinite(theta), "Theta is not finite"
