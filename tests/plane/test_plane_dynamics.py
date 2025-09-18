import jax.numpy as jnp

from plane_env.integration import compute_velocity_and_pos_from_acceleration_integration
from plane_env.plane.dynamics import (
    aero_coefficients,
    clamp_altitude,
    compute_acceleration,
)
from plane_env.plane.env import EnvParams

# Assume your functions are imported:
# from aircraft_model import aero_coefficients, compute_acceleration, compute_velocity_and_pos_from_acceleration_integration, clamp_altitude


def test_aero_coefficients_cl_cd_ranges():
    """Ensure CL and CD are reasonable for typical AoA."""
    aoas = [-10, 0, 5, 10, 15]
    machs = [0.0, 0.5, 0.8, 0.85]

    for aoa in aoas:
        for M in machs:
            CL, CD = aero_coefficients(aoa, M, params=EnvParams())
            # CL should be within physical limits
            assert -1.0 <= CL <= 2.0, f"CL out of range for AoA {aoa}, M={M}, got {CL}"
            # CD should be positive
            assert CD > 0, f"CD negative for AoA {aoa}, M={M}"


def test_compute_acceleration_consistency():
    """Check that accelerations are reasonable and moments are finite."""
    thrust = 50000
    stick = 0.0
    x_dot = 200.0
    z_dot = 0.0
    theta_dot = None  # not needed here
    velocities = (x_dot, z_dot, theta_dot)
    positions = (None, 1000, 0.02)
    params = EnvParams()
    accelerations, metrics = compute_acceleration(
        action=(thrust, stick),
        velocities=velocities,
        positions=positions,
        params=params,
    )
    a_x = accelerations[0]
    a_z = accelerations[1]
    alpha_y = accelerations[2]
    # Linear accelerations should be within reason
    assert -50 <= a_x <= 50, f"F_x acceleration unreasonable: {a_x}"
    assert -50 <= a_z <= 50, f"F_z acceleration unreasonable: {a_z}"
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
    # accelerations = jnp.array([1.0, -9.8, 0.001])
    velocities = jnp.array([200.0, 0.0, 0.01])
    positions = jnp.array([0.0, 1000.0, 0.05])
    (V_x, V_z, theta_dot), (x, z, theta), _ = (
        compute_velocity_and_pos_from_acceleration_integration(
            velocities=velocities,
            positions=positions,
            delta_t=0.1,
            compute_acceleration=lambda x, y: (jnp.array([1.0, -9.8, 0.001]), None),
        )
    )
    # velocities updated
    assert 200.0 <= V_x <= 201.0, f"V_x integration unexpected: {V_x}"
    assert -1.0 <= V_z <= 0.0, f"V_z integration unexpected: {V_z}"
    # positions updated
    assert x > 0, "X position did not increase"
    assert z > 0, "Z position should remain above ground"
    # angle finite
    assert jnp.isfinite(theta), "Theta is not finite"
