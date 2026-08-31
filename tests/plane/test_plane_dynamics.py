import jax.numpy as jnp
import numpy as np

from target_gym.integration import (
    integrate_dynamics,
)
from target_gym.plane.dynamics import (
    aero_coefficients,
    clamp_altitude,
    compute_acceleration,
)
from target_gym.plane.env import PlaneParams

# Assume your functions are imported:
# from aircraft_model import aero_coefficients, compute_acceleration, integrate_dynamics, clamp_altitude


def test_aero_coefficients_cl_cd_ranges():
    """CL/CD stay in physical bounds.

    Split by regime. Incompressibly, CL must respect the wing's own stall
    limits [CL_min, CL_max]. At high Mach the Prandtl-Glauert 1/beta factor
    inflates |CL| beyond them -- that is deviation D3 in PHYSICS.md, bounded
    only by the model's final +/-2 clip. Asserting one loose bound across both
    regimes (the old `-1.0 <= CL <= 2.0`) tested neither properly: it was
    calibrated to the too-shallow lift slope fixed in D1/D2.
    """
    params = PlaneParams()
    aoas = [-15, -10, 0, 5, 10, 15, 20]

    # The stall clamp is applied *before* the 1/beta compressibility factor,
    # so the attainable band scales with 1/beta and is finally hard-clipped
    # at +/-2. This is the model's exact contract.
    for aoa in aoas:
        for M in (0.0, 0.3, 0.5, 0.8, 0.85):
            beta = np.sqrt(max(1e-6, 1.0 - M**2))
            lo = max(params.CL_min / beta, -2.0) - 1e-5
            hi = min(params.CL_max / beta, 2.0) + 1e-5
            CL, CD = aero_coefficients(aoa, M, params=params)
            assert lo <= CL <= hi, f"CL out of band for AoA {aoa}, M={M}: {CL}"
            assert CD > 0, f"CD non-positive for AoA {aoa}, M={M}"

    # Incompressibly the band is exactly the wing's own stall limits.
    for aoa in aoas:
        CL, _ = aero_coefficients(aoa, 0.0, params=params)
        assert params.CL_min - 1e-5 <= CL <= params.CL_max + 1e-5


def test_compressibility_amplifies_lift_magnitude():
    """Prandtl-Glauert: |CL| grows with Mach in the attached-flow region."""
    params = PlaneParams()
    lows = [
        abs(float(aero_coefficients(5.0, m, params=params)[0])) for m in (0.0, 0.5, 0.8)
    ]
    assert lows[0] < lows[1] < lows[2]


def test_compute_acceleration_consistency():
    """Check that accelerations are reasonable and moments are finite."""
    thrust = 50000
    stick = 0.0
    x_dot = 200.0
    z_dot = 0.0
    theta_dot = None  # not needed here
    velocities = (x_dot, z_dot, theta_dot)
    positions = (None, 1000, 0.02)
    params = PlaneParams()
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
    (V_x, V_z, theta_dot), (x, z, theta), _ = integrate_dynamics(
        velocities=velocities,
        positions=positions,
        delta_t=0.1,
        compute_acceleration=lambda x, y: (jnp.array([1.0, -9.8, 0.001]), None),
    )
    # velocities updated
    assert 200.0 <= V_x <= 201.0, f"V_x integration unexpected: {V_x}"
    assert -1.0 <= V_z <= 0.0, f"V_z integration unexpected: {V_z}"
    # positions updated
    assert x > 0, "X position did not increase"
    assert z > 0, "Z position should remain above ground"
    # angle finite
    assert jnp.isfinite(theta), "Theta is not finite"
