"""Physics validation for the NREL 5 MW reference wind turbine.

Enforces ``src/target_gym/energy/wind_turbine/PHYSICS.md``. Assertions are the
published ratings, the Cp surface's shape, and emergent power at the rated
point -- not restatements of the equations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.energy.wind_turbine.env import (
    WindTurbineParams,
    WindTurbineState,
    aerodynamic_torque,
    compute_velocity,
    drivetrain_inertia,
    electrical_power,
    omega_rated,
    power_coefficient,
    rated_generator_torque,
    rotor_area,
)
from target_gym.energy.wind_turbine.env_jax import WindTurbine


@pytest.fixture(scope="module")
def params():
    return WindTurbineParams()


# ---------------------------------------------------------------------------
# 1. Published ratings (PHYSICS.md §2)
# ---------------------------------------------------------------------------


def test_rated_generator_torque_matches_the_reference(params):
    """43 093.55 N.m -- pins rated power, rated speed, gearbox and efficiency.

    The load-bearing check: it is a joint consequence of four parameters, so
    matching it validates all of them at once.
    """
    assert float(rated_generator_torque(params)) == pytest.approx(43_093.55, rel=1e-4)


def test_rated_power_emerges_at_the_rated_operating_point(params):
    """Aero power at rated wind and speed, zero pitch, should give ~5 MW.

    Emergent rather than imposed: nothing in the model sets this.
    """
    _, power = aerodynamic_torque(omega_rated(params), params.v_rated, 0.0, params)
    electrical = float(power) * params.eta_gen
    assert 4.5e6 < electrical < 5.5e6


def test_rotor_geometry_and_inertia(params):
    assert float(rotor_area(params)) == pytest.approx(np.pi * 63.0**2, rel=1e-6)
    assert float(omega_rated(params)) == pytest.approx(12.1 * 2 * np.pi / 60, rel=1e-6)
    # Referred to the rotor: J_rotor + N^2 J_gen.
    assert float(drivetrain_inertia(params)) == pytest.approx(4.379e7, rel=0.01)


def test_rotor_time_constant_is_seconds_not_minutes(params):
    """J w^2 / P sets how fast the rotor responds; ~14 s for this machine."""
    tau = (
        float(drivetrain_inertia(params))
        * float(omega_rated(params)) ** 2
        / params.P_rated
    )
    assert 5.0 < tau < 30.0


# ---------------------------------------------------------------------------
# 2. The Cp surface
# ---------------------------------------------------------------------------


def test_cp_peak_matches_the_reference_value(params):
    """Cp_max ~0.48 near TSR 7-9, and never above the Betz limit."""
    lams = np.linspace(1.0, 15.0, 500)
    cps = np.array([float(power_coefficient(float(l), 0.0, params)) for l in lams])
    peak = cps.max()
    assert 0.44 < peak < 0.52
    assert 6.5 < lams[int(cps.argmax())] < 9.0
    assert np.all(cps <= 0.593 + 1e-9)  # Betz


def test_pitching_sheds_power(params):
    """Feathering the blades must reduce Cp at fixed tip-speed ratio."""
    cps = [float(power_coefficient(7.0, b, params)) for b in (0.0, 5.0, 10.0, 20.0)]
    assert all(a > b for a, b in zip(cps, cps[1:]))


# ---------------------------------------------------------------------------
# 3. Dynamics
# ---------------------------------------------------------------------------


def test_excess_aerodynamic_torque_accelerates_the_rotor(params):
    """Sign check on the drivetrain balance."""
    w = omega_rated(params)
    tau_aero, _ = aerodynamic_torque(w, 15.0, 0.0, params)
    small_torque = float(tau_aero) / params.N_gear * 0.5
    large_torque = float(tau_aero) / params.N_gear * 1.5
    v_small, _ = compute_velocity(
        jnp.array([w, 0.0, small_torque, 0.0, 0.0]),
        action=(0.0, 0.0),
        v_wind=15.0,
        params=params,
    )
    v_large, _ = compute_velocity(
        jnp.array([w, 0.0, large_torque, 0.0, 0.0]),
        action=(0.0, 0.0),
        v_wind=15.0,
        params=params,
    )
    assert float(v_small[0]) > 0.0  # under-braked -> speeds up
    assert float(v_large[0]) < 0.0  # over-braked -> slows down


def test_electrical_power_is_torque_times_speed(params):
    """P = eta N tau w -- and it scales the way that implies."""
    w = omega_rated(params)
    p1 = float(electrical_power(w, 20_000.0, params))
    p2 = float(electrical_power(w, 40_000.0, params))
    assert p2 == pytest.approx(2.0 * p1, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Task interface
# ---------------------------------------------------------------------------


def test_reset_starts_regulating_at_rated_speed(params):
    env = WindTurbine()
    for seed in range(5):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        assert isinstance(state, WindTurbineState)
        assert float(state.omega) == pytest.approx(float(omega_rated(params)), rel=1e-6)
        assert float(state.target_power) == pytest.approx(
            params.rating * params.P_rated
        )
