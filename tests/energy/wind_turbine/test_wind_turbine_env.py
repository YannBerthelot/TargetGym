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
    available_power,
    check_is_terminal,
    compute_reward,
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


def test_cp_is_non_negative_everywhere(params):
    """The fit goes negative outside its valid region; the model must not."""
    for lam in np.linspace(0.5, 20.0, 40):
        for beta in np.linspace(0.0, 40.0, 20):
            assert float(power_coefficient(float(lam), float(beta), params)) >= 0.0


def test_available_power_rises_with_wind(params):
    w = omega_rated(params)
    powers = [float(available_power(v, w, params)) for v in (12.0, 15.0, 18.0)]
    assert powers[0] < powers[1] < powers[2]


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
        jnp.array([w, 0.0, small_torque]),
        action=(0.0, small_torque),
        v_wind=15.0,
        params=params,
    )
    v_large, _ = compute_velocity(
        jnp.array([w, 0.0, large_torque]),
        action=(0.0, large_torque),
        v_wind=15.0,
        params=params,
    )
    assert float(v_small[0]) > 0.0  # under-braked -> speeds up
    assert float(v_large[0]) < 0.0  # over-braked -> slows down


def test_pitch_actuator_is_rate_limited(params):
    """A large pitch demand cannot move faster than pitch_rate_max."""
    w = omega_rated(params)
    v, _ = compute_velocity(
        jnp.array([w, 0.0, 40_000.0]),
        action=(params.pitch_max, 40_000.0),  # demand full feather
        v_wind=15.0,
        params=params,
    )
    assert float(v[1]) == pytest.approx(params.pitch_rate_max, rel=1e-6)


def test_electrical_power_is_torque_times_speed(params):
    """P = eta N tau w -- and it scales the way that implies."""
    w = omega_rated(params)
    p1 = float(electrical_power(w, 20_000.0, params))
    p2 = float(electrical_power(w, 40_000.0, params))
    assert p2 == pytest.approx(2.0 * p1, rel=1e-6)


def test_state_stays_finite_over_an_episode(params):
    env = WindTurbine()
    p = params.replace(max_steps_in_episode=200)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    for _ in range(200):
        key, sub = jax.random.split(key)
        _, state, reward, terminated, _ = env.step_env(sub, state, jnp.zeros(2), p)
        for leaf in jax.tree_util.tree_leaves(state):
            assert np.all(np.isfinite(np.asarray(leaf)))
        assert np.isfinite(float(reward))
        if bool(terminated):
            break


# ---------------------------------------------------------------------------
# 4. Task interface
# ---------------------------------------------------------------------------


def test_observation_hides_the_wind(params):
    """A nacelle anemometer sits in the rotor wake; the controller must infer."""
    env = WindTurbine()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    state = state.replace(v_wind=42.42, v_mean=37.37)
    obs = env.get_obs(state, params)
    assert obs.shape == (5,)
    for hidden in (42.42, 37.37):
        assert not bool(jnp.any(jnp.isclose(obs, hidden)))


def test_overspeed_and_underspeed_both_terminate(params):
    """Losing rotor-speed regulation trips the turbine, either way."""
    env = WindTurbine()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    w = float(omega_rated(params))
    fast = state.replace(omega=w * (params.overspeed_factor + 0.05))
    slow = state.replace(omega=w * (params.underspeed_factor - 0.05))
    assert bool(check_is_terminal(fast, params)[0])
    assert bool(check_is_terminal(slow, params)[0])
    assert not bool(check_is_terminal(state.replace(omega=w), params)[0])


def test_reward_peaks_on_the_power_setpoint(params):
    env = WindTurbine()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    w = omega_rated(params)
    target = 4.0e6
    on_torque = target / (params.eta_gen * params.N_gear * float(w))
    on = state.replace(
        omega=w, torque=on_torque, target_power=target, pitch=5.0, pitch_cmd=5.0
    )
    off = on.replace(torque=on_torque * 0.5)
    assert float(compute_reward(on, params)) == pytest.approx(1.0, abs=1e-3)
    assert float(compute_reward(on, params)) > float(compute_reward(off, params))


def test_wind_range_stays_above_rated(params):
    """Below rated the setpoint is unachievable, so the task would be ill-posed."""
    assert params.v_mean_range[0] > params.v_rated + 1.5


def test_reset_starts_regulating_at_rated_speed(params):
    env = WindTurbine()
    for seed in range(5):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        assert isinstance(state, WindTurbineState)
        assert float(state.omega) == pytest.approx(float(omega_rated(params)), rel=1e-6)
        assert params.target_power_range[0] <= float(state.target_power)
        assert float(state.target_power) <= params.target_power_range[1]
