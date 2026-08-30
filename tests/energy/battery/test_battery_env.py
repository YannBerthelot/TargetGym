"""Physics validation for the grid battery.

Enforces ``src/target_gym/energy/battery/PHYSICS.md``. The parameters are
sized to reproduce published *behaviour* rather than copied from a documented
machine, so these tests carry more of the weight than usual: round-trip
efficiency, voltage window, thermal rise and charge conservation are what make
the model defensible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.energy.battery.env import (
    BatteryParams,
    BatteryState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_velocity,
    current_for_power,
    degradation_rate,
    max_deliverable_power,
    open_circuit_voltage,
    round_trip_efficiency,
    terminal_voltage,
)
from target_gym.energy.battery.env_jax import GridBattery


@pytest.fixture(scope="module")
def params():
    return BatteryParams()


# ---------------------------------------------------------------------------
# 1. Validation targets (PHYSICS.md §2)
# ---------------------------------------------------------------------------


def test_round_trip_efficiency_is_in_the_published_band(params):
    """88-95 % for a Li-ion grid battery. This is what sizes R0."""
    one_way = float(round_trip_efficiency(params.power_max, 0.5, params))
    assert 0.88 < one_way**2 < 0.95


def test_cell_voltage_window_is_realistic(params):
    """2.7-4.2 V is the usable Li-ion window."""
    empty = float(open_circuit_voltage(0.05, params)) / params.n_series
    full = float(open_circuit_voltage(1.0, params)) / params.n_series
    assert 2.7 < empty < 3.2
    assert 4.0 < full < 4.25


def test_open_circuit_voltage_is_monotone_in_charge(params):
    socs = np.linspace(0.02, 1.0, 200)
    v = np.array([float(open_circuit_voltage(float(s), params)) for s in socs])
    assert np.all(np.diff(v) > 0.0)


def test_thermal_rise_at_rated_power_is_physical(params):
    """~10-20 K for an actively cooled pack.

    A passive UA gives a 438 K rise, which is how the cooling assumption was
    caught during derivation.
    """
    current = float(current_for_power(params.power_max, 0.5, 0.0, params))
    steady_rise = current**2 * params.R0 / params.UA_thermal
    assert 5.0 < steady_rise < 25.0


def test_thermal_time_constant_is_tens_of_minutes(params):
    tau_min = params.C_thermal / params.UA_thermal / 60.0
    assert 20.0 < tau_min < 120.0


def test_energy_budget_matches_a_two_hour_system(params):
    """10->90 % at rated power should take ~96 min for a 2 h system."""
    minutes = 0.8 * params.energy_nominal / params.power_max / 60.0
    assert 80.0 < minutes < 120.0


def test_coulomb_counting_conserves_charge(params):
    """Integrating current must move state of charge by exactly I*t/Q."""
    soc0 = 0.6
    v, _ = compute_velocity(
        jnp.array([soc0, 0.0, params.T_ambient, 0.0]),
        action=params.power_max,
        params=params,
    )
    current = float(current_for_power(params.power_max, soc0, 0.0, params))
    assert float(v[0]) == pytest.approx(-current / params.capacity_As, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Circuit behaviour
# ---------------------------------------------------------------------------


def test_power_current_solution_satisfies_the_circuit(params):
    """The solved current must actually deliver the requested power."""
    for power in (-8.0e5, -2.0e5, 2.0e5, 8.0e5):
        for soc in (0.2, 0.5, 0.9):
            i = current_for_power(power, soc, 0.0, params)
            delivered = float(terminal_voltage(i, soc, 0.0, params) * i)
            assert delivered == pytest.approx(power, rel=1e-3)


def test_discharging_lowers_charge_and_charging_raises_it(params):
    """Sign convention: positive power discharges."""
    discharge, _ = compute_velocity(
        jnp.array([0.5, 0.0, params.T_ambient, 0.0]),
        action=params.power_max,
        params=params,
    )
    charge, _ = compute_velocity(
        jnp.array([0.5, 0.0, params.T_ambient, 0.0]),
        action=-params.power_max,
        params=params,
    )
    assert float(discharge[0]) < 0.0
    assert float(charge[0]) > 0.0


def test_deliverable_power_falls_as_the_pack_empties(params):
    """The (OCV - v_rc)^2 / 4R0 limit tightens with state of charge."""
    limits = [float(max_deliverable_power(s, 0.0, params)) for s in (0.1, 0.5, 0.9)]
    assert limits[0] < limits[1] < limits[2]
    # ...and it stays above the pack rating, so the rating is the binding limit.
    assert limits[0] > params.power_max


def test_losses_grow_with_current(params):
    """Ohmic loss is quadratic, so efficiency falls as power rises."""
    effs = [float(round_trip_efficiency(p, 0.5, params)) for p in (2.0e5, 5.0e5, 1.0e6)]
    assert effs[0] > effs[1] > effs[2]


def test_same_power_costs_more_at_low_charge(params):
    """Lower OCV means more current for the same power, so more loss."""
    assert float(round_trip_efficiency(params.power_max, 0.2, params)) < float(
        round_trip_efficiency(params.power_max, 0.8, params)
    )


def test_current_heats_the_pack(params):
    """I^2 R heating must raise temperature when the pack is at ambient."""
    v, _ = compute_velocity(
        jnp.array([0.5, 0.0, params.T_ambient, 0.0]),
        action=params.power_max,
        params=params,
    )
    assert float(v[2]) > 0.0


# ---------------------------------------------------------------------------
# 3. Degradation
# ---------------------------------------------------------------------------


def test_degradation_is_positive_and_grows_with_use(params):
    """Fade only accumulates, and throughput and heat both accelerate it."""
    idle = float(degradation_rate(0.0, 25.0, params))
    working = float(degradation_rate(1500.0, 25.0, params))
    hot = float(degradation_rate(1500.0, 45.0, params))
    assert idle > 0.0
    assert working > idle
    assert hot > working


def test_degradation_is_slow_enough_to_be_a_running_cost(params):
    """Fade over one episode should be small -- a cost, not a cliff."""
    fade = float(degradation_rate(1500.0, 30.0, params)) * (
        params.delta_t * params.max_steps_in_episode
    )
    assert 0.0 < fade < 0.01  # under 1 % capacity in an hour of hard use


# ---------------------------------------------------------------------------
# 4. Task interface
# ---------------------------------------------------------------------------


def test_observation_hides_polarisation_and_fade(params):
    """Neither is measurable, and the fade is the cost being traded against."""
    env = GridBattery()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    state = state.replace(v_rc=42.42, q_loss=0.1234)
    obs = env.get_obs(state, params)
    assert obs.shape == (5,)
    for hidden in (42.42, 0.1234):
        assert not bool(jnp.any(jnp.isclose(obs, hidden)))


def test_charge_limits_terminate_the_episode(params):
    """Running empty or full is irrecoverable -- that is the point."""
    env = GridBattery()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert bool(check_is_terminal(state.replace(soc=params.soc_min - 0.01), params)[0])
    assert bool(check_is_terminal(state.replace(soc=params.soc_max + 0.01), params)[0])
    assert not bool(check_is_terminal(state.replace(soc=0.5), params)[0])


def test_overheating_terminates(params):
    env = GridBattery()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert bool(check_is_terminal(state.replace(T_cell=params.T_max + 1.0), params)[0])


def test_reward_peaks_on_dispatch(params):
    env = GridBattery()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    on = state.replace(power=5.0e5, target_power=5.0e5, soc=0.5, current=0.0)
    off = on.replace(power=0.0)
    assert float(compute_reward(on, params)) > float(compute_reward(off, params))


def test_reset_starts_rested(params):
    """A battery entering a dispatch window has been idle."""
    env = GridBattery()
    for seed in range(5):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        assert isinstance(state, BatteryState)
        assert float(state.v_rc) == 0.0
        assert float(state.q_loss) == 0.0
        assert float(state.T_cell) == pytest.approx(params.T_ambient)
        assert params.initial_soc_range[0] <= float(state.soc)
        assert float(state.soc) <= params.initial_soc_range[1]


def test_state_stays_finite_over_an_episode(params):
    env = GridBattery()
    p = params.replace(max_steps_in_episode=200)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    for _ in range(200):
        key, sub = jax.random.split(key)
        _, state, reward, terminated, _ = env.step_env(sub, state, jnp.zeros(1), p)
        for leaf in jax.tree_util.tree_leaves(state):
            assert np.all(np.isfinite(np.asarray(leaf)))
        assert np.isfinite(float(reward))
        if bool(terminated):
            break
