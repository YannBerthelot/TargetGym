"""Physics validation for the single-zone building HVAC model.

Enforces ``src/target_gym/hvac/PHYSICS.md``. Assertions are emergent figures of
merit checked against published values for a well-insulated heavyweight
dwelling -- time constant, design heat load, free-float drift -- plus
structural invariants that need no source (energy-balance signs, monotonicity,
node ordering).
"""

import jax
import jax.numpy as jnp
import pytest

from target_gym.hvac.env import (
    HVACParams,
    HVACState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    hour_of_day,
    internal_gain,
    is_occupied,
    outdoor_temperature,
    scheduled_setpoint,
    solar_gain,
    solve_air_and_surface,
    split_gains,
    total_heat_loss_coefficient,
    zone_conductances,
)
from target_gym.hvac.env_jax import BuildingHVAC

# Published targets for a well-insulated heavyweight dwelling (PHYSICS.md §2).
TAU_RANGE_H = (30.0, 100.0)
DESIGN_LOAD_RANGE_W_PER_M2 = (25.0, 55.0)
HEATER_MARGIN_RANGE = (1.3, 2.0)


@pytest.fixture(scope="module")
def params():
    return HVACParams()


def _steady_free_float(params, hours, T_out_fixed, T_start=20.0):
    """Free-float (no heating) with a constant outdoor temperature."""
    env = BuildingHVAC()
    steps = int(hours * 3600 / params.delta_t)
    p = params.replace(
        max_steps_in_episode=steps + 1,
        T_out_amplitude=0.0,
        T_out_mean=T_out_fixed,
        T_out_noise_std=0.0,
        gain_occupied=0.0,
        gain_unoccupied=0.0,
        solar_peak=0.0,
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    state = state.replace(T_mass=T_start, weather_dev=0.0)
    _jstep = jax.jit(env.step_env)
    for _ in range(steps):
        _, state, _, terminated, _ = _jstep(key, state, jnp.array([-1.0]), p)
        if bool(terminated):
            break
    return state


# ---------------------------------------------------------------------------
# 1. Structural invariants (first principles)
# ---------------------------------------------------------------------------


def test_conductances_are_positive_and_ordered(params):
    """Film conductances dwarf envelope losses -- that is what 5R1C encodes."""
    c = zone_conductances(params)
    for name, value in c.items():
        assert value > 0.0, f"{name} = {value}"
    # Internal surface coupling is far stronger than heat loss to outdoors.
    assert c["H_tr_is"] > 10 * c["H_tr_op"]
    assert c["H_tr_ms"] > 10 * c["H_tr_op"]


def test_envelope_series_split_is_consistent(params):
    """1/H_tr_op = 1/H_tr_em + 1/H_tr_ms, the standard's series decomposition."""
    c = zone_conductances(params)
    reconstructed = 1.0 / (1.0 / c["H_tr_em"] + 1.0 / c["H_tr_ms"])
    assert reconstructed == pytest.approx(c["H_tr_op"], rel=1e-6)


def test_air_sits_between_outdoor_and_mass_when_heating_is_off(params):
    """With no heat input the air must lie between the two things driving it."""
    T_air, T_surface = solve_air_and_surface(
        T_mass=20.0, T_out=0.0, Q_heat=0.0, phi_st=0.0, phi_ia=0.0, params=params
    )
    assert 0.0 < T_air < 20.0
    assert 0.0 < T_surface < 20.0
    # The surface is closer to the mass than the air is (it couples to it directly).
    assert T_surface > T_air


def test_heating_raises_air_temperature(params):
    """Monotonicity: more heat, warmer air."""
    temps = [
        solve_air_and_surface(20.0, 0.0, q, 0.0, 0.0, params)[0]
        for q in (0.0, 2000.0, 5000.0)
    ]
    assert temps[0] < temps[1] < temps[2]


def test_colder_outdoors_lowers_air_temperature(params):
    temps = [
        solve_air_and_surface(20.0, t, 2000.0, 0.0, 0.0, params)[0]
        for t in (-10.0, 0.0, 10.0)
    ]
    assert temps[0] < temps[1] < temps[2]


def test_gain_split_follows_the_iso_convention(params):
    """The node shares sum to the total *minus* the glazing re-transmission.

    ISO 13790 withholds ``H_tr_w / (9.1 * A_tot)`` of the remainder -- the
    radiant fraction absorbed by the windows and passed straight back
    outdoors. Asserting plain conservation would be asserting a different
    standard than the one implemented.
    """
    phi_int, phi_sol = 1200.0, 3000.0
    phi_ia, phi_st, phi_m = split_gains(phi_int, phi_sol, params)
    c = zone_conductances(params)
    remainder = 0.5 * phi_int + phi_sol
    withheld = remainder * c["H_tr_w"] / (9.1 * c["A_tot"])
    assert phi_ia + phi_st + phi_m == pytest.approx(
        phi_int + phi_sol - withheld, rel=1e-6
    )
    assert all(x >= 0.0 for x in (phi_ia, phi_st, phi_m))
    # The withheld share is a small correction, not a leak of most of the gain.
    assert 0.0 < withheld < 0.05 * (phi_int + phi_sol)


def test_mass_node_is_driven_toward_its_surroundings(params):
    """The mass warms when the zone is hotter than it, and cools when colder.

    Compared at a fixed heat input: what matters is the *sign of the gradient*
    between the mass and everything around it, so both cases use the same
    (zero) heating and the same weather.
    """
    from target_gym.hvac.env import compute_velocity

    cold_outdoors = params.replace(T_out_mean=0.0, T_out_amplitude=0.0)

    def dT_mass(T_mass, p):
        v, _ = compute_velocity(
            jnp.array([T_mass, 0.0]),
            action=0.0,
            time=40,
            weather_dev=0.0,
            params=p,
        )
        return float(v[0])

    # A mass far below its surroundings warms; far above, it cools.
    assert dT_mass(-20.0, cold_outdoors) > 0.0
    assert dT_mass(30.0, cold_outdoors) < 0.0
    # And the drive is monotone in the gradient.
    assert dT_mass(-20.0, cold_outdoors) > dT_mass(0.0, cold_outdoors)


def test_emitter_lags_the_command(params):
    """The heating system cannot deliver a step change in output."""
    from target_gym.hvac.env import compute_velocity

    v, _ = compute_velocity(
        jnp.array([20.0, 0.0]),
        action=params.Q_heat_max,
        time=40,
        weather_dev=0.0,
        params=params,
    )
    expected = params.Q_heat_max / params.emitter_tau
    assert float(v[1]) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Schedules and disturbances
# ---------------------------------------------------------------------------


def test_hour_of_day_wraps_over_a_day(params):
    steps_per_day = int(86400 / params.delta_t)
    assert float(hour_of_day(0, params)) == pytest.approx(0.0)
    assert float(hour_of_day(steps_per_day // 2, params)) == pytest.approx(12.0)
    assert float(hour_of_day(steps_per_day, params)) == pytest.approx(0.0)


def test_occupancy_and_setpoint_follow_the_schedule(params):
    steps_per_hour = int(3600 / params.delta_t)
    night = 3 * steps_per_hour
    midday = 13 * steps_per_hour
    assert not bool(is_occupied(night, params))
    assert bool(is_occupied(midday, params))
    assert float(scheduled_setpoint(night, 21.0, params)) == pytest.approx(
        params.setpoint_setback
    )
    assert float(scheduled_setpoint(midday, 21.0, params)) == pytest.approx(21.0)


def test_internal_gain_is_higher_when_occupied(params):
    steps_per_hour = int(3600 / params.delta_t)
    assert float(internal_gain(13 * steps_per_hour, params)) > float(
        internal_gain(3 * steps_per_hour, params)
    )


def test_solar_gain_is_zero_at_night_and_peaks_midday(params):
    steps_per_hour = int(3600 / params.delta_t)
    assert float(solar_gain(2 * steps_per_hour, params)) == pytest.approx(0.0)
    assert float(solar_gain(23 * steps_per_hour, params)) == pytest.approx(0.0)
    midday = float(solar_gain(13 * steps_per_hour, params))
    assert midday > 0.0
    assert midday > float(solar_gain(8 * steps_per_hour, params))


def test_outdoor_temperature_is_coldest_before_dawn(params):
    steps_per_hour = int(3600 / params.delta_t)
    pre_dawn = float(outdoor_temperature(3 * steps_per_hour, 0.0, params))
    afternoon = float(outdoor_temperature(15 * steps_per_hour, 0.0, params))
    assert pre_dawn < afternoon
    assert afternoon == pytest.approx(
        params.T_out_mean + params.T_out_amplitude, abs=0.1
    )


# ---------------------------------------------------------------------------
# 3. Validation against published building data (PHYSICS.md §2)
# ---------------------------------------------------------------------------


def test_building_time_constant_matches_a_heavyweight_dwelling(params):
    """tau = C_m / H should be tens of hours. This is what makes the task slow."""
    c = zone_conductances(params)
    tau_h = c["C_m"] / total_heat_loss_coefficient(params) / 3600.0
    assert TAU_RANGE_H[0] < tau_h < TAU_RANGE_H[1], f"tau = {tau_h:.1f} h"


def test_design_heat_load_matches_a_well_insulated_building(params):
    """Load at -10 C outdoor / 20 C indoor should be 25-55 W/m^2."""
    H = total_heat_loss_coefficient(params)
    load_per_m2 = H * (20.0 - (-10.0)) / params.A_floor
    assert (
        DESIGN_LOAD_RANGE_W_PER_M2[0] < load_per_m2 < DESIGN_LOAD_RANGE_W_PER_M2[1]
    ), f"{load_per_m2:.1f} W/m2"


def test_heater_is_sized_against_the_design_load(params):
    """Plant sizing convention: 1.3-2x the design load."""
    H = total_heat_loss_coefficient(params)
    design_load = H * (20.0 - (-10.0))
    margin = params.Q_heat_max / design_load
    assert HEATER_MARGIN_RANGE[0] < margin < HEATER_MARGIN_RANGE[1], f"{margin:.2f}x"


def test_heating_capacity_holds_setpoint_on_a_design_day(params):
    """The heater must actually be able to hold 20 C at -10 C outdoors."""
    T_air, _ = solve_air_and_surface(
        T_mass=20.0,
        T_out=-10.0,
        Q_heat=params.Q_heat_max,
        phi_st=0.0,
        phi_ia=0.0,
        params=params,
    )
    assert T_air > 20.0


def test_free_float_drifts_slowly_overnight(params):
    """A heavyweight building loses only a few K over an unheated night."""
    state = _steady_free_float(params, hours=8.0, T_out_fixed=0.0, T_start=20.0)
    drop = 20.0 - float(state.T_mass)
    assert 1.0 < drop < 8.0, f"dropped {drop:.1f} K in 8 h"


def test_free_float_converges_toward_outdoor_temperature(params):
    """Left alone for many time constants, the zone must approach outdoors.

    Termination bounds are relaxed for this test: an unheated building really
    does drop below the 5 C freeze limit, and stopping there would measure the
    limit rather than the convergence.
    """
    relaxed = params.replace(T_air_min=-50.0, T_air_max=100.0)
    state = _steady_free_float(relaxed, hours=400.0, T_out_fixed=0.0, T_start=20.0)
    assert abs(float(state.T_mass) - 0.0) < 2.0


# ---------------------------------------------------------------------------
# 4. Task interface
# ---------------------------------------------------------------------------


def test_observation_hides_the_thermal_mass(params):
    """The mass governs the response and must not be observable.

    obs = [T_air, T_out, heat_pct, solar_norm, sin_h, cos_h, target_T].
    """
    env = BuildingHVAC()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    state = state.replace(T_mass=1234.5, T_surface=6789.0, weather_dev=4321.0)
    obs = env.get_obs(state, params)
    assert obs.shape == (7,)
    for hidden in (1234.5, 6789.0, 4321.0):
        assert not bool(
            jnp.any(jnp.isclose(obs, hidden))
        ), f"hidden state {hidden} leaked into the observation"


def test_action_maps_to_the_heating_range(params):
    env = BuildingHVAC()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    for raw, expected in ((-1.0, 0.0), (1.0, params.Q_heat_max)):
        new_state, _ = compute_next_state(raw, state, params, jax.random.PRNGKey(0))
        assert float(new_state.Q_command) == pytest.approx(expected, abs=1e-3)


def test_reward_peaks_on_setpoint_with_no_heating(params):
    env = BuildingHVAC()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    on = state.replace(T_air=21.0, target_T=21.0, Q_emitter=0.0)
    off = state.replace(T_air=25.0, target_T=21.0, Q_emitter=0.0)
    assert float(compute_reward(on, params)) == pytest.approx(1.0, abs=1e-5)
    assert float(compute_reward(on, params)) > float(compute_reward(off, params))


def test_reward_is_clipped_below_at_zero_comfort(params):
    """The comfort term must saturate, not turn back upward.

    Without the clip the quadratic rises again past 2*comfort_band, so a large
    error would score like a small one -- the bug that made the first MPC stop
    heating entirely.
    """
    env = BuildingHVAC()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    errors = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
    rewards = [
        float(
            compute_reward(
                state.replace(T_air=21.0 + e, target_T=21.0, Q_emitter=0.0), params
            )
        )
        for e in errors
    ]
    assert all(a >= b - 1e-9 for a, b in zip(rewards, rewards[1:])), rewards
    assert rewards[-1] == pytest.approx(0.0, abs=1e-9)


def test_energy_use_reduces_reward(params):
    env = BuildingHVAC()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    cold = state.replace(T_air=21.0, target_T=21.0, Q_emitter=0.0)
    hot = cold.replace(Q_emitter=params.Q_heat_max)
    assert float(compute_reward(cold, params)) > float(compute_reward(hot, params))


def test_terminates_when_the_zone_freezes_or_overheats(params):
    env = BuildingHVAC()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert bool(
        check_is_terminal(state.replace(T_air=params.T_air_min - 1.0), params)[0]
    )
    assert bool(
        check_is_terminal(state.replace(T_air=params.T_air_max + 1.0), params)[0]
    )
    assert not bool(check_is_terminal(state.replace(T_air=21.0), params)[0])


def test_reset_starts_thermally_settled(params):
    """Mass and air must start consistent, not several K apart.

    A building whose mass disagreed with its air at t=0 would spend the first
    day of every episode relaxing -- an artefact, not a control problem.
    """
    env = BuildingHVAC()
    for seed in range(5):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        assert isinstance(state, HVACState)
        assert abs(float(state.T_air) - float(state.T_mass)) < 6.0
        assert (
            params.initial_T_range[0]
            <= float(state.T_mass)
            <= params.initial_T_range[1]
        )
