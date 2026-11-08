"""Behaviour tests for the slab-heated building against real practice
(charter step 3).

Each test is one of the eleven proposed in ``COLD_REVIEW.md``. The source, the
quantity and the tolerance are the audit's, fixed before any run; sources are
REALITY.md's numbered list. "Quiet weather" is a constant outdoor temperature:
no daily swing, no OU deviation and, unless the test says otherwise, no sun and
unoccupied gains all day.

The heating curve recipe is the task's expert (``hvac/experts.py``): Rehau's
construction from the design point [12, p. 7] on the 24 h outdoor mean [9, 13].
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.hvac.env import (
    HVACParams,
    floor_surface_temperature,
    slab_conductances,
    steady_state,
)
from target_gym.hvac.env_jax import BuildingHVAC
from target_gym.hvac.experts import curve_supply, make_hvac_curve

pytestmark = pytest.mark.medium

ENV = BuildingHVAC()
DT_MIN = 15.0  # minutes per step
DAY = 96  # steps


def _quiet(T_out, sun=False, gains="unoccupied", **kw):
    """Params with constant outdoor ``T_out`` and fixed gains."""
    base = HVACParams()
    gain = base.gain_unoccupied if gains == "unoccupied" else base.gain_occupied
    fields = dict(
        sensor_noise_scale=0.0,
        T_out_mean=float(T_out),
        T_out_amplitude=0.0,
        T_out_noise_std=0.0,
        solar_peak=base.solar_peak if sun else 0.0,
        gain_occupied=gain,
        gain_unoccupied=gain,
    )
    if gains == "schedule":
        fields.update(
            gain_occupied=base.gain_occupied, gain_unoccupied=base.gain_unoccupied
        )
    fields.update(kw)
    return base.replace(**fields)


def _settled(params, T_air):
    """A reset state moved to the steady state holding ``T_air``."""
    _, state = ENV.reset_env(jax.random.PRNGKey(0), params)
    T_mass, T_core, T_s, T_sup = steady_state(params, params.T_out_mean, T_air)
    state = state.replace(
        T_mass=T_mass,
        T_core=T_core,
        T_surface=T_s,
        T_air=jnp.asarray(T_air, jnp.float32),
        T_out=jnp.asarray(params.T_out_mean, jnp.float32),
        T_out_hist=jnp.full_like(state.T_out_hist, params.T_out_mean),
        weather_dev=jnp.zeros(()),
        T_supply=T_sup,
    )
    return state, float(T_sup)


def _raw(T_sup, params):
    span = params.T_supply_max - params.T_supply_min
    return jnp.atleast_1d(2.0 * (T_sup - params.T_supply_min) / span - 1.0)


def _run(params, state, policy, steps, params_at=None):
    """Roll ``steps`` steps; ``policy(obs, k) -> supply C``. Returns the
    states, stacked."""
    obs = ENV.get_obs(state, params)

    def body(carry, k):
        obs, state = carry
        p = params if params_at is None else params_at(k)
        action = _raw(policy(obs, k), p)
        obs, state, _, _, _ = ENV.step_env(jax.random.PRNGKey(0), state, action, p)
        return (obs, state), state

    _, states = jax.lax.scan(body, (obs, state), jnp.arange(steps))
    return states


def _curve_policy(curve):
    return lambda obs, k: curve_supply(curve, obs[2])


def _room_flux(states, params):
    s = slab_conductances(params)
    return np.asarray(s["H_slab"] * (states.T_core - states.T_air))


# 1. [10, p. 7, Fig. 3]: concrete core TABS (type E), heating, 145 to 462 min to
#    63.2 % of the step response; [6, p. 2] 4 to 8 h. Heat flux from the slab
#    into the room after a step in supply temperature, from a steady state at
#    0 C outdoor, no sun or gains. Tolerance: 145 to 462 min, no margin.
def test_slab_heating_time_constant_is_a_concrete_core_tabs():
    params = _quiet(0.0, gain_occupied=0.0, gain_unoccupied=0.0)
    state, T_sup = _settled(params, 21.0)
    before = float(_room_flux(state, params))
    states = _run(params, state, lambda obs, k: T_sup + 5.0, 10 * DAY)
    flux = _room_flux(states, params)
    final = flux[-1]
    k = int(np.argmax(flux - before >= 0.632 * (final - before)))
    minutes = (k + 1) * DT_MIN
    assert 145.0 <= minutes <= 462.0, minutes


# 2. [8, p. 15]: in TABS and embedded systems it can take one hour to several
#    hours before a setpoint change is felt. Time from full heat to a 0.5 K rise
#    in room air, settled at 20 C, outdoor 0 C, quiet weather. At least 60 min.
def test_a_change_is_felt_only_after_an_hour_or_more():
    params = _quiet(0.0)
    state, _ = _settled(params, 20.0)
    states = _run(params, state, lambda obs, k: params.T_supply_max, DAY)
    T_air = np.asarray(states.T_air)
    k = int(np.argmax(T_air >= 20.5))
    assert T_air[k] >= 20.5, "never felt within a day"
    assert (k + 1) * DT_MIN >= 60.0, (k + 1) * DT_MIN


# 3. [11, p. 2]: lowering an underfloor curve 2 C lowers the room 2 C. Two runs
#    of the curve recipe offset 2 K, three days each, last 24 h mean compared.
#    Ratio 0.75 to 1.25.
def test_curve_shift_moves_the_room_about_one_for_one():
    params = HVACParams(sensor_noise_scale=0.0)
    _, state = ENV.reset_env(jax.random.PRNGKey(7), params)
    means = []
    for offset in (0.0, -2.0):
        curve = make_hvac_curve(offset=offset)
        states = _run(params, state, _curve_policy(curve), 3 * DAY)
        means.append(float(np.mean(np.asarray(states.T_air)[-DAY:])))
    ratio = (means[0] - means[1]) / 2.0
    assert 0.75 <= ratio <= 1.25, (means, ratio)


# 4. [7, p. 7], [9, p. 7]: self-regulation. Fixed supply, then a step of 8 W/m2
#    internal gain: the slab's heat output is strictly lower one hour later.
def test_slab_output_falls_when_gains_rise_at_fixed_supply():
    params = _quiet(0.0)
    state, T_sup = _settled(params, 21.0)
    gained = params.replace(
        gain_occupied=params.gain_occupied + 8.0,
        gain_unoccupied=params.gain_unoccupied + 8.0,
    )
    before = float(state.Q_slab)
    states = _run(gained, state, lambda obs, k: T_sup, 4)
    assert float(states.Q_slab[-1]) < before, (before, float(states.Q_slab[-1]))


# 5. [15]: setback recovery takes 1.0 h (old, uninsulated) to 6.9 h (super
#    insulated). 8 h setback from 21 C at -10 C (water off), then full heat:
#    time to within 0.5 K of 21 C is at least 1.0 h.
def test_setback_recovery_takes_hours():
    params = _quiet(-10.0)
    state, _ = _settled(params, 21.0)
    off = _run(params, state, lambda obs, k: params.T_supply_min, 32)
    state = jax.tree_util.tree_map(lambda x: x[-1], off)
    states = _run(params, state, lambda obs, k: params.T_supply_max, 2 * DAY)
    T_air = np.asarray(states.T_air)
    k = int(np.argmax(T_air >= 20.5))
    assert (k + 1) * DT_MIN >= 60.0, (float(off.T_air[-1]), (k + 1) * DT_MIN)


# 6. [11, p. 3]: underfloor heating built after 2010, 30 to 35 C supply at -8 C
#    outdoor. Steady supply that holds 20 C, no sun, unoccupied gains.
#    Tolerance: 30 to 35 C, +-1 K for the steady-state criterion.
def test_steady_supply_at_minus_eight_matches_the_guidance():
    params = _quiet(-8.0)
    *_, T_sup = steady_state(params, -8.0, 20.0)
    assert 29.0 <= float(T_sup) <= 36.0, float(T_sup)


# 7. [11, p. 3], [9, p. 5]: heating limit 8 to 15 C outdoor (13.1 C in [9]).
#    Water off, constant outdoor mean with the daily sun and occupancy, no OU,
#    five days per point; bisect for a 20 C daily mean. Tolerance 8 to 15 C.
def test_heating_limit_is_where_free_gains_hold_twenty():
    def daily_mean(T_out):
        params = _quiet(T_out, sun=True, gains="schedule")
        state, _ = _settled(params, 20.0)
        states = _run(params, state, lambda obs, k: params.T_supply_min, 5 * DAY)
        return float(np.mean(np.asarray(states.T_air)[-DAY:]))

    lo, hi = 0.0, 20.0
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if daily_mean(mid) < 20.0 else (lo, mid)
    assert 8.0 <= 0.5 * (lo + hi) <= 15.0, 0.5 * (lo + hi)


# 8. [5, p. 23]: 29 C floor surface, occupied area; [17]: 55 C supply cut-off.
#    Design outdoor -10 C, the curve holding 21 C: surface at most 29 C and
#    supply at most 55 C, no margin.
def test_floor_surface_and_supply_stay_within_limits_at_design():
    params = _quiet(-10.0)
    _, T_core, _, T_sup = steady_state(params, -10.0, 21.0)
    surface = float(floor_surface_temperature(T_core, 21.0, params))
    assert surface <= 29.0, surface
    assert float(T_sup) <= 55.0, float(T_sup)


# 9. [12, p. 6]: a heavy system overshoots when it changes from reduced to
#    normal mode. The curve with 50 % room influence ([16, p. 80] range), a
#    setback 22:00 to 07:00: peak air above the setpoint in the 4 h after the
#    setback ends is greater than 0.
def test_morning_overshoot_after_reduced_mode():
    params = HVACParams(sensor_noise_scale=0.0)
    _, state = ENV.reset_env(jax.random.PRNGKey(3), params)
    normal = make_hvac_curve()
    reduced = make_hvac_curve(T_set=17.0)

    def policy(obs, k):
        hour = (k % DAY) * DT_MIN / 60.0
        night = (hour >= 22.0) | (hour < 7.0)
        curve = jax.tree_util.tree_map(
            lambda a, b: jnp.where(night, a, b), reduced, normal
        )
        # 50 % room influence, ours as half a kelvin of supply per kelvin
        # of room error: [16] gives the share, not the law.
        return curve_supply(curve, obs[2]) + 0.5 * (curve.T_set - obs[0])

    states = _run(params, state, policy, 3 * DAY)
    T_air = np.asarray(states.T_air)
    morning = slice(2 * DAY + 28, 2 * DAY + 44)  # 07:00 to 11:00 on day 3
    assert float(np.max(T_air[morning] - normal.T_set)) > 0.0


# 10. [15]: a heater sized to the load "may never recover from setback" at
#     design. -10 C, 8 h with the water off, then the steady design supply:
#     still more than 0.5 K below 21 C 15 h later.
def test_design_supply_does_not_recover_a_setback():
    params = _quiet(-10.0)
    state, T_sup = _settled(params, 21.0)
    off = _run(params, state, lambda obs, k: params.T_supply_min, 32)
    state = jax.tree_util.tree_map(lambda x: x[-1], off)
    states = _run(params, state, lambda obs, k: T_sup, 60)
    assert float(states.T_air[-1]) < 20.5, float(states.T_air[-1])


# 11. [4]: ASHRAE 55 drift, 1.1 K in 15 min and 2.2 K in 1 h, without occupant
#     control. The curve recipe on the held-out weather seeds: the largest
#     occupied air change over any 15 min and any 1 h window.
@pytest.mark.parametrize("seed", range(200, 205))
def test_drift_under_the_curve_stays_within_ashrae_55(seed):
    params = HVACParams(sensor_noise_scale=0.0)
    _, state = ENV.reset_env(jax.random.PRNGKey(seed), params)
    states = _run(params, state, _curve_policy(make_hvac_curve()), 7 * DAY)
    T_air = np.asarray(states.T_air)
    hour = (np.arange(1, len(T_air) + 1) % DAY) * DT_MIN / 60.0
    occupied = (hour >= params.occupied_start_h) & (hour < params.occupied_end_h)
    d15 = np.abs(np.diff(T_air))[occupied[1:]]
    d60 = np.abs(T_air[4:] - T_air[:-4])[occupied[4:]]
    assert d15.max() <= 1.1, d15.max()
    assert d60.max() <= 2.2, d60.max()
