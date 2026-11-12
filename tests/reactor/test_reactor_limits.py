"""The reactor's limits, trips and instruments under Amendment 3 (owner, 2026-11-08).

Contracts, from the family's sources, written before the code:

- Offset band as a penalty clock (EDF scope, Westinghouse allowance): time with the
  axial offset outside +-0.05 of its reference counts at every power during load
  follow. 1 h in 24 h is allowed (D.C. Cook and RAOC technical specifications)
  and costs nothing. Past it the plant must derate to 50 % (the same
  specifications), priced as that derate's lost output: the power above 50 % at
  the spot price, $80/MWh on 1 GWe. At 50 % or below there is nothing to give up.
- Rod insertion limit (family shape, KAIST's lead): a straight line in power, 22 %
  of the stroke in at full power, 78 % at half. Time deeper than the limit counts,
  resets once the rods are back above it, and 2 h of it (STS 3.1.6) ends the run
  in hot standby.
- Trips are lost production at the spot price on 1 GWe: a scram's restart takes
  48 h (ours, family 24 to 70 h: Doel-3's 6 h shutdown then 18 h at reduced power,
  INIS 1ypr9-dk595; APR1400 restart windows of 22 to 70 h, KNS 20A-065); hot
  standby after a timed limit's overrun, 15 h (ours: Doel-3's 6 h down and 18 h
  reduced, about 15 h of full output lost).
- Tavg is charged past its 1.6 K dead band on either side of its program, at the
  same price per kelvin (the physics review's finding 1, owner 2026-11-10,
  reversing Amendment 3 decision 7): both cited procedures keep a two-sided band
  (IAEA NP-T-3.23 annex II-5.1; NRC HRTD 8.1), and at part power the hot side is
  bounded by the steam side, not by DNB.
- The operator reads the recent power history the starting xenon is balanced with,
  and the core boron only through a boronmeter 20 min late (Areva, US 8,489,338).

Hand prices: derate at full power, 0.5 GW x 1000 x $80/MWh = $40,000/h, $55.556 per
5 s sub-step. Scram 48 h x $80,000/h = $3,840,000; standby 15 h = $1,200,000.
"""

import jax
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    check_is_terminal,
    compute_next_state,
    compute_reward_terms,
    rod_insertion_limit,
    rod_worth,
    tavg_dnb_limit,
    tavg_program,
)


def reset(seed=0, **changes):
    env, p = Reactor(), ReactorParams(**changes)
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    return env, p, obs, s


def at(s, **values):
    return s.replace(**{k: s.n * 0 + v for k, v in values.items()})


def step(env, p, s):
    hold = np.array([1.0 - 2.0 * float(s.rod_in), 0.0])
    s1, _ = compute_next_state(hold, s, p, integration_method=env.integration_method)
    return s1


# --- the offset's penalty clock ---


@pytest.mark.parametrize("n", [1.0, 0.5])
def test_time_outside_the_offset_band_counts_at_every_power(n):
    env, p, obs, s = reset()
    s = at(s, n=n, ao_outside=100.0)
    s1 = step(env, p, s.replace(Xe_tilt=s.Xe_tilt * 0 - 0.2))
    assert abs(float(s1.delta_I) / float(s1.n)) > p.ao_band
    assert float(s1.ao_outside) == pytest.approx(100.0 + p.delta_t)


def test_time_inside_the_band_does_not_count():
    env, p, obs, s = reset()
    s1 = step(env, p, at(s, ao_outside=100.0))
    assert float(s1.ao_outside) == pytest.approx(100.0)


@pytest.mark.parametrize(
    "n, used, expected",
    [
        (1.0, 3599.0, 0.0),  # inside the allowed hour: free
        (1.0, 3600.0, 55.556),  # past it: derate from 100 to 50 %
        (0.75, 4000.0, 27.778),  # from 75 %
        (0.5, 4000.0, 0.0),  # already at 50 %
    ],
)
def test_an_overrun_costs_the_derates_lost_output(n, used, expected):
    env, p, obs, s = reset()
    s = at(s, n=n, delta_I=0.2 * n, ao_outside=used)
    s = s.replace(T_ref=0.5 * (s.T_coolant + s.T_cold))  # Tavg on program
    terms = compute_reward_terms(s, p)
    assert float(terms["tracking"]) == pytest.approx(expected, rel=1e-3, abs=1e-6)


# --- the rod insertion limit ---


@pytest.mark.parametrize("n, limit", [(1.0, 0.22), (0.75, 0.50), (0.5, 0.78)])
def test_the_insertion_limit_is_a_straight_line_in_power(n, limit):
    assert float(rod_insertion_limit(n, ReactorParams())) == pytest.approx(limit)


def test_time_past_the_insertion_limit_counts_and_resets():
    env, p, obs, s = reset()
    # 30 % in at full power, past 22 %, with boron trimmed so the core stays critical.
    extra = float(rod_worth(0.3, p) - rod_worth(float(s.rod_in), p))
    deep = at(
        s, rod_in=0.3, ril_outside=100.0, boron=float(s.boron) - extra / p.boron_worth
    )
    assert float(step(env, p, deep).ril_outside) == pytest.approx(100.0 + p.delta_t)
    shallow = at(s, ril_outside=100.0)
    assert float(step(env, p, shallow).ril_outside) == 0.0


@pytest.mark.parametrize("seconds, ends", [(7195.0, False), (7200.0, True)])
def test_two_hours_past_the_insertion_limit_ends_the_run(seconds, ends):
    env, p, obs, s = reset()
    terminated, _ = check_is_terminal(at(s, ril_outside=seconds), p)
    assert bool(terminated) is ends


# --- trips as lost production ---


def test_a_scram_costs_two_days_of_output():
    env, p, obs, s = reset()
    terms = compute_reward_terms(at(s, n=1.5), p)  # overpower
    assert float(terms["failure"]) == pytest.approx(3.84e6, rel=1e-4)


@pytest.mark.parametrize(
    "field, value", [("below_floor", 900.0), ("ril_outside", 7200.0)]
)
def test_a_timed_limits_overrun_costs_hot_standby(field, value):
    env, p, obs, s = reset()
    terms = compute_reward_terms(at(s, **{field: value}), p)
    assert float(terms["failure"]) == pytest.approx(4.8e5, rel=1e-4)


# --- Tavg above program ---


# test_tavg_above_its_program_costs_nothing was retired with decision 7, which the
# owner reversed on 2026-11-10 (the physics review's finding 1).


@pytest.mark.parametrize("hot, expected", [(1.59, 0.0), (3.0, 0.44722)])
def test_tavg_above_its_program_is_charged_past_the_band(hot, expected):
    """Lined out at half load, Tavg ``hot`` K over its program. Past the 1.6 K band,
    the cold side's price: 1.25 x 0.23 %/K x $80/MWh x 1 GW = $0.0639 per K s, so
    (3 - 1.6) K x 5 s = $0.44722 per sub-step ($2.68 per 30 s step)."""
    env, p, obs, s = reset()
    tavg = 0.5 * (s.T_coolant + s.T_cold)
    s = at(s, n=0.5).replace(T_ref=tavg - hot)
    terms = compute_reward_terms(s, p)
    assert float(terms["tracking"]) == pytest.approx(expected, rel=1e-3, abs=1e-6)


# --- what the operator reads ---


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_observation_carries_the_power_the_xenon_is_balanced_with(seed):
    env, p, obs, s = reset(seed)
    assert obs.shape == (13,)
    lo, hi = p.initial_xenon_power_range
    assert lo <= float(obs[10]) <= hi
    assert float(obs[10]) == pytest.approx(float(s.xenon_power))


def test_the_boron_reading_is_twenty_minutes_late():
    env, p, obs, s = reset(demand_sigma=0.0)
    start = float(obs[7])
    rods = 1.0 - 2.0 * float(s.rod_in)
    readings, core = [], []
    for k in range(70):
        dilute = np.array([rods, -1.0 if k < 4 else 0.0], np.float32)  # 2 min
        obs, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, dilute, p)
        readings.append(float(obs[7]))
        core.append(float(s.boron))
    core, readings = np.array(core), np.array(readings)
    lag = 40  # 20 min of 30 s steps
    assert core[lag - 1] < start - 0.5  # the core has moved by then
    np.testing.assert_allclose(readings[: lag - 1], start, atol=1e-3)
    np.testing.assert_allclose(readings[lag:], core[: len(core) - lag], rtol=1e-6)


# --- Tavg's upper limit (DNB), the hot side ---
# Westinghouse STS LCO 3.4.1 (as summarised for the US EPR GTS and the AP1000 bases):
# RCS average temperature at or below the COLR's limit, restored within 2 h, else
# MODE 2 within 6 h. The limit is an absolute value: Braidwood's COLR sets nominal
# Tavg at rated power at 588.0 F and the limit at 593.1 F, 5.1 F = 2.833 K above
# (from a search summary; nrc.gov blocks automated reads). So the limit is the
# full-power program plus 2.833 K at every power, and an overrun goes to hot standby.

HOT_MARGIN = 5.1 / 1.8


def test_time_above_the_dnb_limit_counts_and_resets():
    env, p, obs, s = reset()
    limit = float(tavg_program(1.0, p)) + HOT_MARGIN
    tavg = float(0.5 * (s.T_coolant + s.T_cold))
    hot = s.replace(
        T_coolant=s.T_coolant + 2 * (limit + 0.5 - tavg),
        T_cold=s.T_cold + 2 * (limit + 0.5 - tavg),
    )
    hot = at(hot, above_dnb=100.0)
    assert float(step(env, p, hot).above_dnb) == pytest.approx(100.0 + p.delta_t)
    assert float(step(env, p, at(s, above_dnb=100.0)).above_dnb) == 0.0


def test_the_dnb_limit_is_the_full_power_programs_at_every_power():
    p = ReactorParams()
    assert float(tavg_dnb_limit(p)) == pytest.approx(
        float(tavg_program(1.0, p)) + HOT_MARGIN, abs=1e-3
    )


@pytest.mark.parametrize("seconds, ends", [(7195.0, False), (7200.0, True)])
def test_two_hours_above_the_dnb_limit_costs_hot_standby(seconds, ends):
    env, p, obs, s = reset()
    s = at(s, above_dnb=seconds)
    terminated, _ = check_is_terminal(s, p)
    assert bool(terminated) is ends
    if ends:
        assert float(compute_reward_terms(s, p)["failure"]) == pytest.approx(
            4.8e5, rel=1e-4
        )


# --- a plant at rest stays at rest ---
# Not a benchmark scenario: a check that the plant, lined out at full load with nothing
# moving (no load wobble, xenon in balance, no burnup, no debt carried in, no sensor
# noise), holds itself for a whole day with the rods parked and no boron. Nothing
# drives it off its program or its offset, so doing nothing costs nothing.


def test_a_plant_at_rest_needs_nothing_for_a_day():
    env, p, obs, s = reset(
        demand_sigma=0.0,
        initial_xenon_power_range=(1.0, 1.0),
        ramp_down_range=(1e6, 1e6),
        burnup_range=(0.0, 0.0),
        carried_days_range=(0.0, 0.0),
        sensor_noise_scale=0.0,
    )
    null = jax.numpy.array([1.0 - 2.0 * p.rod_park, 0.0])
    keys = jax.random.split(jax.random.PRNGKey(1), p.max_steps_in_episode)

    def step(s, k):
        _, s1, r, _, info = env.step_env(k, s, null, p)
        tavg = 0.5 * (s1.T_coolant + s1.T_cold)
        return s1, (-r, info["tripped"], tavg - s1.T_ref, s1.delta_I / s1.n)

    _, (cost, tripped, error, offset) = jax.lax.scan(step, s, keys)
    assert not bool(np.any(tripped))
    assert float(np.max(np.abs(error))) < p.e_tol  # inside the dead band all day
    assert float(np.max(np.abs(offset - p.ao_ref))) < p.ao_band
    assert float(np.sum(cost)) < 1.0  # dollars over the day
