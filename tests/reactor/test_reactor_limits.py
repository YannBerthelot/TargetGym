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
- Tavg above its program is not lost output (rule 15): it costs nothing; below,
  past the dead band, the charge stands.
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
    assert float(terms["failure"]) == pytest.approx(1.2e6, rel=1e-4)


# --- Tavg above program ---


def test_tavg_above_its_program_costs_nothing():
    env, p, obs, s = reset()
    hot = s.replace(T_coolant=s.T_coolant + 5.0, T_cold=s.T_cold + 5.0)
    assert float(compute_reward_terms(hot, p)["tracking"]) == 0.0


# --- what the operator reads ---


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_observation_carries_the_power_the_xenon_is_balanced_with(seed):
    env, p, obs, s = reset(seed)
    assert obs.shape == (11,)
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
