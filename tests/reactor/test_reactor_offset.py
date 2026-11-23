"""The axial offset's limits as the standard specification states them (owner,
2026-11-15).

Contract, from NUREG-1431 Rev. 3.0, Bases B 3.2.3A and its Figure B 3.2.3A-1
(REALITY.md section 4), written before the code:

- The flux difference is delta I, a percentage of rated power. Its target scales with
  power, and the target band is 5 % of rated delta I on either side of the target at
  every power: the figure's two dashed lines run parallel, 10 % apart.
- Outside the band, penalty deviation time accumulates: a minute a minute at 50 % of
  rated power and above, half a minute a minute from 15 % to 50 %, none below 15 %.
- One hour of it in the last 24 h is allowed. Past it the unit must be under 50 % of
  rated power (Required Action C.1), priced as that reduction's lost output.
- At 90 % and above the flux difference must be inside the band: 15 min to restore
  it, then 15 min to be under 90 % (Required Actions A.1 and B.1). So after 30 min
  outside without a break the unit must be under 90 %, priced as that reduction's
  lost output.
- From 50 % to 90 % the flux difference must also stay inside the acceptable
  operation limits, straight lines from 11 % of rated delta I at 90 % power to 31 %
  at 50 % (the figure). Outside them for 30 min without a break, the unit must be
  under 50 % (C.1's completion time).
- Below 50 % there is no outer limit.
- Where two reductions are due, the unit makes the deeper one: the prices do not add.
- The operator reads the accumulated penalty time: the process computer keeps it "to
  assess the status of the penalty deviation time". Observation 15, in hours.

Hand prices at $80/MWh on 1 GWe: $111.11 per 5 s sub-step per unit of rated power,
so $55.556 from full power to 50 % and $11.111 from full power to 90 %.
"""

import jax
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    compute_next_state,
    compute_reward_terms,
    get_obs,
    offset_limit,
    penalty_rate,
)

PENALTY = 15  # observation channel
QUIET = dict(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    cycle_range=(0.0, 0.0),
    initial_tilt_share=0.0,
)


def reset(seed=0, **changes):
    env, p = Reactor(), ReactorParams(**{**QUIET, **changes})
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    return env, p, obs, s


def lined_out_at(load):
    """A day planned at ``load`` all through, the plant lined out there with xenon
    in balance."""
    return reset(
        load_min=load,
        ramp_down_range=(-1e5, -1e5),
        hold_range=(1e6, 1e6),
        initial_xenon_power_range=(load, load),
    )


def at(s, **values):
    return s.replace(**{k: s.n * 0 + v for k, v in values.items()})


def step(env, p, s):
    hold = np.array([1.0 - 2.0 * float(s.rod_in), 0.0, 1.0])
    s1, _ = compute_next_state(hold, s, p, integration_method=env.integration_method)
    return s1


# --- the rule, by its numbers ---


@pytest.mark.parametrize(
    "n, delta_I, outside",
    [
        (1.0, 0.049, False),
        (1.0, 0.051, True),
        (1.0, -0.051, True),
        (0.5, 0.049, False),  # 9.8 % of offset at half power: inside the band
        (0.5, 0.051, True),
        (0.2, 0.049, False),
        (0.2, -0.051, True),
    ],
)
def test_the_band_is_five_percent_of_rated_delta_i_at_every_power(n, delta_I, outside):
    assert (float(penalty_rate(delta_I, n, ReactorParams())) > 0.0) is outside


@pytest.mark.parametrize(
    "n, rate", [(1.0, 1.0), (0.5, 1.0), (0.49, 0.5), (0.15, 0.5), (0.149, 0.0)]
)
def test_penalty_time_accumulates_by_the_power(n, rate):
    assert float(penalty_rate(0.2, n, ReactorParams())) == pytest.approx(rate)


@pytest.mark.parametrize(
    "n, limit",
    [
        (0.95, 0.05),
        (0.9, 0.05),  # at 90 % and above, the band
        (0.89, 0.115),  # just under it, on the line through (11, 90)
        (0.8, 0.16),
        (0.7, 0.21),
        (0.5, 0.31),
        (0.49, np.inf),
    ],
)
def test_the_acceptable_operation_limits_are_the_figures(n, limit):
    """At 90 % and above only the band is acceptable; from (11, 90) the limit widens
    by 5 % of delta I per 10 % of power to (31, 50); below 50 % there is none."""
    assert float(offset_limit(n, ReactorParams())) == pytest.approx(limit)


# --- the plant keeps the three clocks ---


@pytest.mark.parametrize("load, counted", [(1.0, 5.0), (0.6, 5.0), (0.3, 2.5)])
def test_time_outside_the_band_counts_by_the_power(load, counted):
    """A strong xenon tilt puts the flux difference outside the band; a 5 s sub-step
    counts 5 s at full power and at 60 %, 2.5 s at 30 %."""
    env, p, obs, s = lined_out_at(load)
    s1 = step(env, p, at(s, ao_outside=100.0, Xe_tilt=-0.2))
    assert abs(float(s1.delta_I)) > p.ao_band
    assert float(s1.ao_outside) == pytest.approx(100.0 + counted)


def test_time_inside_the_band_does_not_count():
    env, p, obs, s = lined_out_at(1.0)
    s1 = step(env, p, at(s, ao_outside=100.0, ao_strict=100.0, ao_far=100.0))
    assert abs(float(s1.delta_I)) < p.ao_band
    assert float(s1.ao_outside) == pytest.approx(100.0)
    assert float(s1.ao_strict) == 0.0
    assert float(s1.ao_far) == 0.0


def test_at_full_power_time_outside_the_band_runs_without_a_break():
    env, p, obs, s = lined_out_at(1.0)
    s1 = step(env, p, at(s, ao_strict=100.0, Xe_tilt=-0.2))
    assert abs(float(s1.delta_I)) > p.ao_band
    assert float(s1.ao_strict) == pytest.approx(100.0 + p.delta_t)
    assert float(s1.ao_far) == 0.0  # the outer limits are the 50 to 90 % rule


@pytest.mark.parametrize("tilt, beyond", [(-0.02, False), (-0.2, True)])
def test_at_seventy_percent_time_outside_the_outer_limits_runs_without_a_break(
    tilt, beyond
):
    """At 70 % the outer limit is 21 % of rated delta I. A mild tilt leaves the band
    but not the limit, a strong one leaves both."""
    env, p, obs, s = lined_out_at(0.7)
    s1 = step(env, p, at(s, ao_far=100.0, ao_strict=100.0, Xe_tilt=tilt))
    off = abs(float(s1.delta_I))
    assert off > p.ao_band
    assert (off > 0.21) is beyond
    assert float(s1.ao_far) == pytest.approx(100.0 + p.delta_t if beyond else 0.0)
    assert float(s1.ao_strict) == 0.0  # under 90 %


# --- what an overrun costs ---


@pytest.mark.parametrize(
    "n, clocks, expected",
    [
        (1.0, dict(ao_outside=3599.0), 0.0),  # inside the allowed hour
        (1.0, dict(ao_outside=3600.0), 55.556),  # to 50 %
        (0.75, dict(ao_outside=4000.0), 27.778),
        (0.5, dict(ao_outside=4000.0), 0.0),  # already at 50 %
        (1.0, dict(ao_strict=1799.0), 0.0),  # 15 min to restore, 15 to reduce
        (1.0, dict(ao_strict=1800.0), 11.111),  # to 90 %
        (0.95, dict(ao_strict=1800.0), 5.556),
        (0.7, dict(ao_far=1799.0), 0.0),
        (0.7, dict(ao_far=1800.0), 22.222),  # to 50 %
        (1.0, dict(ao_outside=3600.0, ao_strict=1800.0), 55.556),  # the deeper one
    ],
)
def test_an_overrun_costs_the_lost_output_of_the_reduction_it_asks_for(
    n, clocks, expected
):
    env, p, obs, s = reset()
    s = at(s, n=n, **clocks)
    s = s.replace(T_ref=0.5 * (s.T_coolant + s.T_cold))  # Tavg on program
    terms = compute_reward_terms(s, p)
    assert float(terms["tracking"]) == pytest.approx(expected, rel=1e-3, abs=1e-6)


def test_the_operator_reads_the_penalty_time_in_hours():
    env, p, obs, s = reset()
    assert float(obs[PENALTY]) == 0.0
    assert float(get_obs(at(s, ao_outside=1800.0), p)[PENALTY]) == pytest.approx(0.5)
