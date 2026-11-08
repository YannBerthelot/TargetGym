"""The reactor as EDF runs load follow (docs/proposals/reactor-plant-check.md).

Contract, from the sources the plant check cites:

- the observation carries the day's plan, the hours until each ramp starts and 0
  once it has, as the dispatch an operator gets in advance (OAPS, section 1);
- Tavg costs nothing within 1.6 K of its program, the black rods' dead band (IAEA
  NP-T-3.23, annex II-5.1), and the linear price past it;
- the axial-offset band (0.05 around the reference, IAEA NP-T-3.23, annex II-4.3)
  is a penalty clock since Amendment 3, tested in test_reactor_limits.py;
- Tavg under the criticality floor is a limit with a time to act: the run ends only
  once it has lasted ``floor_allowed``.

Prices by hand: Tavg, 1.25 x 0.0023 x $80/MWh x 1000 MW = $230/h per K, $0.3194 per K
over a 5 s sub-step.
"""

import jax
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    check_is_terminal,
    compute_next_state,
    compute_reward_terms,
)


def reset(seed=0):
    env, p = Reactor(), ReactorParams()
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    return env, p, obs, s


def test_the_observation_carries_the_hours_to_each_ramp():
    env, p, obs, s = reset()
    assert obs.shape == (11,)
    np.testing.assert_allclose(obs[8], float(s.ramp_down_at) / 3600.0, rtol=1e-6)
    np.testing.assert_allclose(obs[9], float(s.ramp_up_at) / 3600.0, rtol=1e-6)


def test_a_ramp_that_has_started_reads_zero():
    env, p, obs, s = reset()
    later = s.replace(physics_time=int(float(s.ramp_down_at) / p.delta_t) + 12)
    obs = env.get_obs(later, p)
    assert float(obs[8]) == 0.0
    assert float(obs[9]) > 0.0


def tavg_off_by(s, kelvin):
    return s.replace(T_coolant=s.T_coolant - kelvin, T_cold=s.T_cold - kelvin)


@pytest.mark.parametrize("off, expected", [(1.0, 0.0), (1.59, 0.0), (2.6, 0.31944)])
def test_tavg_costs_only_past_the_rods_dead_band(off, expected):
    env, p, obs, s = reset()
    terms = compute_reward_terms(tavg_off_by(s, off), p)
    assert float(terms["tracking"]) == pytest.approx(expected, rel=1e-3, abs=1e-6)


@pytest.mark.parametrize("seconds, ends", [(0.0, False), (895.0, False), (900.0, True)])
def test_the_criticality_floor_ends_the_run_only_after_its_allowed_time(seconds, ends):
    env, p, obs, s = reset()
    s = tavg_off_by(s, 30.0).replace(below_floor=s.n * 0 + seconds)
    terminated, _ = check_is_terminal(s, p)
    assert bool(terminated) is ends


def test_the_time_under_the_floor_counts_up_and_resets():
    env, p, obs, s = reset()
    hold = np.array([1.0 - 2.0 * p.rod_park, 0.0])
    cold = tavg_off_by(s, 30.0).replace(below_floor=s.n * 0 + 100.0)
    s1, _ = compute_next_state(hold, cold, p, integration_method=env.integration_method)
    assert float(s1.below_floor) == pytest.approx(100.0 + p.delta_t)
    s2, _ = compute_next_state(
        hold,
        s.replace(below_floor=s.n * 0 + 100.0),
        p,
        integration_method=env.integration_method,
    )
    assert float(s2.below_floor) == 0.0
