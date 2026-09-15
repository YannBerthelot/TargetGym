"""The reach-and-hold protocol on synthetic cost series: the renewal
decomposition of Theorem 4, and what the reach columns mean."""

from __future__ import annotations

import numpy as np
import pytest

from target_gym import eval as E


def _episode(cost, target_change):
    n = len(cost)
    return E.Episode(
        cost=np.asarray(cost, float),
        in_band=np.ones(n, bool),
        target_change=np.asarray(target_change, bool),
        failed=np.zeros(n, bool),
    )


def test_gain_is_hold_plus_change_rate_times_reach_cost():
    """Constant hold cost 0.1, an excursion of +1 for 5 steps after each target
    change at rate p = 0.01: gain ~ 0.1 + p * 5, hold ~ 0.1, reach ~ 5."""
    rng = np.random.default_rng(0)
    T, p = 20000, 0.01
    tc = rng.random(T) < p
    cost = np.full(T, 0.1)
    for i in np.flatnonzero(tc):
        cost[i : i + 5] += 1.0
    m = E.evaluate([_episode(cost, tc)], burn_in=100, rho_floor=0.1, rho_ref=0.2)
    assert m["hold"] == pytest.approx(0.1, abs=0.01)
    assert m["reach_cost"] == pytest.approx(5.0, abs=0.5)
    assert m["gain"] == pytest.approx(0.1 + p * 5.0, abs=0.01)
    assert m["unsettled_fraction"] == 0.0
    assert m["nea"] == pytest.approx(0.5, abs=0.05)


def test_transient_cost_compares_controllers_that_reach_cost_cannot():
    """Reach B is the transient above the cycle's own hold level, over a
    transient detected against that level: a controller holding far off has
    its excursion swallowed (B small). The transient cost sums the first
    ``window`` steps of the cycle with nothing subtracted, and orders the two
    controllers by what the change actually cost."""
    tc = np.zeros(200, bool)
    tc[0] = tc[100] = True
    near = np.full(200, 1.0)
    far = np.full(200, 100.0)
    for c in (near, far):
        c[0:10] += 50.0
        c[100:110] += 50.0
    b_near, _ = E.reach_cost([_episode(near, tc)])
    b_far, _ = E.reach_cost([_episode(far, tc)])
    assert b_near == pytest.approx(500.0)
    assert b_far < b_near  # the 150 vs 100 excursion is "within hold" for far
    t_near, _ = E.transient_cost([_episode(near, tc)], window=20)
    t_far, _ = E.transient_cost([_episode(far, tc)], window=20)
    assert t_near == pytest.approx(500.0 + 20.0)
    assert t_far == pytest.approx(500.0 + 2000.0)


def test_a_cost_still_rising_at_the_end_reads_as_unsettled_with_negative_reach():
    """No hold reached in the window: the 'hold level' is above the transient,
    B comes out negative, and the cycle is counted as unsettled."""
    tc = np.zeros(100, bool)
    tc[0] = True
    rising = np.linspace(0.0, 10.0, 100)
    ep = _episode(rising, tc)
    b, _ = E.reach_cost([ep])
    assert b < 0.0
    assert E.unsettled_fraction([ep]) == 1.0
    m = E.evaluate([ep], burn_in=0)
    assert m["unsettled_fraction"] == 1.0 and m["reach_cost"] < 0.0
