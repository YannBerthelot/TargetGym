"""slow_mode measures how long a plant takes to forget where it started.

Contract (docs/proposals/long-strata.md, R1):
- two runs share the seed, so outside processes are identical and leave no gap;
- ``t_s`` is the decay time of the largest scaled state gap after the hand-over;
- the slowest state sets ``t_s``, observed or not;
- bookkeeping named in ``ignore`` is not fitted;
- an element that never forgets is named, and gives ``t_s = inf`` unless ignored.

Oracle: a linear plant ``x' = a x + u`` with the reference holding ``u = 0`` forgets
as ``a**t``, so ``t_s = -1 / ln(a)`` exactly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.slow_mode import coupled_pair, slow_mode


class State(NamedTuple):
    fast: jnp.ndarray
    slow: jnp.ndarray  # hidden: not in the observation
    walk: jnp.ndarray  # an outside process, drawn from the step key only
    total: jnp.ndarray  # a cumulative cost


class TwoModePlant:
    """``fast`` and ``slow`` decay at ``a_fast`` and ``a_slow``; the action drives both."""

    def __init__(self, a_fast=0.8, a_slow=0.98):
        self.a_fast, self.a_slow = a_fast, a_slow

    def reset_env(self, key, params):
        s = State(*(jnp.zeros(()) for _ in range(4)))
        return jnp.stack([s.fast]), s

    def step_env(self, key, s, action, params):
        u = action[0]
        walk = s.walk + jax.random.normal(key)
        fast = self.a_fast * s.fast + u + 0.1 * walk
        slow = self.a_slow * s.slow + u + 0.1 * walk
        total = s.total + jnp.abs(fast)
        new = State(fast, slow, walk, total)
        return jnp.stack([fast]), new, -jnp.abs(fast), False, {}


def hold_zero(params):
    return lambda obs, state: np.zeros(1)


def measure(plant, length, seeds=(0, 1, 2), ignore=("total",)):
    pairs = [
        coupled_pair(
            plant,
            None,
            hold_zero,
            seed=k,
            pre=30,
            length=length,
            offset=np.ones(1),
            low=-np.ones(1) * 5,
            high=np.ones(1) * 5,
        )
        for k in seeds
    ]
    return slow_mode(pairs, ignore=ignore)


def test_the_slowest_hidden_state_sets_t_s_at_its_closed_form_decay():
    result = measure(TwoModePlant(0.8, 0.98), length=400)
    # -1 / ln(0.98) = 49.50 steps; the fit is on exact exponentials, so 1 % covers it.
    assert result.t_s == pytest.approx(-1 / np.log(0.98), rel=0.01)
    assert "slow" in result.slowest


def test_a_faster_plant_forgets_faster():
    result = measure(TwoModePlant(0.5, 0.9), length=200)
    assert result.t_s == pytest.approx(-1 / np.log(0.9), rel=0.01)


def test_an_unnamed_cumulative_cost_is_reported_and_blocks_a_finite_t_s():
    result = measure(TwoModePlant(0.8, 0.98), length=400, ignore=())
    assert result.t_s == np.inf
    assert [n for n in result.never_forgot if "total" in n]
    assert not [n for n in result.never_forgot if "walk" in n]


def test_a_plant_integrator_that_never_forgets_gives_no_finite_t_s():
    result = measure(TwoModePlant(0.8, 1.0), length=100)
    assert result.t_s == np.inf
    assert [n for n in result.never_forgot if "slow" in n]
