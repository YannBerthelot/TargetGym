"""slow_mode measures how long a plant takes to forget where it started.

Contract (docs/proposals/long-strata.md, R1):
- two runs share the seed, so outside processes are identical and leave no gap;
- ``t_s`` is the decay time of the extra cost still to come from B's start
  (the burn-in's bias), however the state carries it;
- the state's own decay is reported beside it: the slowest element, observed
  or not, and the elements that never forget;
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

    def __init__(self, a_fast=0.8, a_slow=0.98, slow_costs=True):
        self.a_fast, self.a_slow, self.slow_costs = a_fast, a_slow, slow_costs

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
        # A cost linear in the states, so the cost gap is the state gap.
        cost = fast + (slow if self.slow_costs else 0.0)
        return jnp.stack([fast]), new, -cost, False, {}


def hold_zero(params):
    return lambda obs, state: np.zeros(1)


def measure(
    plant,
    length,
    seeds=(0, 1, 2),
    ignore=(".total",),
    bias=False,
    offset=1.0,
    observe=None,
):
    pairs = [
        coupled_pair(
            plant,
            None,
            hold_zero,
            bias=bias,
            observe=observe,
            seed=k,
            pre=30,
            length=length,
            offset=offset * np.ones(1),
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
    assert result.state_t_s == pytest.approx(-1 / np.log(0.98), rel=0.01)
    assert "slow" in result.slowest


def test_a_hidden_state_the_cost_never_reads_does_not_hold_the_burn_in():
    # The burn-in removes bias in the cost; a state that never reaches the cost
    # is reported, but t_s is the fast mode the cost reads.
    result = measure(TwoModePlant(0.8, 0.98, slow_costs=False), length=400)
    assert result.t_s == pytest.approx(-1 / np.log(0.8), rel=0.02)
    assert "slow" in result.slowest
    assert result.state_t_s == pytest.approx(-1 / np.log(0.98), rel=0.01)


def test_a_faster_plant_forgets_faster():
    result = measure(TwoModePlant(0.5, 0.9), length=200)
    assert result.t_s == pytest.approx(-1 / np.log(0.9), rel=0.01)


def test_an_unnamed_cumulative_cost_is_reported_and_blocks_a_finite_state_t_s():
    result = measure(TwoModePlant(0.8, 0.98), length=400, ignore=())
    assert result.state_t_s == np.inf
    assert [n for n in result.never_forgot if "total" in n]
    assert not [n for n in result.never_forgot if "walk" in n]


def test_a_plant_integrator_that_never_forgets_gives_no_finite_t_s():
    # It costs on every step, so its bias never leaves the cost either.
    result = measure(TwoModePlant(0.8, 1.0), length=100)
    assert result.t_s == np.inf
    assert result.state_t_s == np.inf
    assert [n for n in result.never_forgot if "slow" in n]


def test_a_closed_loop_excursion_finds_the_same_decay():
    # The decay rate is the plant's under the reference, not the excursion's.
    result = measure(TwoModePlant(0.8, 0.98), length=400, bias=True)
    assert result.t_s == pytest.approx(-1 / np.log(0.98), rel=0.01)


def test_an_ignored_name_matches_whole_fields_only():
    # ".slow" must not skip a field merely starting with it; here it skips "slow"
    # itself, so the fast mode is all the state diagnostic has left.
    result = measure(TwoModePlant(0.8, 0.98), length=400, ignore=(".total", ".slow"))
    assert result.state_t_s == pytest.approx(-1 / np.log(0.8), rel=0.02)


def test_an_excursion_that_barely_moves_the_plant_gives_no_t_s():
    # Run B stays where A is: no start gap, nothing to fit, and no number.
    result = measure(TwoModePlant(0.8, 0.98), length=200, offset=0.0)
    assert np.isnan(result.t_s)


def test_angles_are_compared_on_the_circle():
    # Two headings a whole turn apart are the same heading.
    from target_gym.slow_mode import wrapped_gap

    gap = wrapped_gap(np.array([0.1, 2 * np.pi + 0.1]), np.array([0.1, 0.1]))
    np.testing.assert_allclose(gap, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        wrapped_gap(np.array([3.0]), np.array([-3.0])), [2 * np.pi - 6.0]
    )


def test_a_derived_quantity_stands_in_for_the_raw_state_it_replaces():
    # A path task measures its path-relative error, not a world-frame coordinate:
    # with the raw slow state ignored, a quantity observed from it still sets t_s.
    result = measure(
        TwoModePlant(0.8, 0.98),
        length=400,
        ignore=(".total", ".slow"),
        observe=lambda state, params: {"slow_error": 2.0 * state.slow},
    )
    assert result.state_t_s == pytest.approx(-1 / np.log(0.98), rel=0.01)
    assert result.slowest == "path.slow_error"


class Ring(NamedTuple):
    c: jnp.ndarray
    s: jnp.ndarray


class Oscillator:
    """An undamped rotation the action pushes: an excursion changes its amplitude
    for good, so the gap keeps swinging back up although each element passes
    through zero, as a limit cycle's phase or a path's phase does."""

    def reset_env(self, key, params):
        return jnp.zeros(1), Ring(jnp.zeros(()), jnp.zeros(()))

    def step_env(self, key, st, action, params):
        w = 0.1
        c = jnp.cos(w) * st.c - jnp.sin(w) * st.s + action[0]
        s = jnp.sin(w) * st.c + jnp.cos(w) * st.s
        return jnp.stack([c]), Ring(c, s), -jnp.abs(c), False, {}


def test_a_gap_that_keeps_swinging_back_never_forgot():
    result = measure(Oscillator(), length=400, ignore=())
    assert result.state_t_s == np.inf
    assert result.floor > 0.5


def _pair(excess, trips=0):
    """A pair with a decaying state gap and the given excess cost per step."""
    from target_gym.slow_mode import Pair

    t = np.arange(len(excess), dtype=float)
    gap = np.exp(-t / 10.0)[:, None]
    values = np.sin(t)[:, None]
    return Pair(values, gap, (".x",), np.asarray(excess, float), 0, trips)


def test_a_pair_that_trips_is_left_out_of_the_cost():
    # A trip's lump cost would swamp the start's bias; the clean pairs decide.
    t = np.arange(400, dtype=float)
    clean = [_pair(np.exp(-t / 20.0)) for _ in range(3)]
    tripped = _pair(np.where(t == 300, 1e9, np.exp(-t / 20.0)), trips=1)
    result = slow_mode(clean + [tripped])
    assert result.t_s == pytest.approx(20.0, rel=0.02)


def test_a_start_bias_lost_in_its_own_noise_is_no_bias():
    # Excess that only flickers about zero: nothing for a burn-in to remove.
    rng = np.random.default_rng(0)
    pairs = [_pair(1e-3 * rng.standard_normal(400)) for _ in range(3)]
    assert np.isnan(slow_mode(pairs).t_s)
