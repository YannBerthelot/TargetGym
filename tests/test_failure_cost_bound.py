"""An episode with a trip must cost more than the worst episode without one.

The failure term (``target_gym.reward``) exists so that a policy never learns to
trip on purpose to get a fresh restart. A trip charges ``restart_steps *
failure_cost`` once (``reward.trip_cost``), and every other term is >= 0, so an
episode with a trip costs at least that lump. If the lump exceeds the worst
episode the plant survives, tripping can never pay. This is the bound, per
episode (owner, 2026-11-08). It replaced a per-step bound, which a plant that
prices wear broke without any reward for tripping: one violent tower cycle on
wind_turbine is priced above a step of downtime, while the 2 h restart still
costs far more than any surviving 10 min.

This pushes every plant to its extremes, with uniform random actions and with
each bound held constant, and requires the trip's lump to exceed the worst
total cost of an episode that did not trip. It is an empirical lower bound on
the true worst case, so a pass is evidence rather than proof; a failure is a
real counterexample.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.registry import REGISTRY

SEEDS = 8

#: Plants whose ``failure_cost`` is known to sit below a reachable step cost.
KNOWN_BELOW: dict[str, str] = {
    # cement_kiln sat here until v3: a kiln starved of fuel ran cold to ~17 %
    # free lime without tripping. It now trips at ``lime_max`` (5 %), the
    # level its failure_cost is sized on.
}


def _worst_surviving_episode(name: str, params=None) -> float:
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.make_test_params() if params is None else params
    space = env.action_space(params)
    low, high = jnp.asarray(space.low), jnp.asarray(space.high)
    steps = int(params.max_steps_in_episode)

    def rollout(key, mode):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset_env(reset_key, params)

        def body(state, k):
            uniform = jax.random.uniform(k, low.shape, minval=low, maxval=high)
            action = jnp.where(mode == 0, uniform, jnp.where(mode == 1, low, high))
            # A constant step key, as every shipped rollout helper passes.
            _, state, reward, _, info = env.step_env(step_key, state, action, params)
            return state, (reward, info["tripped"])

        _, out = jax.lax.scan(body, state, jax.random.split(step_key, steps))
        return out

    run = jax.jit(jax.vmap(rollout, in_axes=(0, None)))
    keys = jax.random.split(jax.random.PRNGKey(0), SEEDS)
    worst = 0.0
    for mode in (0, 1, 2):  # random, all low, all high
        reward, tripped = (np.asarray(x) for x in run(keys, mode))
        cost = np.where(tripped.any(axis=1), -np.inf, -reward.sum(axis=1))
        worst = max(worst, float(cost.max()))
    return worst


def _trip_lump(params) -> float:
    return float(params.restart_steps) * float(params.failure_cost)


@pytest.mark.parametrize(
    "name",
    [
        (
            pytest.param(n, marks=pytest.mark.xfail(reason=KNOWN_BELOW[n], strict=True))
            if n in KNOWN_BELOW
            else n
        )
        for n in sorted(REGISTRY)
        # A plant with no trip (charter B5: hvac, battery) has nothing to
        # bound; its failure cost is 0 by construction.
        if float(REGISTRY[n].make_test_params().failure_cost) > 0.0
    ],
)
def test_an_episode_with_a_trip_costs_more_than_any_without(name):
    params = REGISTRY[name].make_test_params()
    worst = _worst_surviving_episode(name, params)
    lump = _trip_lump(params)
    assert worst < lump, (
        f"{name}: an episode that did not trip cost {worst:.4g}, above the "
        f"trip's {lump:.4g}, so tripping on purpose can pay"
    )


def test_the_bound_catches_a_deliberately_cheap_trip():
    """The check must fail where tripping pays: wind_turbine with its trip
    priced at a millionth of the sourced one."""
    params = REGISTRY["wind_turbine"].make_test_params()
    cheap = params.replace(failure_cost=params.failure_cost * 1e-6)
    assert _worst_surviving_episode("wind_turbine", cheap) >= _trip_lump(cheap)
