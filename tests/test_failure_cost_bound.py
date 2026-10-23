"""A trip must cost more per step than any step the plant survives.

The failure term (``target_gym.reward``) rests on one inequality: a tripped
plant is down at ``failure_cost`` per step, "above the largest tracking cost
the envelope can produce". If a surviving step can cost more, the lump
``restart_steps * failure_cost`` can be cheaper than riding out a bad state, and
a policy learns to trip on purpose to get a fresh restart.

Each plant derives ``failure_cost`` by hand from its own worst case, and nothing
checked the derivation against the dynamics. This pushes every plant to its
extremes, with uniform random actions and with each bound held constant, and
requires the worst cost of a step that did not trip to stay below
``failure_cost``. It is an empirical lower bound on the true worst case, so a
pass is evidence rather than proof; a failure is a real counterexample.
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


def _worst_surviving_cost(name: str) -> float:
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.make_test_params()
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
        cost = np.where(tripped, -np.inf, -reward)
        worst = max(worst, float(cost.max()))
    return worst


@pytest.mark.parametrize(
    "name",
    [
        (
            pytest.param(n, marks=pytest.mark.xfail(reason=KNOWN_BELOW[n], strict=True))
            if n in KNOWN_BELOW
            else n
        )
        for n in sorted(REGISTRY)
    ],
)
def test_a_trip_costs_more_per_step_than_any_surviving_step(name):
    failure_cost = float(REGISTRY[name].make_test_params().failure_cost)
    worst = _worst_surviving_cost(name)
    assert worst < failure_cost, (
        f"{name}: a step that did not trip cost {worst:.4g}, above "
        f"failure_cost {failure_cost:.4g}, so tripping on purpose can pay"
    )
