"""compressor_surge's shipped expert on its training days (owner, 2026-11-27,
picker "Decouple compressor's reference", from a practitioner review's release
plan).

Contract, the owner's acceptance of the pair (PHYSICS.md section 7):

- The limit cycle is gone. On the normal day, demand 0.7 to 0.95 with the point
  right of the control line and "the recycle shut or nearly" [C3, p. 2], the valve
  sits near shut: open past 0.02 (the recipes' own threshold for an open valve)
  on at most 5 % of the steps after the first 10 s (ours), and travelling at most
  ``GUARD_MAX_TRAVEL`` over the last 10 s, the guard's level for a valve that has
  stopped ringing.
- No trip and no counted surge on the training seeds, 100 to 109 (the recipes'),
  in every scenario.

The first is a finding today. The pair's valve is open on 46 % of those steps
and travels up to 0.52 in the last 10 s. The static decoupler the owner chose,
run as a fixed recipe on 2026-11-27, leaves both where they were (PHYSICS.md
section 7). It stays a strict xfail until a recipe removes the cycle.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.benchmark import seed_params
from target_gym.compressor_surge.experts import (
    GUARD_MAX_TRAVEL,
    make_compressor_surge_pid_controller,
)
from target_gym.registry import REGISTRY

pytestmark = pytest.mark.medium

SPEC = REGISTRY["compressor_surge"]
TRAIN = tuple(range(100, 110))
#: Steps of the 0.1 s plant in the first and the last 10 s.
SETTLE = TAIL = 100
OPEN, SHARE_OPEN = 0.02, 0.05


@pytest.fixture(scope="module")
def days():
    """Per scenario, the shipped expert's run on each training seed: the valve's
    opening after every step, whether the step tripped and the surge count."""
    env = SPEC.make_env()
    expert = make_compressor_surge_pid_controller()
    out = {}
    for s in scenarios.suite(SPEC):
        p = scenarios.params_for(SPEC, s)
        rows = [seed_params(env, p, seed) for seed in TRAIN]
        per_seed = jax.tree.map(
            lambda *v: jnp.stack([jnp.asarray(x) for x in v]), *rows
        )
        steps = int(p.max_steps_in_episode)

        def one(key, q):
            obs, state = env.reset_env(key, q)

            def body(carry, t):
                obs, state, memory = carry
                u, memory = expert.act(memory, obs)
                obs, state, _, _, info = env.step_env(
                    jax.random.fold_in(key, t), state, u, q
                )
                return (obs, state, memory), (state.x, info["tripped"], obs[7])

            carry = (obs, state, expert.init_state)
            return jax.lax.scan(body, carry, jnp.arange(steps))[1]

        keys = jnp.stack([jax.random.PRNGKey(seed) for seed in TRAIN])
        out[s.name] = [np.asarray(a) for a in jax.jit(jax.vmap(one))(keys, per_seed)]
    return out


@pytest.mark.xfail(
    strict=True,
    reason="finding (owner, 2026-11-27): the pair's valve rings on the normal day, "
    "open on 46 % of the steps after 10 s and travelling up to 0.52 in the last "
    "10 s; the static decoupler of the owner's recipe leaves both unchanged",
)
def test_on_the_normal_day_the_recycle_sits_near_shut(days):
    x, _, _ = days["normal"]
    assert float((x[:, SETTLE:] > OPEN).mean()) <= SHARE_OPEN
    travel = np.abs(np.diff(x[:, -(TAIL + 1) :], axis=1)).sum(axis=1)
    assert float(travel.max()) <= GUARD_MAX_TRAVEL


@pytest.mark.parametrize("scenario", ["normal", "slow_approach", "fast_approach"])
def test_no_trip_and_no_surge_on_the_training_seeds(days, scenario):
    _, tripped, surges = days[scenario]
    assert not tripped.any()
    assert float(surges.max()) == 0.0
