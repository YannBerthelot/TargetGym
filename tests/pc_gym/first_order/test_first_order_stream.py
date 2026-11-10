"""The first-order lag as a continuing plant (long-strata draft 4, S1, S2).

Contract:
- a window starts off its target and draws the target once; a stream starts in
  established operation, on its target, and changes the target as events: each
  change (``start_event``) a fresh draw over ``target_x_range``, observed from
  the next step and held until the next;
- a trip's restart puts the plant back on the target it is held to then;
- two policies on one seed meet one target sequence (S2).

The task is not opted in (``STREAM_READY``): the shared conformance suite's 2
events an hour give its 15 s run 0.008 events. The stream here is built with the
scenario the task would declare, at a rate scaled to its 7.5 s window.

Oracles: the declared target range, and the plant on its target.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.registry import REGISTRY
from target_gym.scenarios import Scenario
from target_gym.stream import StreamEnv

SPEC = REGISTRY["first_order"]
ENV = SPEC.make_env()
#: The stream's normal day: on target, a target change every window on average.
NORMAL = Scenario(
    "normal", {}, events_per_h=480.0, stream_overrides={"start_on_target": 1.0}
)
STREAM = StreamEnv(ENV, SPEC.make_test_params(), [NORMAL])
PARAMS = STREAM.scenario_params(STREAM.default_params, 0)


def test_a_stream_starts_and_restarts_on_its_target():
    _, s = ENV.reset_env(jax.random.PRNGKey(0), PARAMS)
    assert float(s.x) == float(s.target_x)
    # Past x_max the plant trips; it restarts on the target it was held to.
    held = s.replace(target_x=jnp.asarray(1.25))
    tripped = held.replace(x=jnp.asarray(PARAMS.x_max + 0.5))
    _, s, _, _, info = ENV.step_env(jax.random.PRNGKey(1), tripped, jnp.ones(1), PARAMS)
    assert bool(info["tripped"])
    assert float(s.x) == 1.25 and float(s.target_x) == 1.25


def test_a_target_change_draws_a_new_target_in_range_and_holds_it():
    _, s = ENV.reset_env(jax.random.PRNGKey(0), PARAMS)
    before = float(s.target_x)
    s = ENV.start_event(s, jax.random.PRNGKey(1), PARAMS)
    lo, hi = PARAMS.target_x_range
    new = float(s.target_x)
    assert new != before and lo <= new <= hi
    step = jax.jit(ENV.step_env)
    for k in range(3):
        obs, s, *_ = step(jax.random.PRNGKey(k), s, jnp.zeros(1), PARAMS)
        assert float(obs[1]) == new


def test_target_changes_are_fresh_draws_over_the_whole_range():
    _, s = ENV.reset_env(jax.random.PRNGKey(0), PARAMS)
    keys = jax.random.split(jax.random.PRNGKey(2), 400)
    targets = np.asarray(
        jax.vmap(lambda k: ENV.start_event(s, k, PARAMS).target_x)(keys)
    )
    lo, hi = PARAMS.target_x_range
    assert targets.min() >= lo and targets.max() <= hi
    se = (hi - lo) / np.sqrt(12.0 * len(targets))
    assert abs(targets.mean() - 0.5 * (lo + hi)) < 4.0 * se
    assert targets.min() < lo + 0.05 * (hi - lo)
    assert targets.max() > hi - 0.05 * (hi - lo)


def test_two_policies_meet_one_target_sequence():
    n = 2 * int(PARAMS.max_steps_in_episode)

    @jax.jit
    def roll(side):
        _, state = STREAM.reset(jax.random.PRNGKey(3), STREAM.default_params)

        def body(state, k):
            _, state, _, term, trunc, info = STREAM.step(
                k, state, jnp.full((1,), side), STREAM.default_params
            )
            return state, (term | trunc, info["event"], state.env.target_x)

        keys = jax.random.split(jax.random.PRNGKey(side > 0), n)
        return jax.lax.scan(body, state, keys)[1]

    low, high = roll(-1.0), roll(1.0)
    assert not bool(low[0].any()) and not bool(high[0].any())
    np.testing.assert_array_equal(low[1], high[1])
    assert int(low[1].sum()) > 0
    np.testing.assert_array_equal(low[2], high[2])
