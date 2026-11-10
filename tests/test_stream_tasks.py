"""Every task wired for the stream keeps its contract (long-strata draft 4, S1, S2).

A task opts in with ``STREAM_READY = True`` on its env; ``stream.make_stream``
refuses the others. For each wired task:
- the stream never signals done, past the task's window N and through trips;
- two different policies on one seed meet the same outside world: every field the
  task names in ``OUTSIDE_WORLD``, the scenario sequence and the event onsets;
- the observation is the task's own: its shape, and finite.

The rates here are the test's, raised so a short run switches and has events; the
contract holds for any rates.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.registry import REGISTRY
from target_gym.stream import StreamEnv, make_stream


def _wired():
    names = []
    for name, spec in REGISTRY.items():
        if getattr(spec.make_env(), "STREAM_READY", False):
            names.append(name)
    return names


WIRED = _wired()


def test_a_task_not_wired_is_refused():
    unwired = next(n for n in REGISTRY if n not in WIRED)
    with pytest.raises(ValueError, match="not wired"):
        make_stream(unwired)


def _busy(name):
    """The task's stream with its scenarios switching every ~0.1 h and two events
    an hour in each, so a short run sees both."""
    stream = make_stream(name)
    fast = [
        dataclasses.replace(s, dwell_h=0.1, events_per_h=2.0) for s in stream.scenarios
    ]
    return StreamEnv(stream.env, stream.default_params, fast)


@partial(jax.jit, static_argnums=(0, 1))
def _roll(stream, side, steps_key):
    """A constant action at one end of the action range, ``len(keys)`` steps."""
    params = stream.default_params
    space = stream.action_space(params)
    action = jnp.broadcast_to(space.low if side < 0 else space.high, space.shape)
    obs, state = stream.reset(jax.random.PRNGKey(3), params)

    def body(state, k):
        obs, state, _, term, trunc, info = stream.step(k, state, action, params)
        world = {f: getattr(state.env, f) for f in stream.env.OUTSIDE_WORLD}
        return state, (term, trunc, info["scenario"], info["event"], world, obs)

    return jax.lax.scan(body, state, steps_key)[1]


@pytest.mark.parametrize("name", WIRED)
def test_a_wired_task_never_ends_and_both_policies_meet_one_world(name):
    stream = _busy(name)
    n = int(stream.default_params.max_steps_in_episode)
    low = _roll(stream, -1, jax.random.split(jax.random.PRNGKey(0), 2 * n))
    high = _roll(stream, +1, jax.random.split(jax.random.PRNGKey(1), 2 * n))
    for run in (low, high):
        assert not bool(run[0].any()) and not bool(run[1].any())
    np.testing.assert_array_equal(low[2], high[2])
    np.testing.assert_array_equal(low[3], high[3])
    assert int(low[3].sum()) > 0 and len(set(np.asarray(low[2]).tolist())) > 1
    for field in stream.env.OUTSIDE_WORLD:
        np.testing.assert_array_equal(
            np.asarray(low[4][field]), np.asarray(high[4][field]), err_msg=field
        )
    obs = np.asarray(low[5])
    assert obs.shape[1:] == tuple(stream.env.obs_shape)
    assert np.isfinite(obs).all()
