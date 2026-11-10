"""Every task wired for the stream keeps its contract (long-strata draft 4, S1, S2).

A task opts in with ``STREAM_READY = True`` on its env; ``stream.make_stream``
refuses the others. For each wired task:
- the stream never signals done, past the task's window N and through trips;
- two different policies on one seed meet the same outside world: every field the
  task names in ``OUTSIDE_WORLD``, the scenario sequence and the event onsets.
  ``RECORDS`` (a surge log, a permit clock) are history the policy writes, carried
  through a trip but not shared, so they are not compared;
- the observation is the task's own: its shape, and finite.

The rates here are the test's, sized to the run so it switches and has events
whatever the task's step length; the contract holds for any rates. A task with one scenario has nothing to switch, and a
task with no ``start_event`` hook has no events: each is checked on what it has.
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
    unwired = [n for n in REGISTRY if n not in WIRED]
    if not unwired:
        pytest.skip("every task is wired")
    with pytest.raises(ValueError, match="not wired"):
        make_stream(unwired[0])


def _has_events(env) -> bool:
    return callable(getattr(env, "start_event", None))


#: Switch draws and events a busy run expects. Sized to the run, not per hour: a
#: plant on 0.1 s steps runs minutes in 2 N steps. At 30, a seed missing every event
#: is a chance of e^-30, and one never leaving a scenario of weight 0.7 is 0.7^30.
_EXPECTED = 30.0


def _busy(name):
    """The task's stream for a run of 2 N steps, with about 30 switch draws and,
    where the task has events, about 30 events over the run, whatever its step."""
    from target_gym.registry import control_step_seconds

    stream = make_stream(name)
    steps = 2 * int(stream.default_params.max_steps_in_episode)
    hours = steps * control_step_seconds(stream.env, stream.default_params) / 3600.0
    rate = _EXPECTED / hours if _has_events(stream.env) else 0.0
    fast = [
        dataclasses.replace(s, dwell_h=hours / _EXPECTED, events_per_h=rate)
        for s in stream.scenarios
    ]
    return StreamEnv(stream.env, stream.default_params, fast), steps


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
    stream, steps = _busy(name)
    low = _roll(stream, -1, jax.random.split(jax.random.PRNGKey(0), steps))
    high = _roll(stream, +1, jax.random.split(jax.random.PRNGKey(1), steps))
    for run in (low, high):
        assert not bool(run[0].any()) and not bool(run[1].any())
    np.testing.assert_array_equal(low[2], high[2])
    np.testing.assert_array_equal(low[3], high[3])
    if _has_events(stream.env):
        assert int(low[3].sum()) > 0
    if len(stream.scenarios) > 1:
        assert len(set(np.asarray(low[2]).tolist())) > 1
    for field in stream.env.OUTSIDE_WORLD:
        np.testing.assert_array_equal(
            np.asarray(low[4][field]), np.asarray(high[4][field]), err_msg=field
        )
    obs = np.asarray(low[5])
    assert obs.shape[1:] == tuple(stream.env.obs_shape)
    assert np.isfinite(obs).all()
