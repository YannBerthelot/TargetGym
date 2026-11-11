"""The CSTR as a continuing plant (long-strata draft 4, S1).

Contract:
- an event carries its own size: started in one scenario, it runs to its return
  through a switch to another, and lasts ``event_hold`` minutes from its onset;
- the setpoint move, the catalyst loss and the feed rise return; the cooling loss
  still lasts ``cooling_loss`` minutes;
- the feed's daily cycle repeats exactly, however long the plant has run;
- a stream starts with no event of the window: only the stream starts events.

Oracles: the scenario's declared sizes and hold, and the cycle's period in steps.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.stream import StreamEnv, make_stream


def _stream_and_params(name):
    stream = make_stream("cstr")
    names = [s.name for s in stream.scenarios]
    return stream, stream.scenario_params(stream.default_params, names.index(name))


def test_a_setpoint_move_runs_through_a_switch_and_returns_after_its_hold():
    stream, moved = _stream_and_params("setpoint_step")
    _, normal = _stream_and_params("normal")
    env, step = stream.env, jax.jit(stream.env.step_env)
    _, s = env.reset_env(jax.random.PRNGKey(0), normal)
    s = env.start_event(s, jax.random.PRNGKey(1), moved)
    hold = int(round(moved.event_hold / moved.delta_t))  # steps
    action = env.initial_action(s, normal)
    seen = []
    for k in range(hold + 20):
        # Every step after the onset runs on the normal scenario's params.
        obs, s, *_ = step(jax.random.PRNGKey(k), s, action, normal)
        seen.append(float(obs[3]))
    seen = np.array(seen)
    # Active over [onset, onset + hold): observed raised from the first step on.
    np.testing.assert_allclose(seen[: hold - 1], normal.T_setpoint + 2.0, atol=1e-4)
    np.testing.assert_allclose(seen[hold:], normal.T_setpoint, atol=1e-4)


def test_the_feed_s_day_repeats_exactly_however_long_the_plant_has_run():
    stream, p = _stream_and_params("normal")
    env, step = stream.env, jax.jit(stream.env.step_env)
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    per_day = int(round(p.feed_T_period / p.delta_t))  # 5760 steps
    action = env.initial_action(s, p)
    key = jax.random.PRNGKey(2)
    early = step(key, s.replace(time=jnp.int32(100)), action, p)[1].T_feed
    late = step(key, s.replace(time=jnp.int32(100 + 2000 * per_day)), action, p)[1]
    assert float(early) == float(late.T_feed)


def test_a_stream_starts_without_the_window_s_event():
    stream = make_stream("cstr")
    moved = next(s for s in stream.scenarios if s.name == "setpoint_step")
    quiet = StreamEnv(
        stream.env,
        stream.default_params,
        [dataclasses.replace(moved, events_per_h=0.0)],
    )
    params = quiet.default_params
    n = int(params.max_steps_in_episode)

    def body(state, k):
        obs, state, *_ = quiet.step(k, state, jnp.zeros(1), params)
        return state, obs[3]

    _, state = quiet.reset(jax.random.PRNGKey(5), params)
    setpoints = jax.lax.scan(
        body, state, jax.random.split(jax.random.PRNGKey(6), 2 * n)
    )[1]
    # The window would move the setpoint 2 K between minutes 30 and 150.
    np.testing.assert_allclose(np.asarray(setpoints), params.T_setpoint, atol=1e-4)
