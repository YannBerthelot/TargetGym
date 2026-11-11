"""The four-tank rig as a continuing plant (long-strata draft 4, S1).

Contract:
- a stream's event is one of the window's three kinds, in the window's own mix
  (``event_shares``): a setpoint move, an extra outlet opened in an upper tank, a
  cup poured into a lower tank;
- a setpoint move moves one lower tank's setpoint by ``setpoint_step`` either way
  from the next step, and returns after ``event_hold``; moves never stack, so the
  setpoints stay within one step of the operating point;
- an extra outlet drains one upper tank for ``leak_duration`` steps, then closes;
- a cup raises one lower tank by ``cup_volume`` over its area in one step;
- a stream starts with none of the window's events: only the stream starts them.

Oracles: the declared sizes and holds, the operating point's steady state, and a
tank's mass balance (a volume over an area).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.pc_gym.four_tank.env import steady_levels
from target_gym.stream import StreamEnv, make_stream

STREAM = make_stream("four_tank")
ENV = STREAM.env
BASE = STREAM.scenario_params(STREAM.default_params, 0)
STEP = jax.jit(ENV.step_env)


def _only(kind, **kw):
    """The stream's params with every event of one kind."""
    shares = [0.0, 0.0, 0.0]
    shares[kind] = 1.0
    return BASE.replace(event_shares=tuple(shares), **kw)


def _rest(params):
    _, s = ENV.reset_env(jax.random.PRNGKey(0), params)
    return s


def _run(state, params, n):
    """``n`` steps at the operating point's pumps: observations and states."""
    action = ENV.initial_action(state, params)

    def body(s, k):
        obs, s, *_ = ENV.step_env(k, s, action, params)
        return s, (obs, s.h1, s.h2, s.h3, s.h4)

    return jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(4), n))[1]


RUN = jax.jit(_run, static_argnums=(2,))


def test_a_setpoint_move_holds_one_tank_then_returns():
    p = _only(0, event_hold=30.0)  # s, the test's hold
    s = ENV.start_event(_rest(p), jax.random.PRNGKey(1), p)
    point = np.array(steady_levels(p.v_nominal, p)[:2])
    obs = np.asarray(RUN(s, p, 40)[0])
    moved = obs[:, 4:6] - point
    # One tank, by the step, either way, for the hold; then both on the point.
    np.testing.assert_allclose(
        np.abs(moved[:29]).max(axis=1), p.setpoint_step, rtol=1e-4
    )
    np.testing.assert_allclose(np.abs(moved[:29]).min(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(moved[30:], 0.0, atol=1e-7)


def test_setpoint_moves_never_stack():
    p = _only(0, event_hold=1000.0)
    s = _rest(p)
    point = np.array(steady_levels(p.v_nominal, p)[:2])
    for k in range(12):
        s = ENV.start_event(s, jax.random.PRNGKey(10 + k), p)
        obs = np.asarray(RUN(s, p, 1)[0])[0]
        moved = np.abs(obs[4:6] - point)
        np.testing.assert_allclose(moved.max(), p.setpoint_step, rtol=1e-4)
        np.testing.assert_allclose(moved.min(), 0.0, atol=1e-7)


def test_an_extra_outlet_drains_one_upper_tank_then_closes():
    p = _only(1)
    s = ENV.start_event(_rest(p), jax.random.PRNGKey(1), p)
    n = int(p.leak_duration)
    _, _, _, h3, h4 = (np.asarray(x) for x in RUN(s, p, n + 50))
    drained = h3 if h3[n - 1] < h4[n - 1] else h4
    other = h4 if drained is h3 else h3
    # The drained tank falls while the outlet is open and refills once it shuts;
    # the other holds its steady level.
    assert np.all(np.diff(drained[:n]) < 0.0)
    assert np.all(np.diff(drained[n + 1 :]) > 0.0)
    np.testing.assert_allclose(other, other[0], rtol=1e-5)


def test_a_cup_raises_one_lower_tank_by_its_volume_over_its_area():
    p = _only(2)
    s = _rest(p)
    s = ENV.start_event(s, jax.random.PRNGKey(1), p)
    _, h1, h2, _, _ = (np.asarray(x) for x in RUN(s, p, 1))
    rise = np.array([h1[0] - s.h1, h2[0] - s.h2])
    by_area = p.cup_volume / np.array([p.A1, p.A2])
    k = int(np.argmax(rise))
    # One step's own outflow change is under 1 % of the rise.
    np.testing.assert_allclose(rise[k], by_area[k], rtol=0.01)
    np.testing.assert_allclose(rise[1 - k], 0.0, atol=1e-6)


def test_event_kinds_come_in_the_declared_mix():
    s = _rest(BASE)
    keys = jax.random.split(jax.random.PRNGKey(2), 600)
    after = jax.vmap(lambda k: ENV.start_event(s, k, BASE))(keys)
    moved = np.asarray(
        (after.target_h1 != s.target_h1) | (after.target_h2 != s.target_h2)
    )
    leak = np.asarray(after.leak_start == s.time)
    cup = np.asarray(after.cup_time == s.time)
    assert np.all(moved.astype(int) + leak + cup == 1)
    shares = np.array([moved.mean(), leak.mean(), cup.mean()])
    want = np.array(BASE.event_shares)
    se = np.sqrt(want * (1.0 - want) / len(keys))
    assert np.all(np.abs(shares - want) < 4.0 * se), shares


def test_a_stream_starts_without_the_window_s_events():
    quiet = StreamEnv(
        ENV,
        STREAM.default_params,
        [dataclasses.replace(STREAM.scenarios[0], events_per_h=0.0)],
    )
    params = quiet.default_params
    n = int(params.max_steps_in_episode)  # the window's events fall inside it

    def body(state, k):
        obs, state, *_ = quiet.step(
            k, state, ENV.initial_action(state.env, params), params
        )
        return state, (obs, jnp.stack([state.env.h3, state.env.h4]))

    _, state = quiet.reset(jax.random.PRNGKey(5), params)
    obs, upper = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(6), n))[
        1
    ]
    obs, upper = np.asarray(obs), np.asarray(upper)
    point = np.array(steady_levels(params.v_nominal, params))
    np.testing.assert_allclose(obs[:, 4:6] - point[:2], 0.0, atol=1e-7)
    # No outlet, no cup: the upper tanks hold their steady levels.
    np.testing.assert_allclose(upper - point[2:], 0.0, atol=1e-6)
