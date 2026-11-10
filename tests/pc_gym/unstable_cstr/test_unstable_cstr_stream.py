"""The LDPE autoclave as a continuing plant (long-strata draft 4, S1).

Contract:
- an event carries its own size: started in one scenario, it runs to its return
  through a switch to another, and lasts ``event_hold`` seconds from its onset;
- each upset returns: the grade change's setpoint move, the feed's warming, the
  initiator's lost efficiency and the feed's rise;
- the monomer feed's wander keeps its period however long the plant has run;
- a stream starts with no event of the window: only the stream starts events.

Oracles: the scenarios' declared sizes and holds, and the wander's period.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.pc_gym.unstable_cstr.env import events
from target_gym.registry import REGISTRY
from target_gym.stream import StreamEnv, make_stream

STREAM = make_stream("unstable_cstr")
NAMES = [s.name for s in STREAM.scenarios]


def _params(name):
    return STREAM.scenario_params(STREAM.default_params, NAMES.index(name))


def _at_rest(params):
    return STREAM.env.reset_env(jax.random.PRNGKey(0), params)[1]


@pytest.mark.parametrize(
    "name",
    ["grade_change", "feed_temperature", "initiator_efficiency", "throughput_step"],
)
def test_an_upset_holds_its_size_through_a_switch_then_returns(name):
    moved, normal = _params(name), _params("normal")
    quiet = _at_rest(normal)
    started = STREAM.env.start_event(quiet, jax.random.PRNGKey(1), moved)
    hold = float(moved.event_hold)  # s; the onset is the reset's second, 0
    # The scenario has switched to normal: the event still runs on its declared
    # size, against the same second with no event. The setpoint moves, the feed
    # warms, the initiator loses a fraction of its radicals, the feed flow rises.
    for second in (1.0, hold - 1.0):
        sp, T_f, f, scale = events(started, normal, second)
        sp0, T_f0, f0, scale0 = events(quiet, normal, second)
        np.testing.assert_allclose(
            [sp - sp0, T_f - T_f0, f / f0, scale / scale0],
            [
                moved.setpoint_step,
                moved.feed_T_step,
                1.0 - moved.efficiency_drop,
                1.0 + moved.feed_flow_step,
            ],
            rtol=1e-5,
            atol=1e-4,
        )
    later = hold + 1.0
    np.testing.assert_array_equal(
        np.array(events(started, normal, later)),
        np.array(events(quiet, normal, later)),
    )


def test_a_grade_change_moves_the_setpoint_seen_until_its_hold_ends():
    moved = _params("grade_change").replace(event_hold=30.0)  # s, the test's hold
    normal = _params("normal")
    env, step = STREAM.env, jax.jit(STREAM.env.step_env)
    s = env.start_event(_at_rest(normal), jax.random.PRNGKey(1), moved)
    action = env.initial_action(s, normal)
    seen = []
    for k in range(40):
        obs, s, *_ = step(jax.random.PRNGKey(k), s, action, normal)
        seen.append(float(obs[2]))
    seen = np.array(seen)
    # Active over [0, 30) s: the setpoint seen after each step is the next one's.
    np.testing.assert_allclose(seen[:29], normal.T_setpoint + 24.0, atol=1e-4)
    np.testing.assert_allclose(seen[30:], normal.T_setpoint, atol=1e-4)


def test_the_feed_s_wander_keeps_its_period_however_long_the_plant_has_run():
    p = _params("normal")
    env, step = STREAM.env, jax.jit(STREAM.env.step_env)
    s = _at_rest(p)
    action = env.initial_action(s, p)
    key = jax.random.PRNGKey(2)
    period = int(round(p.feed_T_wander_period / p.delta_t))  # 600 steps
    # 1e8 s on, float32 counts seconds only to 8 s: a clock read there would
    # move the feed by up to 2 pi * 8 / 600 rad of its 1 K wander.
    late_time = 100 + period * (100_000_000 // period)
    early = step(key, s.replace(time=jnp.int32(100)), action, p)[1].T
    late = step(key, s.replace(time=jnp.int32(late_time)), action, p)[1].T
    # One second's response to the feed through a 31 s residence: float32's
    # spacing at 513 K is 6e-5 K.
    np.testing.assert_allclose(float(late), float(early), atol=1e-4)


def test_a_stream_starts_without_the_window_s_event():
    window = scenarios.params_for(
        REGISTRY["unstable_cstr"],
        next(s for s in STREAM.scenarios if s.name == "grade_change"),
    )
    moved = next(s for s in STREAM.scenarios if s.name == "grade_change")
    quiet = StreamEnv(
        STREAM.env,
        STREAM.default_params,
        [dataclasses.replace(moved, events_per_h=0.0)],
    )
    params = quiet.default_params
    # Past the latest second the window would move the setpoint at.
    n = int(window.event_window[1] / window.delta_t) + 100

    def body(state, k):
        obs, state, *_ = quiet.step(k, state, jnp.zeros(1), params)
        return state, obs[2]

    _, state = quiet.reset(jax.random.PRNGKey(5), params)
    setpoints = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(6), n))[1]
    np.testing.assert_allclose(np.asarray(setpoints), params.T_setpoint, atol=1e-4)
