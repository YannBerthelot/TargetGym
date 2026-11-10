"""The pH neutralization as a continuing plant (long-strata draft 4, S1).

Contract:
- an influent upset (the strong acid's strength, the metered flow) carries its own
  size: started in one scenario, it runs to its return through a switch to
  another. It ramps over ``acid_upset_ramp`` to a level within its scenario's
  size of nominal, holds it for ``event_hold`` from its onset, then ramps back to
  nominal;
- an upset of one kind does not end an upset of the other kind in progress;
- an upset keeps its shape however long the plant has run;
- a stream starts with none of the window's upsets: only the stream starts them.

Oracles: the scenarios' declared sizes and holds, the ramp's length, nominal 1.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.ph_neutralization.env import acid_scale, influent_flow
from target_gym.stream import StreamEnv, make_stream

STREAM = make_stream("ph_neutralization")
NAMES = [s.name for s in STREAM.scenarios]
ENV = STREAM.env
STEP = jax.jit(ENV.step_env)
HOLD_S = 1800.0  # the tests' hold, so a run stays short; the contract holds for any


def _params(name):
    return STREAM.scenario_params(STREAM.default_params, NAMES.index(name))


def _run(state, params, n):
    """``n`` steps on ``params`` at the reset's valves: the acid and flow scales
    each step runs on."""
    action = ENV.initial_action(state, params)
    acid, flow = [], []
    for k in range(n):
        acid.append(float(acid_scale(state, params)))
        flow.append(float(influent_flow(state, params) / params.q1))
        _, state, *_ = STEP(jax.random.PRNGKey(k), state, action, params)
    return np.array(acid), np.array(flow), state


@pytest.mark.parametrize(
    "name, channel, size_field",
    [("acid_strength", 0, "acid_upset_size"), ("flow_swing", 1, "flow_swing")],
)
def test_an_upset_holds_its_level_through_a_switch_then_ramps_back(
    name, channel, size_field
):
    upset = _params(name).replace(event_hold=HOLD_S)
    normal = _params("normal")
    _, s = ENV.reset_env(jax.random.PRNGKey(0), normal)
    s = ENV.start_event(s, jax.random.PRNGKey(1), upset)
    ramp = int(upset.acid_upset_ramp / upset.delta_t)  # 120 steps
    hold = int(HOLD_S / upset.delta_t)  # 360 steps
    runs = _run(s, normal, hold + ramp + 10)
    moved, other = runs[channel], runs[1 - channel]
    # Ramped up, then held at a level within the scenario's size of nominal.
    level = moved[ramp]
    np.testing.assert_allclose(moved[ramp:hold], level, rtol=1e-6)
    assert 0.0 < abs(level - 1.0) <= getattr(upset, size_field)
    # Back to nominal one ramp after the hold ends, the other kind untouched.
    np.testing.assert_allclose(moved[hold + ramp :], 1.0, rtol=1e-6)
    np.testing.assert_allclose(other, 1.0, rtol=1e-6)


def test_an_acid_upset_does_not_end_a_flow_swing_in_progress():
    both = _params("bad_day").replace(event_hold=HOLD_S)
    acid = _params("acid_strength").replace(event_hold=HOLD_S)
    normal = _params("normal")
    _, s = ENV.reset_env(jax.random.PRNGKey(0), normal)
    s = ENV.start_event(s, jax.random.PRNGKey(1), both)
    ramp = int(both.acid_upset_ramp / both.delta_t)
    _, flow, s = _run(s, normal, 2 * ramp)
    swing = flow[-1]
    assert abs(swing - 1.0) > 0.0
    s = ENV.start_event(s, jax.random.PRNGKey(2), acid)
    hold = int(HOLD_S / both.delta_t)
    _, flow, _ = _run(s, normal, hold - 2 * ramp)
    np.testing.assert_allclose(flow, swing, rtol=1e-6)


def test_an_upset_keeps_its_shape_however_long_the_plant_has_run():
    upset = _params("acid_strength").replace(event_hold=HOLD_S)
    normal = _params("normal")
    _, s = ENV.reset_env(jax.random.PRNGKey(0), normal)
    half = int(0.5 * upset.acid_upset_ramp / upset.delta_t)  # half the ramp
    levels = []
    # 1e8 steps on, float32 counts the plant's seconds only to 32 s.
    for start in (100, 100_000_000):
        t = s.replace(time=jnp.int32(start))
        t = ENV.start_event(t, jax.random.PRNGKey(1), upset)
        levels.append(float(acid_scale(t.replace(time=t.time + half), normal)))
    np.testing.assert_allclose(levels[1], levels[0], rtol=1e-6)
    assert levels[0] != 1.0


def test_a_stream_starts_without_the_window_s_upsets():
    bad = next(s for s in STREAM.scenarios if s.name == "bad_day")
    quiet = StreamEnv(
        ENV, STREAM.default_params, [dataclasses.replace(bad, events_per_h=0.0)]
    )
    params = quiet.default_params
    n = int(params.max_steps_in_episode)  # the window's upsets fall inside it

    def body(state, k):
        _, state, *_ = quiet.step(
            k, state, ENV.initial_action(state.env, params), params
        )
        p = quiet.scenario_params(params, 0)
        return state, (acid_scale(state.env, p), influent_flow(state.env, p) / p.q1)

    _, state = quiet.reset(jax.random.PRNGKey(5), params)
    acid, flow = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(6), n))[
        1
    ]
    np.testing.assert_array_equal(np.asarray(acid), 1.0)
    np.testing.assert_allclose(np.asarray(flow), 1.0, rtol=1e-6)
