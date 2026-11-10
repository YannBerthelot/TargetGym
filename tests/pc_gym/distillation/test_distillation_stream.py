"""The distillation column as a continuing plant (long-strata draft 4, S1).

Contract:
- a feed step (the rate, metered; the composition, hidden) carries its own size:
  started in one scenario, it runs through a switch to another at its
  scenario's size and a drawn sign, holds ``event_hold`` minutes from its onset,
  then returns to nominal;
- a step of one kind does not end a step of the other kind in progress;
- a stream starts with none of the window's steps: only the stream starts them.

Oracles: the scenarios' declared sizes and holds, the nominal feed.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from target_gym.stream import StreamEnv, make_stream

STREAM = make_stream("distillation")
NAMES = [s.name for s in STREAM.scenarios]
ENV = STREAM.env
STEP = jax.jit(ENV.step_env)
HOLD_MIN = 30  # the tests' hold, so a run stays short; the contract holds for any


def _params(name):
    return STREAM.scenario_params(STREAM.default_params, NAMES.index(name))


def _hold_action(params):
    """The raw action that holds the nominal reflux and boilup."""
    _, s = ENV.reset_env(jax.random.PRNGKey(0), params)
    span = lambda v, lo, hi: 2.0 * (v - lo) / (hi - lo) - 1.0  # noqa: E731
    return np.array(
        [
            span(s.L, params.L_min, params.L_max),
            span(s.V, params.V_min, params.V_max),
        ]
    )


def _feed(state, params, n):
    """The feed rate and composition each of ``n`` steps on ``params`` runs on,
    as deviations from nominal."""
    action = _hold_action(params)
    rate, comp = [], []
    for k in range(n):
        _, state, *_ = STEP(jax.random.PRNGKey(k), state, action, params)
        rate.append(float(state.F) / params.F - 1.0)
        comp.append(float(state.zF) - params.zF_nominal)
    return np.array(rate), np.array(comp), state


@pytest.mark.parametrize(
    "name, channel, size_field",
    [("feed_rate_step", 0, "feed_step"), ("feed_composition_step", 1, "zF_step")],
)
def test_a_feed_step_holds_through_a_switch_then_returns(name, channel, size_field):
    upset = _params(name).replace(event_hold=float(HOLD_MIN))
    normal = _params("normal")
    _, s = ENV.reset_env(jax.random.PRNGKey(0), normal)
    s = ENV.start_event(s, jax.random.PRNGKey(1), upset)
    runs = _feed(s, normal, HOLD_MIN + 10)
    moved, other = runs[channel], runs[1 - channel]
    size = getattr(upset, size_field)
    # Its scenario's size, either sign, from the first step after the onset.
    np.testing.assert_allclose(np.abs(moved[:HOLD_MIN]), size, rtol=1e-5)
    assert len(set(np.sign(moved[:HOLD_MIN]).tolist())) == 1
    np.testing.assert_allclose(moved[HOLD_MIN:], 0.0, atol=1e-7)
    np.testing.assert_allclose(other, 0.0, atol=1e-7)


def test_a_composition_step_does_not_end_a_feed_rate_step_in_progress():
    both = _params("both_steps").replace(event_hold=float(HOLD_MIN))
    comp = _params("feed_composition_step").replace(event_hold=float(HOLD_MIN))
    normal = _params("normal")
    _, s = ENV.reset_env(jax.random.PRNGKey(0), normal)
    s = ENV.start_event(s, jax.random.PRNGKey(1), both)
    rate, _, s = _feed(s, normal, 10)
    s = ENV.start_event(s, jax.random.PRNGKey(2), comp)
    later, _, _ = _feed(s, normal, HOLD_MIN - 10)
    np.testing.assert_allclose(later, rate[-1], rtol=1e-6)
    assert rate[-1] != 0.0


def test_a_stream_starts_without_the_window_s_steps():
    both = next(s for s in STREAM.scenarios if s.name == "both_steps")
    quiet = StreamEnv(
        ENV, STREAM.default_params, [dataclasses.replace(both, events_per_h=0.0)]
    )
    params = quiet.default_params
    window_end = int(params.step_window[1])  # the window's steps come before it
    action = _hold_action(params)

    def body(state, k):
        _, state, *_ = quiet.step(k, state, action, params)
        return state, (state.env.F, state.env.zF)

    _, state = quiet.reset(jax.random.PRNGKey(5), params)
    F, zF = jax.lax.scan(
        body, state, jax.random.split(jax.random.PRNGKey(6), window_end + 10)
    )[1]
    np.testing.assert_array_equal(np.asarray(F), params.F)
    np.testing.assert_array_equal(np.asarray(zF), params.zF_nominal)
