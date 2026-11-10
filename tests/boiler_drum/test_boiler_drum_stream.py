"""The drum boiler as a continuing plant (long-strata draft 4, S1).

Contract:
- a runback carries its own kind and size: started in an upset regime, it runs to
  its return through a switch to normal operation. The valve's load runs back to
  ``runback_load`` and the fans hold firing to it until ``event_hold`` after the
  onset; then the fans are back and the load ramps up at the plan's rate;
- in a load-following regime the event is a planned load change: announced ahead
  (``s_to_ramp`` from ``ramp_start_range``), it ramps from where the plan stands to
  a load in the regime's ``load_range`` at a rate in its ``ramp_rate_range``;
- the plan keeps exact time however long the plant has run;
- a stream starts on a flat plan, with neither the window's ramp nor its upset.

Oracles: the scenarios' declared ranges and hold, ``runback_load``, the step of 2 s.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.stream import make_stream

# Observation channels.
LOAD_NOW, PLANNED, S_TO_RAMP, FUEL_CAP = 7, 8, 9, 10
#: Steps every run here takes: the longest notice (900 s) and the slowest ramp
#: (1 %/min) over the widest move (75 %), 5400 s, at 2 s a step, with a margin.
N = 2800


@pytest.fixture(scope="module")
def boiler():
    """The stream's boiler, its scenarios' params as the stream runs them, and its
    reset, event hook and an ``N``-step run holding an action, compiled once."""
    stream = make_stream("boiler_drum")
    env = stream.env
    names = [s.name for s in stream.scenarios]

    def params(name):
        return stream.scenario_params(stream.default_params, names.index(name))

    @jax.jit
    def run(state, p, action):
        def body(s, k):
            obs, s, *_ = env.step_env(k, s, action, p)
            return s, obs

        keys = jax.random.split(jax.random.PRNGKey(9), N)
        return jax.lax.scan(body, state, keys)[1]

    return SimpleNamespace(
        env=env,
        params=params,
        reset=jax.jit(env.reset_env),
        start=jax.jit(env.start_event),
        run=lambda state, p: np.asarray(run(state, p, env.initial_action(state, p))),
    )


def test_a_runback_runs_through_a_switch_and_ramps_back_after_its_hold(boiler):
    normal = boiler.params("normal")
    fan = boiler.params("id_fan_runback").replace(event_hold=300.0)  # s
    _, s = boiler.reset(jax.random.PRNGKey(0), fan)  # a load in 75 to 100 %
    load0 = float(s.q_demand)
    s = boiler.start(s, jax.random.PRNGKey(1), fan)
    # Every step after the onset runs on the normal regime's params.
    obs = boiler.run(s, normal)[:400]
    runback = fan.runback_load * fan.q_steam_nominal
    hold = int(fan.event_hold / fan.delta_t)  # 150 steps
    # Back to the runback load within the hold, the fans holding firing.
    np.testing.assert_allclose(obs[hold - 10 : hold - 1, LOAD_NOW], runback, rtol=1e-6)
    assert (obs[: hold - 1, FUEL_CAP] < 100.0).all()
    # After the hold the fans are back and the load ramps up at the plan's rate.
    assert (obs[hold + 1 :, FUEL_CAP] == 100.0).all()
    climb = np.diff(obs[hold + 1 : hold + 20, LOAD_NOW]) / fan.delta_t
    np.testing.assert_allclose(climb, float(s.ramp_rate), rtol=1e-3)
    assert runback < obs[-1, LOAD_NOW] <= load0 + 1e-3


def test_a_planned_load_change_is_announced_then_ramps_inside_the_regime_s_ranges(
    boiler,
):
    low = boiler.params("low_load")
    _, s = boiler.reset(jax.random.PRNGKey(2), boiler.params("normal"))
    s = s.replace(time=jnp.int32(1000))
    from_load = float(s.q_demand)
    obs = boiler.run(boiler.start(s, jax.random.PRNGKey(3), low), low)
    nominal = low.q_steam_nominal
    notice = obs[0, S_TO_RAMP] + low.delta_t  # read one step after the onset
    assert low.ramp_start_range[0] <= notice <= low.ramp_start_range[1]
    target = obs[0, PLANNED] / nominal
    assert low.load_range[0] <= target <= low.load_range[1]
    wait = int(notice / low.delta_t)
    np.testing.assert_allclose(obs[: wait - 1, LOAD_NOW], from_load, rtol=1e-6)
    rate = np.abs(np.diff(obs[wait + 1 : wait + 10, LOAD_NOW])) / low.delta_t
    pct_per_min = rate / nominal * 100.0 * 60.0
    lo, hi = low.ramp_rate_range
    assert (pct_per_min >= lo * (1 - 1e-3)).all()
    assert (pct_per_min <= hi * (1 + 1e-3)).all()
    np.testing.assert_allclose(obs[-1, LOAD_NOW], target * nominal, rtol=1e-5)


def test_the_plan_keeps_exact_time_however_long_the_plant_has_run(boiler):
    p = boiler.params("normal")
    _, s = boiler.reset(jax.random.PRNGKey(4), p)
    loads = []
    for t0 in (10, 50_000_010):  # 100 million s later: float32 counts it in 8 s
        started = boiler.start(s.replace(time=jnp.int32(t0)), jax.random.PRNGKey(5), p)
        loads.append(boiler.run(started, p)[:, LOAD_NOW])
    np.testing.assert_array_equal(loads[0], loads[1])


def test_a_stream_starts_on_a_flat_plan_without_the_window_s_events(boiler):
    # The upset regime as the stream runs it: the window would ramp once in 5 to
    # 15 min and run the unit back.
    fan = boiler.params("id_fan_runback")
    _, s = boiler.reset(jax.random.PRNGKey(6), fan)
    obs = boiler.run(s, fan)
    np.testing.assert_array_equal(obs[:, LOAD_NOW], obs[0, LOAD_NOW])
    np.testing.assert_array_equal(obs[:, PLANNED], obs[0, LOAD_NOW])
    assert (obs[:, S_TO_RAMP] == 0.0).all() and (obs[:, FUEL_CAP] == 100.0).all()
