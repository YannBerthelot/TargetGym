"""The glass furnace as a continuing plant (long-strata draft 4, S1).

Contract, from the design and the scenario suite, not from the code:
- a planned pull change is announced when the stream starts it: the plan the
  operators see ``pull_notice_h`` ahead starts moving at once, the charged pull
  moves ``pull_notice_h`` later over ``pull_ramp_h``, holds ``event_hold_h`` from
  its start, then goes back over the same ramp. It carries the size it started
  with, so it runs to its return on another scenario's params;
- an unannounced gas swing moves the gas's heating value from its onset over
  ``gas_swing_h``, holds ``event_hold_h``, and comes back over the same ramp, with
  the size it started with;
- one change of each kind at a time: a pull change drawn while one is under way
  is absorbed, and one drawn after the last returned starts anew;
- the plan after an onset is the same however long the plant has run;
- a stream starts with no event of the window: only the stream starts events.

Oracles: the scenario's declared sizes (10 % of pull, 6 % of heating value), the
stream's hold, and the task's notice and ramps; the walks are off, so the plan
and the charged pull are exactly the declared shapes.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.glass_furnace import env_jax as E
from target_gym.stream import StreamEnv, make_stream

PLAN = E.OBS_NAMES.index("pull planned, kg/s")
CHARGED = E.OBS_NAMES.index("pull, kg/s")
ONSET = 500  # step of the onset, well inside any window


@pytest.fixture(scope="module")
def stream():
    return make_stream("glass_furnace")


def _quiet(stream, name):
    names = [s.name for s in stream.scenarios]
    p = stream.scenario_params(stream.default_params, names.index(name))
    return p.replace(walk_sd=(0.0,) * len(E.WALKS), sensor_noise_scale=0.0)


def _started(stream, name, onset=ONSET):
    """A normal reset with the scenario's event started by the stream at ``onset``."""
    env, normal = stream.env, _quiet(stream, "normal")
    _, s = env.reset_env(jax.random.PRNGKey(0), normal)
    s = s.replace(walk=jnp.zeros_like(s.walk), time=jnp.int32(onset))
    return env.start_event(s, jax.random.PRNGKey(1), _quiet(stream, name)), normal


def _at(state, onset, hours, p):
    return state.replace(time=jnp.int32(onset + round(hours * 3600.0 / p.delta_t)))


def test_a_pull_change_is_announced_ramps_holds_and_returns_on_another_scenario(
    stream,
):
    s, normal = _started(stream, "pull_up")
    up = _quiet(stream, "pull_up")
    pull, size = normal.pull, 0.10  # the suite's 10 % rise
    notice, ramp, hold = normal.pull_notice_h, normal.pull_ramp_h, up.event_hold_h
    assert hold > notice + 2 * ramp  # the stream's hold outlasts both ramps

    def plan(h):  # what the operators see, h hours after the onset
        return float(stream.env.get_obs(_at(s, ONSET, h, normal), normal)[PLAN])

    # The plan looks notice ahead: it moves from the onset, over the ramp.
    assert plan(0.0) == pytest.approx(pull, rel=1e-6)
    assert plan(ramp / 2) == pytest.approx(pull * (1 + size / 2), rel=1e-5)
    assert plan(ramp) == pytest.approx(pull * (1 + size), rel=1e-6)
    assert plan(hold - 0.5) == pytest.approx(pull * (1 + size), rel=1e-6)
    assert plan(hold + ramp / 2) == pytest.approx(pull * (1 + size / 2), rel=1e-5)
    assert plan(hold + ramp) == pytest.approx(pull, rel=1e-6)

    step = jax.jit(stream.env.step_env)
    action = stream.env.initial_action(s, normal)

    def charged(h):  # the pull charged on the step ending h hours after the onset
        end = _at(s, ONSET, h, normal)
        obs = step(
            jax.random.PRNGKey(2), end.replace(time=end.time - 1), action, normal
        )
        return float(obs[0][CHARGED])

    assert charged(notice - 0.5) == pytest.approx(pull, rel=1e-6)
    assert charged(notice + ramp) == pytest.approx(pull * (1 + size), rel=1e-6)
    assert charged(notice + hold + ramp) == pytest.approx(pull, rel=1e-6)


def test_a_gas_swing_starts_at_once_holds_and_returns_on_another_scenario(stream):
    s, normal = _started(stream, "gas_lean")
    lean = _quiet(stream, "gas_lean")
    size, ramp, hold = -0.06, normal.gas_swing_h, lean.event_hold_h
    assert hold > ramp

    def heating(h):  # the fuel bill against the same valve an hour before the onset
        def bill(hours):
            st = _at(s, ONSET, hours, normal)
            return float(stream.env.reward_terms(st, normal)["running"])

        return bill(h) / bill(-1.0)

    assert heating(0.0) == pytest.approx(1.0, rel=1e-6)
    assert heating(ramp / 2) == pytest.approx(1 + size / 2, rel=1e-5)
    assert heating(ramp) == pytest.approx(1 + size, rel=1e-6)
    assert heating(hold) == pytest.approx(1 + size, rel=1e-6)
    assert heating(hold + ramp) == pytest.approx(1.0, rel=1e-6)


def test_one_pull_change_at_a_time_and_a_new_one_after_the_return(stream):
    s, normal = _started(stream, "pull_down")
    down = _quiet(stream, "pull_down")
    hold, ramp, notice = down.event_hold_h, normal.pull_ramp_h, normal.pull_notice_h

    def plan(state, h):
        return float(stream.env.get_obs(_at(state, ONSET, h, normal), normal)[PLAN])

    # Drawn mid-hold: absorbed, the change under way keeps its timing.
    mid = _at(s, ONSET, hold / 2, normal)
    again = stream.env.start_event(mid, jax.random.PRNGKey(3), down)
    for h in (hold / 2 + 1.0, hold + ramp / 2, hold + ramp + 1.0):
        assert plan(again, h) == pytest.approx(plan(s, h), rel=1e-6)
    # Drawn after the charged pull is back, notice + hold + ramp after the onset:
    # a new change, announced at once.
    after = _at(s, ONSET, notice + hold + ramp + 1.0, normal)
    anew = stream.env.start_event(after, jax.random.PRNGKey(4), down)
    later = int(after.time) - ONSET
    h = later * normal.delta_t / 3600.0 + ramp
    assert plan(anew, h) == pytest.approx(normal.pull * 0.9, rel=1e-6)


def test_the_plan_after_an_onset_is_the_same_however_long_the_plant_has_run(stream):
    early, normal = _started(stream, "pull_up", onset=10)
    late, _ = _started(stream, "pull_up", onset=10 + 2**23)
    for h in (0.37, 1.0, 1.9):
        a = stream.env.get_obs(_at(early, 10, h, normal), normal)[PLAN]
        b = stream.env.get_obs(_at(late, 10 + 2**23, h, normal), normal)[PLAN]
        assert float(a) == float(b)


def test_a_stream_starts_without_the_window_s_events(stream):
    quiet = StreamEnv(
        stream.env,
        stream.default_params,
        [
            dataclasses.replace(s, events_per_h=0.0)
            for s in stream.scenarios
            if s.name in ("pull_up", "gas_rich")
        ],
    )
    for seed in range(4):
        _, state = quiet.reset(jax.random.PRNGKey(seed), quiet.default_params)
        p = quiet.scenario_params(quiet.default_params, state.scenario)
        # The window starts its event between 24 and 96 h; a stream never does.
        for h in (30.0, 100.0, 200.0):
            st = _at(state.env, 0, h, p).replace(walk=jnp.zeros_like(state.env.walk))
            assert float(stream.env.get_obs(st, p)[PLAN]) == pytest.approx(
                p.pull, rel=1e-6
            )
            terms = stream.env.reward_terms(st, p)["running"]
            flat = stream.env.reward_terms(st, p.replace(gas_swing=0.0))["running"]
            np.testing.assert_allclose(float(terms), float(flat), rtol=1e-6)
