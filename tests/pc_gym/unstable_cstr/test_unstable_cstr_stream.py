"""The LDPE autoclave as a continuing plant (long-strata draft 4, S1).

Contract:
- an upset carries its own size: started in one scenario, it runs to its return
  through a switch to another, and lasts ``event_hold`` seconds from its onset.
  The feed's warming, the initiator's lost efficiency and the feed's rise return;
- a grade is a held level (owner, 2026-11-11; REALITY.md [ST4, ST5]): a grade
  change moves the zone setpoint to the other grade, 513 or 537 K [L1, p. 3], and
  it holds there through any switch of regime, however long the plant runs, until
  the next change;
- a grade change leaves a running upset to run to its own return;
- a stream starts on either grade with chance 1/2, the grades' long-run law, in
  established operation at it (S1);
- a trip keeps the grade, and the plant restarts in established operation at it;
- grades change about twice a month [ST5] and a transition takes 50 min to
  3 h 5 min [ST4]: the regime is the transition, so its share of time times its
  rate is 2 a month, its dwell lies in that range, and it brings one move a
  visit. The other upsets keep their shares (owner: only the grade's weight moves);
- the monomer feed's wander keeps its period however long the plant has run;
- a stream starts with no event of the window: only the stream starts events.

Oracles: the scenarios' declared sizes and holds, [L1]'s two zone setpoints,
[ST4]'s and [ST5]'s figures, a fair coin's binomial spread, and the wander's
period.
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

SPEC = REGISTRY["unstable_cstr"]
STREAM = make_stream("unstable_cstr")
NAMES = [s.name for s in STREAM.scenarios]
STEP = jax.jit(STREAM.env.step_env)
#: The grades' zone setpoints, K: the Politeno reactor's zones at 513 and 537 K
#: [L1, p. 3].
GRADES = (513.0, 537.0)
HOUR, DAY = 3600.0, 86400.0
#: A month's hours, a year's over twelve.
MONTH_H = 365.25 * 24.0 / 12.0


def _params(name):
    return STREAM.scenario_params(STREAM.default_params, NAMES.index(name))


def _at_rest(params, seed=0):
    return STREAM.env.reset_env(jax.random.PRNGKey(seed), params)[1]


def _setpoint_after_a_step(state, params):
    """The setpoint the plant shows after one held step, and the state then."""
    action = STREAM.env.initial_action(state, params)
    obs, state, *_ = STEP(jax.random.PRNGKey(0), state, action, params)
    return float(obs[2]), state


@pytest.mark.parametrize(
    "name", ["feed_temperature", "initiator_efficiency", "throughput_step"]
)
def test_an_upset_holds_its_size_through_a_switch_then_returns(name):
    moved, normal = _params(name), _params("normal")
    quiet = _at_rest(normal)
    started = STREAM.env.start_event(quiet, jax.random.PRNGKey(1), moved)
    hold = float(moved.event_hold)  # s; the onset is the reset's second, 0
    # The scenario has switched to normal: the event still runs on its declared
    # size, against the same second with no event. The feed warms, the initiator
    # loses a fraction of its radicals, the feed flow rises.
    for second in (1.0, hold - 1.0):
        sp, T_f, f, scale = events(started, normal, second)
        sp0, T_f0, f0, scale0 = events(quiet, normal, second)
        np.testing.assert_allclose(
            [sp - sp0, T_f - T_f0, f / f0, scale / scale0],
            [
                0.0,
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


def test_a_grade_change_moves_to_the_other_grade_and_holds_it():
    change, normal = _params("grade_change"), _params("normal")
    key = jax.random.PRNGKey(1)
    before, _ = _setpoint_after_a_step(_at_rest(normal), normal)
    moved = STREAM.env.start_event(_at_rest(normal), key, change)
    # The regime switches back to normal at once; the grade stays.
    after, state = _setpoint_after_a_step(moved, normal)
    assert sorted([before, after]) == pytest.approx(GRADES, abs=1e-4)
    # Past the old 8 h hold, and a month on: still the new grade.
    for later in (8.0 * HOUR + 1.0, 30.0 * DAY):
        then = state.replace(time=state.time + jnp.int32(later / normal.delta_t))
        assert _setpoint_after_a_step(then, normal)[0] == pytest.approx(after, abs=1e-4)
    # The next change moves it back.
    back = STREAM.env.start_event(state, key, change)
    assert _setpoint_after_a_step(back, normal)[0] == pytest.approx(before, abs=1e-4)


def test_a_grade_change_leaves_a_running_upset_to_its_return():
    warm, change = _params("feed_temperature"), _params("grade_change")
    normal = _params("normal")
    quiet = _at_rest(normal)
    key = jax.random.PRNGKey(1)
    both = STREAM.env.start_event(STREAM.env.start_event(quiet, key, warm), key, change)
    hold = float(warm.event_hold)  # s, from the reset's second, 0
    for second in (1.0, hold - 1.0, hold + 1.0):
        sp, T_f, *_ = events(both, normal, second)
        sp0, T_f0, *_ = events(quiet, normal, second)
        warming = warm.feed_T_step if second < hold else 0.0
        np.testing.assert_allclose(
            [abs(sp - sp0), T_f - T_f0], [24.0, warming], atol=1e-4
        )


def test_a_stream_starts_on_either_grade_half_the_time_at_rest_on_it():
    n = 400
    keys = jax.random.split(jax.random.PRNGKey(11), n)
    _, states = jax.vmap(STREAM.reset, (0, None))(keys, STREAM.default_params)
    setpoint = np.asarray(states.env.target_T)
    assert np.isin(np.round(setpoint, 3), GRADES).all()
    # A fair coin over 400 starts: standard deviation 0.025; 4 of them.
    assert abs(np.mean(setpoint > np.mean(GRADES)) - 0.5) <= 4 * np.sqrt(0.25 / n)
    np.testing.assert_allclose(np.asarray(states.env.T), setpoint, atol=1e-4)


def _restarts(state, params, steps=900):
    """Full initiator for ``steps`` steps: the plant's temperature and setpoint
    on every step that tripped, which a restart has already replaced."""

    def body(s, k):
        _, s, _, _, info = STREAM.env.step_env(k, s, jnp.ones(1), params)
        return s, (info["tripped"], s.T, s.target_T)

    keys = jax.random.split(jax.random.PRNGKey(3), steps)
    tripped, T, target = jax.jit(lambda s: jax.lax.scan(body, s, keys)[1])(state)
    tripped = np.asarray(tripped)
    assert tripped.any(), "full initiator should trip the reactor"
    return np.asarray(T)[tripped], np.asarray(target)[tripped]


def test_a_trip_keeps_the_grade_and_restarts_the_plant_on_it():
    change, normal = _params("grade_change"), _params("normal")
    state = _at_rest(normal)
    # Two changes visit both grades, whichever the start drew. Each restart
    # draws afresh, so every one of them is checked.
    for _ in range(2):
        state = STREAM.env.start_event(state, jax.random.PRNGKey(1), change)
        grade, state = _setpoint_after_a_step(state, normal)
        T, target = _restarts(state, normal)
        np.testing.assert_allclose(target, grade, atol=1e-4)
        np.testing.assert_allclose(T, grade, atol=1e-3)


def test_grades_change_twice_a_month_and_a_transition_takes_one_to_three_hours():
    shares = scenarios.weights(SPEC)
    grade = STREAM.scenarios[NAMES.index("grade_change")]
    # [ST5]: a specialty line averaged 2 transitions a month.
    assert shares["grade_change"] * grade.events_per_h * MONTH_H == pytest.approx(2.0)
    # [ST4]: 50 min (the best run) to 3 h 5 min (the latest run of another).
    assert 50.0 / 60.0 <= grade.dwell_h <= 3.0 + 5.0 / 60.0
    # The regime is the transition: one move a visit, on average.
    assert grade.events_per_h * grade.dwell_h == pytest.approx(1.0)
    # Only the grade's weight moved: the other upsets keep their shares.
    assert shares["feed_temperature"] == pytest.approx(0.1)
    assert shares["initiator_efficiency"] == pytest.approx(0.05)
    assert shares["throughput_step"] == pytest.approx(0.05)


def test_the_feed_s_wander_keeps_its_period_however_long_the_plant_has_run():
    p = _params("normal")
    s = _at_rest(p)
    action = STREAM.env.initial_action(s, p)
    key = jax.random.PRNGKey(2)
    period = int(round(p.feed_T_wander_period / p.delta_t))  # 600 steps
    # 1e8 s on, float32 counts seconds only to 8 s: a clock read there would
    # move the feed by up to 2 pi * 8 / 600 rad of its 1 K wander.
    late_time = 100 + period * (100_000_000 // period)
    early = STEP(key, s.replace(time=jnp.int32(100)), action, p)[1].T
    late = STEP(key, s.replace(time=jnp.int32(late_time)), action, p)[1].T
    # One second's response to the feed through a 31 s residence: float32's
    # spacing at 513 K is 6e-5 K.
    np.testing.assert_allclose(float(late), float(early), atol=1e-4)


def test_a_stream_starts_without_the_window_s_event():
    window = scenarios.params_for(
        SPEC, next(s for s in STREAM.scenarios if s.name == "grade_change")
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
    setpoints = np.asarray(
        jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(6), n))[1]
    )
    # The grade it started on, held: no window event moves it.
    assert np.isin(np.round(setpoints[0], 3), GRADES)
    np.testing.assert_allclose(setpoints, setpoints[0], atol=1e-4)
