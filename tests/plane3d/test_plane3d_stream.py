"""The 3D aircraft as a continuing plant (long-strata draft 4, S1, S1b and R8).

Contract:
- the heading task's vectors are events: one draws a heading over
  ``target_heading_range``, its size the turn from the heading flown, and
  without an event no vector comes, whatever the clock;
- after a switch the steady wind moves to the new scenario's as a first-order
  lag of ``wind_tau``, never in one step;
- a trip restarts the sortie's replacement on the sortie's own pattern: on the
  circle, the hold or the figure eight it flies, or on its cleared heading,
  even where the wind has changed since the sortie began.

Oracles: the first-order lag's closed form; the pattern's geometry, carried in
the outside world; the declared heading range.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.base import failure_kernel
from target_gym.plane3d.env import (
    distance_to_circle,
    distance_to_racetrack,
    heading_3d,
    nearest_point_on_twisted_lemniscate,
    wrap_angle,
)
from target_gym.stream import make_stream


@lru_cache(maxsize=None)
def _task(name):
    stream = make_stream(name)
    return stream, jax.jit(stream.env.step_env)


def _scenario(stream, name):
    names = [s.name for s in stream.scenarios]
    return stream.scenario_params(stream.default_params, names.index(name))


def test_a_vector_is_an_event_sized_by_the_turn_from_the_heading_flown():
    stream, _ = _task("plane3d_heading")
    env = stream.env
    p = _scenario(stream, "calm").replace(event_mix=(0.0, 0.0, 1.0))
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(time=jnp.int32(3000))
    vectored = env.start_event(s, jax.random.PRNGKey(1), p)
    lo, hi = p.target_heading_range
    assert lo <= float(vectored.target_heading) <= hi
    turn = abs(float(wrap_angle(vectored.target_heading - heading_3d(s, p))))
    np.testing.assert_allclose(float(vectored.heading_change), turn, atol=1e-5)
    assert float(vectored.clearance_time) == 3000.0


def test_without_an_event_no_vector_comes_however_long_it_runs():
    stream, step = _task("plane3d_heading")
    env = stream.env
    p = _scenario(stream, "calm")
    _, s = env.reset_env(jax.random.PRNGKey(2), p)
    heading = float(s.target_heading)
    hold = env.trim_action(s, p)
    # Past the window's 150 s vectors, twice.
    for k in range(310):
        _, s, *_ = step(jax.random.PRNGKey(k), s, hold, p)
    assert float(s.target_heading) == heading


def test_after_a_switch_the_wind_moves_to_the_new_one_as_a_first_order_lag():
    stream, step = _task("plane3d_heading")
    env = stream.env
    calm, windy = _scenario(stream, "calm"), _scenario(stream, "strong_wind")
    _, s = env.reset_env(jax.random.PRNGKey(3), calm)
    assert float(s.mean_wind_x) == 0.0
    hold = env.trim_action(s, calm)
    n = 120
    for k in range(n):
        _, s, *_ = step(jax.random.PRNGKey(k), s, hold, windy)
    # The gap closes by exp(-dt / tau) a step: after n steps, exp(-n dt / tau).
    left = np.exp(-n * windy.delta_t / windy.wind_tau)
    np.testing.assert_allclose(
        float(s.mean_wind_x), windy.wind_x * (1.0 - left), rtol=1e-4
    )
    assert float(s.mean_wind_x) < 0.1 * windy.wind_x


def _restarted(name, scenario, then, steps=5):
    """The task flown ``steps`` on ``scenario``, then tripped on ``then``."""
    stream, step = _task(name)
    env = stream.env
    p, q = _scenario(stream, scenario), _scenario(stream, then)
    _, s = env.reset_env(jax.random.PRNGKey(6), p)
    hold = env.trim_action(s, p)
    for k in range(steps):
        _, s, *_ = step(jax.random.PRNGKey(k), s, hold, p)

    class Trips:
        is_terminated = staticmethod(lambda state, params: jnp.ones((), bool))
        reset_env = staticmethod(env.reset_env)
        OUTSIDE_WORLD = env.OUTSIDE_WORLD
        RECORDS = env.RECORDS
        restart_plant = staticmethod(env.restart_plant)

    proposal = s.replace(time=s.time + 1)
    out, tripped, *_ = failure_kernel(Trips(), jax.random.PRNGKey(9), s, proposal, q)
    assert bool(tripped)
    return out, q


@pytest.mark.parametrize(
    "name, scenario, then",
    [
        ("plane3d_circle", "normal", "normal"),
        # A hold sized for the calm it began in, restarted in a 50 kt wind.
        ("plane3d_racetrack", "calm", "low_hold_wind"),
    ],
)
def test_a_restart_puts_the_replacement_on_the_sortie_s_own_pattern(
    name, scenario, then
):
    out, _ = _restarted(name, scenario, then)
    distance = distance_to_circle if name == "plane3d_circle" else distance_to_racetrack
    assert abs(float(distance(out))) < 1.0
    np.testing.assert_allclose(float(out.z), float(out.target_altitude), atol=1e-2)


def test_a_restart_puts_the_replacement_on_the_figure_eight_it_flies():
    out, q = _restarted("plane3d_figure8", "normal", "normal")
    # The distance to the curve, in 3D (its fourth value).
    assert float(nearest_point_on_twisted_lemniscate(out, q)[3]) < 1.0


def test_a_restart_flies_the_replacement_on_the_cleared_heading():
    out, q = _restarted("plane3d_heading", "calm", "calm")
    error = abs(float(wrap_angle(heading_3d(out, q) - out.target_heading)))
    assert error < 1e-3
