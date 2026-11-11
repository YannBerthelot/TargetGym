"""The 2D aircraft as a continuing plant (long-strata draft 4, S1 and S1b).

Contract:
- a turbulence zone carries its intensity: started in one scenario, it keeps
  its gusts through a switch to still air until its end;
- a stream starts no zone of the window's own: only the stream starts zones;
- ``plane_energy``'s clearances are events: one moves the level by 1 000 or
  2 000 ft or the speed by 10 kt, the band trails it at 500 fpm [AIM 4-4-10],
  and without an event nothing is cleared, whatever the clock;
- ``plane_sine``'s sinusoid repeats exactly, however long the plant has run.

Oracles: the scenario's declared intensities, the AIM's 500 fpm, RVSM's
1 000 ft steps, the sinusoid's 240 s period.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.stream import StreamEnv, make_stream

FT, KT = 0.3048, 1852.0 / 3600.0


@lru_cache(maxsize=None)
def _task(name):
    stream = make_stream(name)
    return stream, jax.jit(stream.env.step_env)


def _scenario(stream, name):
    names = [s.name for s in stream.scenarios]
    return stream.scenario_params(stream.default_params, names.index(name))


def test_a_zone_keeps_its_intensity_through_a_switch_to_still_air():
    stream, step = _task("plane")
    env = stream.env
    worst, calm = _scenario(stream, "worst_zone"), _scenario(stream, "calm")
    _, s = env.reset_env(jax.random.PRNGKey(0), calm)
    zoned = env.start_event(
        s, jax.random.PRNGKey(1), worst.replace(event_mix=(0.0, 1.0, 0.0))
    )
    assert float(zoned.zone_sigma) == float(np.float32(worst.turbulence_sigma))
    length = float(zoned.zone_end - zoned.zone_start)
    lo, hi = worst.zone_duration_range
    assert lo <= length <= hi
    hold = env.trim_action(zoned, calm)
    gusts, still = [], []
    for k in range(60):
        # Every step after the onset runs on still air's params.
        _, zoned, *_ = step(jax.random.PRNGKey(k), zoned, hold, calm)
        _, s, *_ = step(jax.random.PRNGKey(k), s, hold, calm)
        gusts.append(float(jnp.hypot(zoned.gust_x, zoned.gust_z)))
        still.append(float(jnp.hypot(s.gust_x, s.gust_z)))
    assert max(still) == 0.0
    # The worst zone's 1.71 m/s: a minute of its gust is nowhere near zero.
    assert np.mean(gusts) > 0.3


def test_a_stream_starts_no_zone_of_the_window_s_own():
    stream = make_stream("plane")
    normal = next(s for s in stream.scenarios if s.name == "normal")
    quiet = StreamEnv(
        stream.env,
        stream.default_params,
        [dataclasses.replace(normal, events_per_h=0.0)],
    )
    params = quiet.default_params
    n = int(params.max_steps_in_episode)

    def body(state, k):
        obs, state, *_ = quiet.step(k, state, jnp.zeros(2), params)
        return state, jnp.hypot(state.env.gust_x, state.env.gust_z)

    _, state = quiet.reset(jax.random.PRNGKey(5), params)
    gusts = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(6), n))[1]
    # The window would put a light zone somewhere in these 30 min.
    assert float(jnp.max(gusts)) == 0.0


def test_a_level_clearance_moves_the_level_and_the_band_trails_it_at_500_fpm():
    stream, _ = _task("plane_energy")
    env = stream.env
    p = _scenario(stream, "normal").replace(
        event_mix=(0.0, 0.0, 1.0), clearance_level_share=1.0
    )
    _, s = env.reset_env(jax.random.PRNGKey(2), p)
    s = s.replace(time=jnp.int32(4000))
    cleared = env.start_event(s, jax.random.PRNGKey(3), p)
    moved = abs(float(cleared.target_altitude - s.target_altitude))
    assert round(moved / FT) in (1000, 2000)
    assert int(cleared.level_change_at) == 4000
    # Still at the old level: inside the allowance at once, and past it once
    # 500 fpm has had time to close the gap and a minute more.
    at_once = env.reward_terms(cleared, p)["tracking"]
    climb_s = moved / (500.0 * FT / 60.0)
    late = cleared.replace(time=jnp.int32(4000 + int(climb_s) + 60))
    assert float(at_once) == 0.0
    assert float(env.reward_terms(late, p)["tracking"]) > 0.0


def test_a_speed_clearance_moves_the_assigned_speed_by_10_kt():
    stream, _ = _task("plane_energy")
    env = stream.env
    p = _scenario(stream, "normal").replace(
        event_mix=(0.0, 0.0, 1.0), clearance_level_share=0.0
    )
    _, s = env.reset_env(jax.random.PRNGKey(2), p)
    cleared = env.start_event(s, jax.random.PRNGKey(3), p)
    np.testing.assert_allclose(
        abs(float(env.assigned_cas(cleared, p) - env.assigned_cas(s, p))),
        10.0 * KT,
        rtol=1e-5,
    )
    assert float(cleared.target_altitude) == float(s.target_altitude)


def test_without_an_event_a_stream_clears_nothing_however_long_it_runs():
    stream, step = _task("plane_energy")
    env = stream.env
    p = _scenario(stream, "normal")
    _, s = env.reset_env(jax.random.PRNGKey(4), p)
    level, cas = float(s.target_altitude), float(env.assigned_cas(s, p))
    hold = env.trim_action(s, p)
    # Past the window's first two treads (300 s each), where it would clear.
    for k in range(610):
        _, s, *_ = step(jax.random.PRNGKey(k), s, hold, p)
    assert float(s.target_altitude) == level
    assert float(env.assigned_cas(s, p)) == cas


def test_the_sinusoid_repeats_exactly_however_long_the_plant_has_run():
    stream, _ = _task("plane_sine")
    env = stream.env
    p = _scenario(stream, "normal")
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    period = int(round(p.target_period / p.delta_t))  # 240 steps
    early = env.commanded_level(s.replace(time=jnp.int32(100)), p)
    late = env.commanded_level(s.replace(time=jnp.int32(100 + 50_000 * period)), p)
    assert float(early) == float(late)
    expected = float(s.base_target_altitude) + p.target_amplitude * np.sin(
        2.0 * np.pi * 100 / period
    )
    np.testing.assert_allclose(float(early), expected, atol=0.01)
