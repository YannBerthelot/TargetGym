"""The aircraft family as a continuing plant (long-strata draft 4, S1b and R8, with
the owner's calls of 2026-11-11).

Contract, for every aircraft task wired to the stream:
- in a stream a sortie's mass is held: the burn does not lower it, while a window
  still burns fuel off it (S1b, "mass is held during a sortie");
- a new sortie starts established in cruise, level, with an all-up mass drawn
  over the task's ``sortie_mass_range``; the weather it flies into (the
  turbulence zone) and every clock carry on (S1b, S2);
- sorties differ: each draws its own mass;
- a trip restarts the plant as the sortie's replacement (R8, owner): established
  at the sortie's level with the sortie's mass, which the outside world keeps.

Oracles: S1b's rules and the ranges each task declares. A sortie is started
through ``start_event``, the stream's hook, with ``event_mix`` set to sorties only.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.base import failure_kernel
from target_gym.stream import make_stream

AIRCRAFT = (
    "plane",
    "plane_energy",
    "plane_sine",
    "plane3d_heading",
    "plane3d_circle",
    "plane3d_racetrack",
    "plane3d_figure8",
    "formation",
    "patrol",
    "patrol_range_bearing",
    "wake_surf",
)
#: Tasks whose state is one aircraft with an assigned level.
SINGLE = AIRCRAFT[:7]
#: Only sorties start: each kind's share of the events (sortie, zone, task's own).
SORTIES = (1.0, 0.0, 0.0)


@lru_cache(maxsize=None)
def _task(name):
    stream = make_stream(name)
    return stream, jax.jit(stream.env.step_env)


def _stream_params(name):
    stream, _ = _task(name)
    return stream.scenario_params(stream.default_params, 0)


def _leaves(state, field):
    """Every aircraft's ``field`` in ``state``, nested aircraft included."""
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return np.concatenate(
        [
            np.ravel(np.asarray(v, np.float64))
            for p, v in flat
            if jax.tree_util.keystr(p).endswith(f".{field}")
        ]
    )


def _full_throttle(env, params):
    """Throttle up on every aircraft the action flies (each takes three values,
    throttle first; the 2D aircraft two, throttle first)."""
    shape = env.action_space(params).shape
    return jnp.zeros(shape).at[::3].set(1.0)


@pytest.mark.parametrize("name", AIRCRAFT)
def test_a_sortie_s_mass_is_held_in_a_stream_while_a_window_burns_it(name):
    stream, step = _task(name)
    env = stream.env
    after = {}
    for label, p in (
        ("stream", _stream_params(name)),
        ("window", stream.default_params),
    ):
        _, s = env.reset_env(jax.random.PRNGKey(0), p)
        start = _leaves(s, "m")
        for k in range(20):
            _, s, *_ = step(jax.random.PRNGKey(k), s, _full_throttle(env, p), p)
        after[label] = (start, _leaves(s, "m"))
    np.testing.assert_array_equal(after["stream"][1], after["stream"][0])
    assert (after["window"][1] < after["window"][0]).all()


@pytest.mark.parametrize("name", AIRCRAFT)
def test_a_new_sortie_starts_established_with_a_fresh_mass_in_the_weather_it_meets(
    name,
):
    stream, _ = _task(name)
    env = stream.env
    p = _stream_params(name).replace(event_mix=SORTIES)
    _, s = env.reset_env(jax.random.PRNGKey(1), p)
    # The clock has run, and the weather is mid-zone where the task has zones.
    s = s.replace(time=jnp.int32(5000))
    zoned = hasattr(s, "zone_sigma")
    if zoned:
        s = s.replace(
            zone_start=jnp.float32(4900.0),
            zone_end=jnp.float32(5500.0),
            zone_sigma=jnp.float32(1.5),
        )
    new = env.start_event(s, jax.random.PRNGKey(2), p)
    np.testing.assert_array_equal(_leaves(new, "time"), _leaves(s, "time"))
    if zoned:
        for f in ("zone_start", "zone_end", "zone_sigma"):
            assert float(getattr(new, f)) == float(getattr(s, f)), f
    lo, hi = p.sortie_mass_range
    m = _leaves(new, "m")
    assert ((m >= lo) & (m <= hi)).all(), (m, lo, hi)
    # Established: level flight, at the assigned level where there is one.
    np.testing.assert_allclose(_leaves(new, "z_dot"), 0.0, atol=1e-6)
    if name in SINGLE:
        np.testing.assert_allclose(float(new.z), float(new.target_altitude), atol=1e-2)


@pytest.mark.parametrize("name", AIRCRAFT)
def test_each_sortie_draws_its_own_mass(name):
    stream, _ = _task(name)
    env = stream.env
    p = _stream_params(name).replace(event_mix=SORTIES)
    _, s = env.reset_env(jax.random.PRNGKey(3), p)
    starts = jax.jit(jax.vmap(lambda k: env.start_event(s, k, p)))
    m = _leaves(starts(jax.random.split(jax.random.PRNGKey(4), 64)), "m")
    lo, hi = p.sortie_mass_range
    # 64 uniform draws or more: a spread under half the range has a chance
    # below 1e-15.
    assert m.max() - m.min() > 0.5 * (hi - lo)


@pytest.mark.parametrize("name", AIRCRAFT)
def test_a_trip_restarts_the_sortie_s_replacement_at_its_level_and_mass(name):
    stream, step = _task(name)
    env = stream.env
    p = _stream_params(name)
    _, s = env.reset_env(jax.random.PRNGKey(6), p)
    hold = jnp.zeros(env.action_space(p).shape)
    for k in range(5):
        _, s, *_ = step(jax.random.PRNGKey(k), s, hold, p)

    class Trips:
        is_terminated = staticmethod(lambda state, params: jnp.ones((), bool))
        reset_env = staticmethod(env.reset_env)
        OUTSIDE_WORLD = env.OUTSIDE_WORLD
        RECORDS = getattr(env, "RECORDS", ())
        restart_plant = staticmethod(env.restart_plant)

    proposal = s.replace(time=s.time + 1)
    out, tripped, *_ = failure_kernel(Trips(), jax.random.PRNGKey(9), s, proposal, p)
    assert bool(tripped)
    np.testing.assert_array_equal(_leaves(out, "m"), _leaves(proposal, "m"))
    # The plant restarts level; a patrol's scripted lead is the outside world.
    plant = out.follower if hasattr(out, "follower") else out
    np.testing.assert_allclose(_leaves(plant, "z_dot"), 0.0, atol=1e-6)
    if name in SINGLE:
        np.testing.assert_allclose(float(out.z), float(out.target_altitude), atol=1e-2)
    assert not bool(env.is_terminated(out, p))
