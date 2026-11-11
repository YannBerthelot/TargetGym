"""The slab building as a continuing plant (long-strata draft 4, S1).

Contract:
- the day keeps running exactly, however long the plant has run: the outdoor
  cycle, the sun, the office's hours and the tariff read the same at a step and a
  whole number of days later;
- the building's scenarios are regimes of weather and gains with no event of their
  own, so the stream's event hook leaves the building as it is;
- at a switch the outdoor mean does not jump: it moves to the new regime's as a
  first-order lag of 6 h (owner, 2026-11-11; ours), and the 24 h forecast knows
  where it is heading.

Oracles: the day's period in steps (24 h / 15 min = 96), the regimes' rows in
``scenarios.py``, which carry no event, and a first-order lag's closed form.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.stream import make_stream

#: Steps in a day: 24 h at 15 min.
DAY = 96
#: Observation channel: the next 24 h's mean outdoor temperature.
FORECAST = 9


def test_the_day_repeats_exactly_however_long_the_plant_has_run():
    stream = make_stream("hvac")
    # No weather noise, so the step at either time meets the same weather.
    p = stream.default_params.replace(T_out_noise_std=0.0)
    env, step = stream.env, jax.jit(stream.env.step_env)
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    action = env.initial_action(s, p)
    key = jax.random.PRNGKey(1)
    # 41 steps in (10:15, the punta tariff), and 200 000 days later.
    early = step(key, s.replace(time=jnp.int32(41)), action, p)
    late = step(key, s.replace(time=jnp.int32(41 + 200_000 * DAY)), action, p)
    for field in ("T_out", "T_air", "T_mass", "T_core"):
        assert float(getattr(early[1], field)) == float(getattr(late[1], field)), field
    # The clock and the tariff channels (sin_h, cos_h, price, hours to a change).
    clock = [6, 7, 10, 11]
    np.testing.assert_array_equal(
        np.asarray(early[0])[clock], np.asarray(late[0])[clock]
    )
    assert float(early[2]) == float(late[2])  # the step's cost, at the same tariff


def test_the_stream_s_event_hook_leaves_the_building_as_it_is():
    stream = make_stream("hvac")
    env, p = stream.env, stream.default_params
    _, s = env.reset_env(jax.random.PRNGKey(2), p)
    for k in range(len(stream.scenarios)):
        started = env.start_event(
            s, jax.random.PRNGKey(3), stream.scenario_params(p, k)
        )
        jax.tree_util.tree_map(np.testing.assert_array_equal, started, s)


#: The outdoor mean's time constant at a switch: 6 h, 24 steps of 15 min.
TAU_STEPS = 24


def test_a_switch_moves_the_outdoor_mean_over_six_hours_not_at_once():
    """A regime 10 K colder: after k steps the mean has covered 1 - exp(-k / 24) of
    the gap. With no weather noise, two runs on the same day differ by that only."""
    stream = make_stream("hvac")
    env, step = stream.env, jax.jit(stream.env.step_env)
    warm = stream.default_params.replace(T_out_noise_std=0.0, T_out_mean=5.0)
    cold = warm.replace(T_out_mean=-5.0)
    _, s = env.reset_env(jax.random.PRNGKey(0), warm)
    action = env.initial_action(s, warm)
    held = moved = s
    gaps, forecast_gaps = [], []
    for k in range(TAU_STEPS):
        key = jax.random.PRNGKey(k)
        obs_held, held, *_ = step(key, held, action, warm)
        obs_moved, moved, *_ = step(key, moved, action, cold)
        gaps.append(float(moved.T_out - held.T_out))
        forecast_gaps.append(float(obs_moved[FORECAST] - obs_held[FORECAST]))
    k = np.arange(1, TAU_STEPS + 1)
    np.testing.assert_allclose(gaps, -10.0 * (1.0 - np.exp(-k / TAU_STEPS)), atol=1e-4)
    # The forecast is the next 24 h's mean: the lag's gap over steps k+1 .. k+96.
    ahead = k[:, None] + np.arange(1, DAY + 1)[None, :]
    expected = np.mean(-10.0 * (1.0 - np.exp(-ahead / TAU_STEPS)), axis=1)
    np.testing.assert_allclose(forecast_gaps, expected, atol=1e-3)
