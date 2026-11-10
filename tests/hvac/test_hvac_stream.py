"""The slab building as a continuing plant (long-strata draft 4, S1).

Contract:
- the day keeps running exactly, however long the plant has run: the outdoor
  cycle, the sun, the office's hours and the tariff read the same at a step and a
  whole number of days later;
- the building's scenarios are regimes of weather and gains with no event of their
  own, so the stream's event hook leaves the building as it is.

Oracles: the day's period in steps (24 h / 15 min = 96), and the regimes' rows in
``scenarios.py``, which carry no event.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.stream import make_stream

#: Steps in a day: 24 h at 15 min.
DAY = 96


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
