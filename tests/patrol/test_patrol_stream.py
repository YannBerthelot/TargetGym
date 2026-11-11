"""The patrol, the wake surf and the formation as continuing plants (long-strata
draft 4, S1b).

Contract:
- the lead's route cycles: its legs repeat exactly, however long the plant has
  run (S1b: "patrol and formation's lead cycles its route");
- a new sortie puts the pair or the formation in cruise on a fresh route, every
  aircraft on its own clock as before, so the turbulence each meets runs on.

Oracles: the route's eight legs of ``lead_leg_seconds``; the clocks before the
sortie.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.patrol.env import N_LEAD_LEGS, lead_leg
from target_gym.stream import make_stream


@pytest.mark.parametrize("name", ["patrol", "wake_surf", "formation"])
def test_the_lead_s_route_repeats_exactly_however_long_the_plant_has_run(name):
    p = make_stream(name).default_params
    route = int(round(p.lead_leg_seconds / p.delta_t)) * N_LEAD_LEGS
    steps = jnp.arange(0, 2 * route, 7, dtype=jnp.int32)
    late = steps + 70_000 * route
    np.testing.assert_array_equal(lead_leg(steps, p), lead_leg(late, p))
    # Leg k is flown over [k, k + 1) legs' time from the start of a route.
    per_leg = route // N_LEAD_LEGS
    np.testing.assert_array_equal(
        lead_leg(steps, p), (np.asarray(steps) % route) // per_leg
    )


@pytest.mark.parametrize("name", ["patrol", "formation"])
def test_a_new_sortie_starts_on_a_fresh_route_on_the_same_clocks(name):
    stream = make_stream(name)
    env, p = stream.env, stream.scenario_params(stream.default_params, 0)
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(time=jnp.int32(9000))
    s = s.replace(lead=s.lead.replace(time=jnp.int32(9000)))
    new = env.start_event(s, jax.random.PRNGKey(1), p)
    assert int(new.time) == 9000 and int(new.lead.time) == 9000
    assert not np.array_equal(
        np.asarray(new.lead_turn_schedule), np.asarray(s.lead_turn_schedule)
    )
    np.testing.assert_array_equal(
        np.asarray(new.gust_history), np.asarray(s.gust_history)
    )
