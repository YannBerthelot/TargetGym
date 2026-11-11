"""``wake_surf``: an A320 surfing an A320's wake (plane3d/PHYSICS.md, wake surfing).

Contract:

- The station is in the lead's upwash: behind a level lead, the wake lifts the
  follower there (NASA AFF's F/A-18s measured the induced drag falling most near a
  13 % tip overlap, the station's 0.87 span).
- Inside the 30 m station box the box costs nothing; past it, the squared excess
  in box widths: 60 m off is one width past, 1 a second.
- A bank past 60 deg is an upset, a trip [EASA AMC 25.1329, A4.2: Hazardous]; the
  patrol, which does not set ``upset_bank``, does not trip on it.
- Both seats are A320s: the lead flies the airliner's schedule, 290 KCAS then
  M 0.78, at the A320's 73.5 t.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.patrol.env import (
    check_is_terminal_patrol,
    desired_slot_position,
    slot_delay,
    wake_air,
)
from target_gym.plane.dynamics import scheduled_airspeed

KEY = jax.random.PRNGKey(5)
KT = 1852.0 / 3600.0


def _reset(name="wake_surf", **changes):
    env, params = tg.make(name, sensor_noise_scale=0.0, turbulence_sigma=0.0)
    params = params.replace(**changes)
    _, state = env.reset_env(KEY, params)
    return env, params, state


def _at_station(state, offset=(0.0, 0.0, 0.0)):
    x, y, z = desired_slot_position(state)
    follower = state.follower.replace(x=x + offset[0], y=y + offset[1], z=z + offset[2])
    return state.replace(follower=follower)


def test_the_station_is_in_the_leads_upwash():
    _, params, state = _reset()
    state = _at_station(state)
    speed = float(jnp.hypot(state.lead.x_dot, state.lead.y_dot))
    gusts, _ = wake_air(
        state.wake_history,
        jnp.atleast_1d(state.follower.x),
        jnp.atleast_1d(state.follower.y),
        jnp.atleast_1d(state.follower.z),
        slot_delay(jnp.atleast_1d(state.slot_back), speed, params),
        params,
    )
    assert float(gusts[0, 2]) > 1.0, f"upwash {float(gusts[0, 2]):.2f} m/s"


@pytest.mark.parametrize("off, cost", [(0.0, 0.0), (20.0, 0.0), (60.0, 1.0)])
def test_the_station_box_charges_past_30_m(off, cost):
    env, params, state = _reset()
    state = _at_station(state, (0.0, 0.0, off))
    tracking = float(env.reward_terms(state, params)["tracking"])
    assert tracking == pytest.approx(cost, abs=1e-3)


@pytest.mark.parametrize("name, trips", [("wake_surf", True), ("patrol", False)])
def test_an_upset_past_60_deg_trips_the_wake_surfer_only(name, trips):
    _, params, state = _reset(name)
    rolled = state.replace(follower=state.follower.replace(phi=jnp.deg2rad(61.0)))
    tripped, _ = check_is_terminal_patrol(rolled, params)
    assert bool(tripped) is trips


def test_both_seats_are_a320s_on_the_airliner_schedule():
    _, params, state = _reset()
    lead = state.lead
    speed = float(jnp.hypot(lead.x_dot, lead.y_dot))
    expected = float(scheduled_airspeed(lead.z, 290.0 * KT, 0.78))
    assert speed == pytest.approx(expected, rel=1e-3)
    assert float(params.initial_mass) == pytest.approx(73_500.0)
    assert float(params.wingspan) == pytest.approx(35.8)
    assert np.isclose(float(lead.m), float(state.follower.m))
