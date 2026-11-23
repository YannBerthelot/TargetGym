"""``wake_surf``: an A320 surfing an A320's wake (patrol/REALITY.md, patrol/PHYSICS.md).

Contract, from the plant record and the owner's decision of 2026-11-23:

- The station is on the vortex plane: below the lead by the pair's sink over the
  time in trail, ``w0 t`` with ``w0 = Gamma / (2 pi b0)``, ``Gamma = W / (rho V b0)``
  and ``b0 = pi b / 4`` [Holzapfel and Steen 2007; Blake and Multhopp]. Two line
  vortices lift most on their own plane, so the upwash falls on either side of it.
- It is 1.2 to 1.3 spans from the lead's centreline, inside the 0.94 to 1.8 spans the
  extended-formation trials flew [REALITY.md: N2, N1 figure 10].
- Held there, the follower needs 3.5 to 10 % less fuel flow, what those trials
  sustained [REALITY.md: N1 II, V.A]. Upwash ``w`` rotates the lift forward by
  ``w / V``, so the saving is ``W w / V`` of the drag.
- Across the wake the station box is 30 m (ours, a span): free inside, then the
  squared excess in box widths. Along track the tolerance is the +-250 ft the
  Gulfstream's pilot was given [REALITY.md: N1 IV], priced the same way.
- The follower starts in established operation, within 10 m of its station
  (charter B5: no capture).
- A bank past 60 deg is an upset, a trip [EASA AMC 25.1329, A4.2: Hazardous]; the
  patrol, which does not set ``upset_bank``, does not trip on it.
- Both seats are A320s: the lead flies the airliner's schedule, 290 KCAS then
  M 0.78, at the A320's 73.5 t.
- The reference surfs (large): on held-out seeds it never trips, stays inside the box
  and the along-track tolerance past the scored burn-in, and burns 3.5 to 10 % less
  fuel than on the same seeds with the wake switched off.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.eval import scored_burn_in
from target_gym.patrol.env import (
    check_is_terminal_patrol,
    desired_slot_position,
    slot_delay,
    slot_error,
    slot_error_vector,
    wake_air,
)
from target_gym.plane.dynamics import (
    compute_air_density_from_altitude,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    scheduled_airspeed,
    trim_level,
)

KEY = jax.random.PRNGKey(5)
KT = 1852.0 / 3600.0
SEEDS = (5, 6, 7)
HELD_OUT = tuple(range(200, 205))
SPACING_M = 250.0 * 0.3048


def _reset(name="wake_surf", seed=5, **changes):
    env, params = tg.make(name, sensor_noise_scale=0.0, turbulence_sigma=0.0)
    params = params.replace(**changes)
    _, state = env.reset_env(jax.random.PRNGKey(seed), params)
    return env, params, state


def _at_station(state, back=0.0, right=0.0, up=0.0):
    """``state`` with the follower at its station plus an offset in the lead's
    frame: behind it, to its right, above."""
    x, y, z = desired_slot_position(state)
    psi = state.lead.psi
    follower = state.follower.replace(
        x=x - back * jnp.cos(psi) + right * jnp.sin(psi),
        y=y - back * jnp.sin(psi) - right * jnp.cos(psi),
        z=z + up,
    )
    return state.replace(follower=follower)


def _lead_speed(state):
    return float(jnp.hypot(state.lead.x_dot, state.lead.y_dot))


def _upwash(state, params, up=0.0):
    """The wake's upwash on a follower ``up`` metres above its station, m/s."""
    follower = _at_station(state, up=up).follower
    gusts, _ = wake_air(
        state.wake_history,
        jnp.atleast_1d(follower.x),
        jnp.atleast_1d(follower.y),
        jnp.atleast_1d(follower.z),
        slot_delay(jnp.atleast_1d(state.slot_back), _lead_speed(state), params),
        params,
    )
    return float(gusts[0, 2])


@pytest.mark.parametrize("seed", SEEDS)
def test_the_station_lies_on_the_vortex_plane(seed):
    _, params, state = _reset(seed=seed)
    lead, speed = state.lead, _lead_speed(state)
    b0 = np.pi * float(params.wingspan) / 4.0
    rho = float(compute_air_density_from_altitude(lead.z))
    circulation = float(lead.m) * float(params.gravity) / (rho * speed * b0)
    sink = circulation / (2.0 * np.pi * b0) * float(state.slot_back) / speed
    assert float(state.slot_up) == pytest.approx(-sink, abs=0.05)
    on_plane = _upwash(state, params)
    assert on_plane > _upwash(state, params, up=4.0)
    assert on_plane > _upwash(state, params, up=-4.0)


@pytest.mark.parametrize("seed", range(10))
def test_the_station_is_where_the_trials_flew(seed):
    _, params, state = _reset(seed=seed)
    spans = abs(float(state.slot_right)) / float(params.wingspan)
    assert 1.2 <= spans <= 1.3


@pytest.mark.parametrize("seed", SEEDS)
def test_the_station_saves_what_extended_formations_sustained(seed):
    _, params, state = _reset(seed=seed)
    lead, speed = state.lead, _lead_speed(state)
    rho = float(compute_air_density_from_altitude(lead.z))
    mach = speed / float(compute_speed_of_sound_from_altitude(lead.z))
    _, _, power = trim_level(speed, lead.z, params.initial_mass, params)
    drag = float(
        compute_thrust_output(power, params.thrust_output_at_sea_level, mach, rho)
    )
    weight = float(params.initial_mass) * float(params.gravity)
    saving = weight * _upwash(state, params) / speed / drag
    assert 0.035 <= saving <= 0.10, f"{100 * saving:.1f} % of fuel flow"


@pytest.mark.parametrize(
    "back, right, up, cost",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 20.0, 0.0),
        (0.0, 0.0, 60.0, 1.0),
        (0.0, 60.0, 0.0, 1.0),
        (60.0, 0.0, 0.0, 0.0),
        (-60.0, 0.0, 0.0, 0.0),
        (2 * SPACING_M, 0.0, 0.0, 1.0),
        (-2 * SPACING_M, 0.0, 0.0, 1.0),
    ],
)
def test_the_box_is_30_m_across_the_wake_and_250_ft_along_it(back, right, up, cost):
    env, params, state = _reset()
    state = _at_station(state, back, right, up)
    tracking = float(env.reward_terms(state, params)["tracking"])
    assert tracking == pytest.approx(cost, abs=2e-3)


@pytest.mark.parametrize("seed", range(20))
def test_the_follower_starts_established_on_its_station(seed):
    _, _, state = _reset(seed=seed)
    assert float(slot_error(state)) < 10.0


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


# ─── The reference, flown on the scored configuration ───


@functools.cache
def _flights(wake_strength):
    """The shipped expert on the held-out seeds: per seed and step the
    along-track error, the distance across the wake and the trips, and the
    fuel each seed burned."""
    env, params = tg.make("wake_surf")
    params = params.replace(wake_strength=wake_strength)
    expert = tg.make_expert("wake_surf")

    def one(seed):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        fuel = state.follower.fuel

        def step(carry, i):
            obs, state, memory = carry
            action, memory = expert.step(memory, obs)
            obs, state, _, _, info = env.step_env(
                jax.random.fold_in(key, i), state, action, params
            )
            back, right, up = slot_error_vector(state)
            return (obs, state, memory), (back, jnp.hypot(right, up), info["tripped"])

        steps = jnp.arange(int(params.max_steps_in_episode))
        (_, last, _), out = jax.lax.scan(step, (obs, state, expert.reset()), steps)
        return out, fuel - last.follower.fuel

    (back, across, tripped), burned = jax.jit(jax.vmap(one))(jnp.array(HELD_OUT))
    burn_in = scored_burn_in("wake_surf", params)
    held = np.asarray(back)[:, burn_in:], np.asarray(across)[:, burn_in:]
    return *held, np.asarray(tripped), np.asarray(burned)


@pytest.mark.large
def test_the_reference_surfs_inside_its_box_and_saves_fuel():
    back, across, tripped, burned = _flights(1.0)
    *_, burned_alone = _flights(0.0)
    assert not tripped.any()
    assert across.max() <= 30.0, f"{across.max():.1f} m across the wake"
    assert np.abs(back).max() <= SPACING_M, f"{np.abs(back).max():.1f} m along it"
    saving = 1.0 - burned / burned_alone
    assert (saving >= 0.035).all() and (saving <= 0.10).all(), saving
