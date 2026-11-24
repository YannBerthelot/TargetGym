"""``wake_surf``: an A320 surfing an A320's wake (patrol/REALITY.md, patrol/PHYSICS.md).

Contract, from the plant record and the owner's decisions of 2026-11-23:

- The station is on the vortex plane: below the lead by the pair's sink over the
  time in trail, ``w0 t`` with ``w0 = Gamma / (2 pi b0)``, ``Gamma = W / (rho V b0)``
  and ``b0 = pi b / 4`` [Holzapfel and Steen 2007; Blake and Multhopp]. Two line
  vortices lift most on their own plane, so the upwash falls on either side of it.
- It is 1.38 spans from the wake's centreline, inside the 0.94 to 1.8 spans the
  extended-formation trials flew [REALITY.md: N2, N1 figure 10], and 1.2 to 1.8 km
  in trail: their 4,000 ft at the near end, the 50 spans NASA calls the larger end
  of the usable range at the far one [REALITY.md: N1 III.D, N3 I].
- Held there, the follower needs 3.5 to 10 % less fuel flow, what those trials
  sustained [REALITY.md: N1 II, V.A]. Upwash ``w`` rotates the lift forward by
  ``w / V``, so the saving is ``W w / V`` of the drag.
- The wake moves with the air the lead left it in: a sample ``k`` steps old has
  drifted by the gust the lead met there times ``k`` steps (Taylor's frozen field,
  as the plant hands the lead's gust to its follower).
- The station the follower is given is a prediction [REALITY.md: N1 III.B]: off
  across the track by its wind estimate's error, 1 m/s (ours; 0.5 to 2 in
  [REALITY.md: M1]), times the time in trail, and held over a 30-minute window.
  What it reads of its own place from the lead is not touched, and with no sensor
  noise the station is exact.
  It is NASA's own arithmetic: a one-knot wind error is 10 ft at six seconds in
  trail [REALITY.md: N1 V.B].
- A wake crossing trips the plant [REALITY.md: N4 VI.D, N1 V.A]: the follower's
  centreline inboard of the near vortex, ``b0 / 2 = pi b / 8`` = 14.06 m from the
  wake's centreline, with the vortex within half a span of its height (ours): a
  follower further above or below flies over or under it. The patrol does not
  trip on it, and with the wake switched off (``wake_strength`` 0) there is no
  vortex to cross.
- The score prices the along-track error past the +-250 ft the Gulfstream's
  pilot was given [REALITY.md: N1 IV] and the height error past +-65 ft about
  the station's height, the RVSM tolerance about an assigned level [14 CFR 91
  App. G] (owner, 2026-11-24), each the squared excess in tolerance widths, and
  nothing sideways: the 30 m across the wake (ours, a span) is a reported band
  (owner, 2026-11-23; ``tests/test_scoring.py``).
- The follower starts in established operation, within 10 m of its place in the
  wake (charter B5: no capture).
- A bank past 60 deg is an upset, a trip [EASA AMC 25.1329, A4.2: Hazardous]; the
  patrol, which does not set ``upset_bank``, does not trip on it.
- Both seats are A320s: the lead flies the airliner's schedule, 290 KCAS then
  M 0.78, at the A320's 73.5 t.
- The reference surfs (large): on held-out seeds, with the station predicted and
  the wake drifting, it never trips, stays within 5 m of its place in the wake on
  average, inside the reported band and the along-track tolerance past the scored
  burn-in, and burns 3.5 to 10 % less fuel than on the same seeds with the wake off.
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
    compute_next_state_patrol,
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
RVSM_M = 65.0 * 0.3048


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
    assert spans == pytest.approx(1.38, abs=0.005)
    assert 1_200.0 <= float(state.slot_back) <= 1_800.0


def test_the_wake_drifts_with_the_gust_the_lead_left_it_in():
    """A steady gust held for 5 steps (no key): a sample ``k`` steps old has moved
    ``k`` times the gust, and the samples laid before the gust began 5 times."""
    _, params, state = _reset()
    gust = jnp.array([0.3, -0.8, 0.5])
    state = state.replace(gust_history=jnp.broadcast_to(gust, (20, 3)))
    drifting, still = state, state
    for _ in range(5):
        drifting, _ = compute_next_state_patrol(jnp.zeros(3), drifting, params)
        still, _ = compute_next_state_patrol(
            jnp.zeros(3), still, params.replace(wake_drift=0.0)
        )
    moved = np.asarray(drifting.wake_history - still.wake_history)[:8, :3]
    age = np.minimum(np.arange(8), 5)[:, None]
    np.testing.assert_allclose(moved, age * np.asarray(gust), atol=0.05)


def _scored(seed=3):
    env, params = tg.make("wake_surf")
    _, state = env.reset_env(jax.random.PRNGKey(seed), params)
    return env, params, state


def test_the_station_given_is_a_prediction_held_over_the_window():
    env, params, state = _scored()
    now = np.asarray(env.get_obs(state, params))
    late = np.asarray(env.get_obs(state.replace(time=1_700), params))
    assert abs(now[17] - float(state.slot_right)) > 0.1  # off, on this seed
    assert late[17] == pytest.approx(now[17], abs=1e-4)


def test_a_predicted_station_leaves_the_followers_own_place_as_read():
    """Its offset from the station moves with the station, so its place from the
    lead, their sum, is the one the fixes give: within 2 m, five times the
    0.38 m of two 0.27 m fixes differenced. The tracked distance follows."""
    env, params, state = _scored()
    obs = np.asarray(env.get_obs(state, params))
    _, right, _ = slot_error_vector(state)
    from_the_lead = float(right) + float(state.slot_right)
    assert obs[11] + obs[17] == pytest.approx(from_the_lead, abs=2.0)
    assert obs[24] == pytest.approx(np.hypot(obs[11], obs[12]), abs=1e-3)


def test_a_one_knot_wind_error_is_ten_feet_at_six_seconds_in_trail():
    """NASA's own figure for its Gulfstream pair: at a time in trail of nearly
    six seconds, "even a one-knot error in the cross-track wind component
    equates to a 10-ft position error" [REALITY.md: N1 V.B]. Flown here at 5.9 s
    with the wind estimate off by exactly one knot: 10 ft, within the half
    foot a figure printed to the foot allows."""
    env, params = tg.make("wake_surf")
    _, state = env.reset_env(jax.random.PRNGKey(3), params)
    speed = float(jnp.hypot(state.lead.x_dot, state.lead.y_dot))
    state = state.replace(slot_back=jnp.asarray(5.9 * speed, jnp.float32))
    off = float(env.get_obs(state, params)[17] - state.slot_right)
    draw = off / (5.9 * float(params.wind_estimate_sigma))
    one_knot = params.replace(wind_estimate_sigma=KT / abs(draw))
    off = float(env.get_obs(state, one_knot)[17] - state.slot_right)
    assert abs(off) / 0.3048 == pytest.approx(10.0, abs=0.5)


def test_the_true_observation_gives_the_true_station():
    """``true_obs`` is the plant now, what a claim about the plant is read on
    (the share of time on station): the station where it is, and the same
    whatever the instruments."""
    from target_gym.base import true_obs

    env, params, state = _scored()
    obs = np.asarray(true_obs(env, state, params))
    assert obs[17] == pytest.approx(float(state.slot_right), abs=1e-4)
    loud = params.replace(sensor_noise_scale=100.0)
    np.testing.assert_array_equal(true_obs(env, state, loud), obs)


def test_with_no_sensor_noise_the_station_is_exact():
    env, params, state = _reset()
    obs = np.asarray(env.get_obs(state, params))
    assert obs[17] == pytest.approx(float(state.slot_right), abs=1e-4)
    assert obs[11] == pytest.approx(float(slot_error_vector(state)[1]), abs=1e-4)


@pytest.mark.large
def test_the_stations_error_is_the_wind_estimates_times_the_time_in_trail():
    """Over 400 windows the error over the time in trail has the wind
    estimate's 1 m/s: within 0.12, three standard errors of a standard
    deviation from 400 draws (1 / sqrt(800))."""
    env, params = tg.make("wake_surf")

    def wind_error(seed):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        speed = jnp.hypot(state.lead.x_dot, state.lead.y_dot)
        error = env.get_obs(state, params)[17] - state.slot_right
        return error * speed / state.slot_back

    errors = np.asarray(jax.jit(jax.vmap(wind_error))(jnp.arange(1_000, 1_400)))
    assert errors.std() == pytest.approx(1.0, abs=0.12)
    assert abs(errors.mean()) < 0.15


@pytest.mark.parametrize("name, trips", [("wake_surf", True), ("patrol", False)])
@pytest.mark.parametrize("from_the_centreline", [12.0, 16.0])
def test_a_wake_crossing_trips_the_wake_surfer_only(name, trips, from_the_centreline):
    _, params, state = _reset(name)
    side = float(jnp.sign(state.slot_right))
    inboard = from_the_centreline - abs(float(state.slot_right))
    state = _at_station(state, right=side * inboard)
    tripped, _ = check_is_terminal_patrol(state, params)
    assert bool(tripped) is (trips and from_the_centreline < 14.06)


@pytest.mark.parametrize(
    "above, trips", [(10.0, True), (-10.0, True), (25.0, False), (-25.0, False)]
)
def test_a_crossing_needs_the_vortex_within_the_wings_reach_in_height(above, trips):
    """12 m from the wake's centreline is inboard of the near vortex, but the
    follower passes through it only with the vortex within half a span of its
    own height, 17.9 m (ours): above or below that it flies over or under it."""
    _, params, state = _reset()
    inboard = 12.0 - abs(float(state.slot_right))
    side = float(jnp.sign(state.slot_right))
    state = _at_station(state, right=side * inboard, up=above)
    tripped, _ = check_is_terminal_patrol(state, params)
    assert bool(tripped) is trips


def test_with_the_wake_switched_off_there_is_no_vortex_to_cross():
    _, params, state = _reset(wake_strength=0.0)
    inboard = 12.0 - abs(float(state.slot_right))
    state = _at_station(state, right=float(jnp.sign(state.slot_right)) * inboard)
    tripped, _ = check_is_terminal_patrol(state, params)
    assert not bool(tripped)


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
        (0.0, 0.0, 15.0, 0.0),
        (0.0, 0.0, -15.0, 0.0),
        (0.0, 0.0, 2 * RVSM_M, 1.0),
        (0.0, 0.0, -2 * RVSM_M, 1.0),
        (0.0, 60.0, 0.0, 0.0),
        (60.0, 0.0, 0.0, 0.0),
        (-60.0, 0.0, 0.0, 0.0),
        (2 * SPACING_M, 0.0, 0.0, 1.0),
        (-2 * SPACING_M, 0.0, 0.0, 1.0),
    ],
)
def test_the_score_prices_250_ft_along_the_track_65_ft_in_height_and_nothing_aside(
    back, right, up, cost
):
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
    along-track error, the offset across the track, the distance across the
    wake and the trips, and the fuel each seed burned."""
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
            out = (back, right, jnp.hypot(right, up), info["tripped"])
            return (obs, state, memory), out

        steps = jnp.arange(int(params.max_steps_in_episode))
        (_, last, _), out = jax.lax.scan(step, (obs, state, expert.reset()), steps)
        return out, fuel - last.follower.fuel

    out, burned = jax.jit(jax.vmap(one))(jnp.array(HELD_OUT))
    burn_in = scored_burn_in("wake_surf", params)
    back, right, across = (np.asarray(series)[:, burn_in:] for series in out[:3])
    return back, right, across, np.asarray(out[3]), np.asarray(burned)


@pytest.mark.large
def test_the_reference_surfs_on_its_station_and_saves_fuel():
    back, right, across, tripped, burned = _flights(1.0)
    *_, burned_alone = _flights(0.0)
    assert not tripped.any()
    assert np.abs(right.mean(1)).max() <= 5.0, right.mean(1)
    assert across.max() <= 30.0, f"{across.max():.1f} m across the wake"
    assert np.abs(back).max() <= SPACING_M, f"{np.abs(back).max():.1f} m along it"
    saving = 1.0 - burned / burned_alone
    assert (saving >= 0.035).all() and (saving <= 0.10).all(), saving
