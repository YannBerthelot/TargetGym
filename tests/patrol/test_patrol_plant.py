"""The patrol family's plant: both patrols and the formation.

The contract is PR 7b of docs/proposals/industrial-fixes.md and PR 7's envelope
(plane/PHYSICS.md), read before any of this was written:

- Two aircraft are in the same air, not in the same gust. The follower meets the
  lead's gust ``slot_back / V`` later (frozen turbulence), and what its lateral
  and vertical offset ``r`` from the lead's path leaves shared is
  ``rho = exp(-r / L)``, Dryden's correlation on MIL-F-8785C's scale ``L``. The
  rest is its own draw, so its gust keeps the lead's variance.
- The rolling gust's scale is the span, far under any slot, so each wing has its
  own draw.
- A centre distance under the span puts wingtips together, so separation trips
  over the span: one Hornet length, 17.1 m, over its 11.4 m span (AFF's 56 ft).
- The route turns at up to +-0.0303 rad/s, the parade turn's 30 deg of bank at
  the band's middle (patrol/env.py, ``lead_turn_rate_range``).
- The lead is the 3D expert on AFF's 250 KCAS (M 0.70 past it), spawned on it.
- Every aircraft flies the amended plant (plane3d/PHYSICS.md, Amendment 2): the
  load factor, bank and Mach are protected and VMO a warning, so none of them
  trips; the ground does.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.patrol.env import GUST_MEMORY, advance_air, step_lead
from target_gym.plane.dynamics import scheduled_airspeed
from target_gym.plane3d import hornet
from target_gym.plane3d.env import check_is_terminal_3d

KEY = jax.random.PRNGKey(0)
KT = 1852.0 / 3600.0
TASKS = ("patrol", "patrol_range_bearing", "formation")
#: The hard end of the schedule's turn, from the row.
HARD_TURN = 0.0303
#: The lead's schedule: AFF's Class B 250 KCAS [AFF], M 0.70 past it.
LEAD_CAS, LEAD_MACH = 250.0 * KT, 0.70


def _reset(name, seed=0, **overrides):
    env, params = tg.make(name)
    params = params.replace(**overrides)
    return (env, params, *env.reset_env(jax.random.PRNGKey(seed), params))


def _follower(state):
    """The patrol's follower, or the formation's first wingman."""
    if hasattr(state, "follower"):
        return state.follower
    return jax.tree.map(lambda x: x[0], state.wingmen)


def _with_follower(state, **fields):
    """``state`` with the follower, or every wingman, changed."""
    if hasattr(state, "follower"):
        return state.replace(follower=state.follower.replace(**fields))
    full = {k: jnp.full_like(getattr(state.wingmen, k), v) for k, v in fields.items()}
    return state.replace(wingmen=state.wingmen.replace(**full))


def _speed(a):
    return jnp.sqrt(a.x_dot**2 + a.y_dot**2 + a.z_dot**2)


def _zero(env):
    return jnp.zeros(env.action_space().shape)


# ─── Separation ─────────────────────────────────────────


@pytest.mark.parametrize("name", TASKS)
def test_separation_trips_before_the_wingtips_meet(name):
    env, params, _, state = _reset(name, follower_spawn_noise=0.0)
    lead = state.lead
    assert params.min_separation > params.wingspan
    abeam = float(params.wingspan) + 0.5
    near = _with_follower(state, x=lead.x, y=lead.y + abeam, z=lead.z)
    assert bool(env.is_terminated(near, params))
    clear = _with_follower(state, x=lead.x, y=lead.y + 41.0, z=lead.z)
    if name == "formation":  # keep the wingmen off each other
        clear = clear.replace(
            wingmen=clear.wingmen.replace(y=lead.y + jnp.array([41.0, -41.0]))
        )
    assert not bool(env.is_terminated(clear, params))


# ─── The envelope ───────────────────────────────────────


def _at(state, lead_seat, **fields):
    if lead_seat:
        return state.replace(lead=state.lead.replace(**fields))
    return _with_follower(state, **fields)


@pytest.mark.parametrize("name", TASKS)
@pytest.mark.parametrize("lead_seat", [True, False], ids=["lead", "follower"])
@pytest.mark.parametrize(
    "fields",
    [
        # Past the Hornet's +7.5 g and -3 g [NATOPS; ours].
        pytest.param({"load_factor": 7.6}, id="over_the_limit_load"),
        pytest.param({"load_factor": -3.1}, id="under_the_negative_limit"),
        # 1.2 rad is 69 deg, past the 67 deg protection and the old 60 deg trip.
        pytest.param({"phi": 1.2}, id="past_the_bank_protection"),
        # 270 m/s at 8000 m, sound 308 m/s, is M 0.88, past the 0.85 MMO. The
        # Hornet's VMO, 700 KEAS, lies past its MMO all through the band.
        pytest.param({"z": 8000.0, "x_dot": 270.0, "y_dot": 0.0}, id="past_mmo"),
    ],
)
def test_no_aircraft_trips_past_its_protected_envelope(name, lead_seat, fields):
    """The slot cannot be lost here (``max_slot_error`` out of reach), so only the
    envelope could trip."""
    env, params, _, state = _reset(name, follower_spawn_noise=0.0, max_slot_error=1e6)
    assert not bool(env.is_terminated(_at(state, lead_seat, **fields), params))


@pytest.mark.parametrize("name", TASKS)
@pytest.mark.parametrize("lead_seat", [True, False], ids=["lead", "follower"])
def test_any_aircraft_on_the_ground_trips(name, lead_seat):
    env, params, _, state = _reset(name, follower_spawn_noise=0.0, max_slot_error=1e6)
    assert not bool(env.is_terminated(state, params))
    grounded = _at(state, lead_seat, z=params.min_alt - 1.0)
    assert bool(env.is_terminated(grounded, params))


# ─── The lead ───────────────────────────────────────────


@pytest.mark.parametrize("name", TASKS)
def test_every_aircraft_spawns_on_the_schedule_inside_the_envelope(name):
    env, params = tg.make(name)
    for seed in range(8):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        lead = state.lead
        want = scheduled_airspeed(lead.z, LEAD_CAS, LEAD_MACH)
        assert float(_speed(lead)) == pytest.approx(float(want), rel=1e-4)
        assert float(_speed(_follower(state))) == pytest.approx(float(want), rel=1e-4)
        assert not bool(env.is_terminated(state, params))


def _fly_lead(params, state, steps):
    """The scripted lead alone: ``(z, speed, psi, phi, out of the envelope)`` per
    step, and its last state."""

    @jax.jit
    def step(lead, memory):
        view = state.replace(lead=lead, lead_memory=memory)
        lead, memory, _ = step_lead(view, params)
        return lead, memory, check_is_terminal_3d(lead, params)[0]

    lead, memory, rows = state.lead, state.lead_memory, []
    for _ in range(steps):
        lead, memory, out = step(lead, memory)
        rows.append([float(x) for x in (lead.z, _speed(lead), lead.psi, lead.phi, out)])
    return np.array(rows), lead


def test_the_scripted_lead_holds_its_level_on_the_schedule():
    """In still air on a straight route: the 3D heading task's 2 m of altitude, and
    the schedule within 1 m/s, at any mass, since there is no fixed trim to wear out
    (the throttle and pitch integrators take it up). The Hornet at HARV's 15.1 t and
    with 2 t of its fuel burned (ours)."""
    for mass in (hornet.MASS, hornet.MASS - 2_000.0):
        _, params, _, state = _reset(
            "patrol",
            turbulence_sigma=0.0,
            lead_turn_rate_range=(0.0, 0.0),
            initial_mass=mass,
        )
        rows, end = _fly_lead(params, state, 90)
        assert not rows[:, 4].any()
        want = scheduled_airspeed(end.z, LEAD_CAS, LEAD_MACH)
        assert np.abs(rows[60:, 0] - float(state.lead.z)).mean() <= 2.0
        assert np.abs(rows[60:, 1] - float(want)).mean() <= 1.0


@pytest.mark.parametrize("altitude", [6096.0, 9754.0])
def test_the_hard_turn_is_a_coordinated_30_deg_bank(altitude):
    """A steady 0.0303 rad/s at the flown speed V takes tan(bank) = V w / g: on 250
    KCAS, 28 deg at 20 000 ft (172.8 m/s) and 33 deg at 32 000 ft (209.2 m/s),
    inside the lead's 45 deg."""
    _, params, _, state = _reset(
        "patrol",
        turbulence_sigma=0.0,
        lead_turn_rate_range=(HARD_TURN, HARD_TURN),
        target_altitude_range=(altitude, altitude),
    )
    rows, _ = _fly_lead(params, state, 60)
    assert not rows[:, 4].any()
    psi = np.unwrap(rows[:, 2])
    assert (psi[-1] - psi[30]) / 29.0 == pytest.approx(HARD_TURN, rel=0.02)
    want = np.arctan(rows[30:, 1].mean() * HARD_TURN / params.gravity)
    assert np.abs(rows[30:, 3]).mean() == pytest.approx(want, abs=np.deg2rad(2.0))


def test_the_default_route_reaches_the_hard_turn():
    assert tg.make("patrol")[1].lead_turn_rate_range == (-HARD_TURN, HARD_TURN)


def test_the_formation_lead_seat_flies_the_schedule():
    """The expert's lead holds the patrol lead's 250 KCAS at the AFF point."""
    env, params = tg.make("formation")
    params = params.replace(
        turbulence_sigma=0.0,
        target_altitude_range=(7620.0, 7620.0),
        lead_turn_rate_range=(0.0, 0.0),
        follower_spawn_noise=0.0,
        sensor_noise_scale=0.0,
    )
    team = tg.make_expert("formation")
    memory = team.reset()
    obs, state = env.reset_env(KEY, params)
    step, act = jax.jit(env.step_env), jax.jit(team.step)
    for t in range(60):
        action, memory = act(memory, obs)
        obs, state, _, _, info = step(jax.random.PRNGKey(t), state, action, params)
        assert not bool(info["tripped"])
    want = scheduled_airspeed(state.lead.z, LEAD_CAS, LEAD_MACH)
    assert float(_speed(state.lead)) == pytest.approx(float(want), abs=2.0)


# ─── The air ────────────────────────────────────────────


@pytest.mark.parametrize(
    "steps_back, weights",
    [(2.0, {2: 1.0}), (1.5, {1: 0.5, 2: 0.5})],
    ids=["two_steps", "between_samples"],
)
def test_the_follower_meets_the_leads_gust_a_slot_distance_later(steps_back, weights):
    """Dead astern (r = 0, so rho = 1) the follower's gust is exactly the lead's,
    delayed by slot_back / V, interpolated between step samples. The lead's speed
    wanders about 1% in the run, so the delay wanders 0.02 of a step, which against
    a 4 sigma innovation of 4.8 m/s between samples is 0.1 m/s."""
    env, params, _, state = _reset(
        "patrol",
        follower_spawn_noise=0.0,
        lead_turn_rate_range=(0.0, 0.0),
        target_altitude_range=(5000.0, 5000.0),
        wake_strength=0.0,  # dead astern, the wake would be the larger term
    )
    back = steps_back * float(_speed(state.lead)) * float(params.delta_t)
    state = state.replace(slot_back=back, slot_right=0.0, slot_up=0.0)
    step = jax.jit(env.step_env)
    action = jnp.array([0.2, 0.2, 0.0])
    lead, follower = [], []
    for t in range(40):
        _, state, _, _, _ = step(jax.random.PRNGKey(t), state, action, params)
        lead.append(np.array([state.lead.gust_x, state.lead.gust_y, state.lead.gust_z]))
        follower.append(
            np.array(
                [state.follower.gust_x, state.follower.gust_y, state.follower.gust_z]
            )
        )
    lead, follower = np.array(lead), np.array(follower)
    assert lead.std() > 0.5
    expected = sum(w * lead[3 - i : 3 - i + 37] for i, w in weights.items())
    np.testing.assert_allclose(follower[3:], expected, atol=0.1)


def _air_samples(back, right, up, chains=4000, steps=60):
    """``chains`` independent runs of the air alone, sampled once each past the
    gust's 5 s memory: the lead's gust ``steps_back`` behind and the follower's."""
    params = tg.make("patrol")[1]
    speed = 200.0

    def chain(key):
        history = jnp.zeros((GUST_MEMORY, 3))
        own = jnp.zeros((1, 3))

        def body(carry, t):
            history, own = carry
            history, own, gusts, _ = advance_air(
                history,
                own,
                jnp.array([back]),
                jnp.array([right]),
                jnp.array([up]),
                speed,
                params,
                key,
                t,
            )
            return (history, own), gusts

        (history, _), gusts = jax.lax.scan(body, (history, own), jnp.arange(steps))
        return history, gusts[-1, 1]

    return jax.jit(jax.vmap(chain))(jax.random.split(KEY, chains))


def test_offset_air_decorrelates_and_keeps_its_variance():
    """Offset 150 m across and 60 m up, r = 161.6 m on L = 533.4 m, so
    rho = exp(-r / L) = 0.739 with the delayed lead gust. The OU's stationary
    std, sigma / sqrt(1 - (1 - theta dt)^2) = 0.36 / sqrt(1 - 0.8^2) = 0.6 m/s
    (AFF's light chop), is the follower's too: the decorrelated part replaces
    variance, it does not add it. Over 4000 independent chains the sample
    correlation's standard error is (1 - rho^2) / sqrt(n) = 0.0072, and over their
    12 000 values (three independent axes each) the std's is 0.6 / sqrt(2 x
    12 000) = 0.0039, and each bound is four of them."""
    history, follower = _air_samples(back=0.0, right=150.0, up=60.0)
    lead = np.asarray(history[:, 0, :])
    follower = np.asarray(follower)
    assert np.corrcoef(lead.ravel(), follower.ravel())[0, 1] == pytest.approx(
        np.exp(-np.hypot(150.0, 60.0) / 533.4), abs=4 * 0.0072
    )
    assert follower.std() == pytest.approx(0.6, abs=4 * 0.0039)
    assert lead.std() == pytest.approx(0.6, abs=4 * 0.0039)


def test_the_steady_wind_has_no_gust_and_no_rolling_gust():
    """``key=None`` keeps a constant wind: the air neither moves nor rolls a wing."""
    params = tg.make("patrol")[1]
    history = jnp.ones((GUST_MEMORY, 3))
    own = 2.0 * jnp.ones((1, 3))
    new_history, new_own, gusts, rolls = advance_air(
        history,
        own,
        jnp.array([0.0]),
        jnp.array([0.0]),
        jnp.array([0.0]),
        200.0,
        params,
        None,
        0,
    )
    np.testing.assert_array_equal(new_history, history)
    np.testing.assert_array_equal(new_own, own)
    np.testing.assert_allclose(gusts, 1.0)
    np.testing.assert_array_equal(rolls, 0.0)


def _banks(name, params, env, state, steps=12):
    """Every seat's bank per step under wings-level demands: the lead's, then the
    follower's or each wingman's."""
    step = jax.jit(env.step_env)
    rows = []
    for t in range(steps):
        _, state, _, _, _ = step(jax.random.PRNGKey(t), state, _zero(env), params)
        wings = state.wingmen.phi if name == "formation" else state.follower.phi[None]
        rows.append(np.concatenate([np.asarray(state.lead.phi)[None], wings]))
    return np.array(rows)


@pytest.mark.parametrize("name", ["patrol", "formation"])
def test_turbulence_rolls_a_follower_held_wings_level(name):
    """Zero aileron gives zero bank in still air (plane3d's test), so a bank in
    turbulence under wings-level demands is the rolling gust."""
    env, params, _, state = _reset(
        name,
        lead_turn_rate_range=(0.0, 0.0),
        follower_spawn_noise=0.0,
        wake_strength=0.0,  # the wake rolls a wing too, in still air
    )
    still = params.replace(turbulence_sigma=0.0)
    np.testing.assert_array_equal(_banks(name, still, env, state)[:, 1:], 0.0)
    assert np.abs(_banks(name, params, env, state)[:, 1:]).max() > 1e-3


def test_each_wing_has_its_own_rolling_gust():
    """Every seat of the formation holds wings level, so their banks differ only
    by the gust: aircraft 150 m or more apart, whose span-sized gusts are unrelated."""
    env, params, _, state = _reset(
        "formation", lead_turn_rate_range=(0.0, 0.0), follower_spawn_noise=0.0
    )
    lead, first, second = _banks("formation", params, env, state).T
    assert np.abs(first - lead).max() > 1e-3
    assert np.abs(second - lead).max() > 1e-3
    assert np.abs(second - first).max() > 1e-3


def test_formation_wingmen_are_in_their_own_air():
    env, params, _, state = _reset("formation", follower_spawn_noise=0.0)
    step = jax.jit(env.step_env)
    action = _zero(env)
    for t in range(20):
        _, state, _, _, _ = step(jax.random.PRNGKey(t), state, action, params)
    gust = np.asarray(state.wingmen.gust_y)
    assert gust.shape == (2,)
    assert abs(gust[0] - gust[1]) > 1e-3


def test_the_air_reaches_the_formations_last_rank():
    """Rank 2 of a four-wingman V sits at twice the 300 m slot depth: 600 m at the
    schedule's slowest 170 m/s is 3.53 steps. Samples ramping by 1 m/s, newest 0,
    become 0, 0, 1, 2, 3 once a steady wind repeats the newest, and 3.53 steps back
    they read 2.53, so the memory reaches that far."""
    params = tg.make("formation")[1]
    history = jnp.arange(float(GUST_MEMORY))[:, None] * jnp.ones((GUST_MEMORY, 3))
    no_offset = jnp.array([0.0])
    _, _, gusts, _ = advance_air(
        history,
        jnp.zeros((1, 3)),
        jnp.array([600.0]),
        no_offset,
        no_offset,
        170.0,
        params,
        None,
        0,
    )
    np.testing.assert_allclose(gusts[1], 600.0 / 170.0 - 1.0, rtol=1e-6)


# ─── The rename ─────────────────────────────────────────


def test_the_old_task_name_points_at_the_new_one():
    with pytest.raises(KeyError, match="is now 'patrol_range_bearing'"):
        tg.make("patrol_bearing_only")
