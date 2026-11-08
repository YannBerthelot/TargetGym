"""``plane3d_heading`` and ``plane3d_racetrack`` against the real aircraft
(charter Amendments 1 and 2).

From src/target_gym/plane/REALITY.md and BEHAVIOUR_TESTS.md, with the bands of
the instrument-rating ACS (FAA-S-ACS-8, IR.III.B.S4: holding at "airspeed +-10
knots, altitude +-100 feet, selected headings within +-10 deg, and track a
selected course ... within 3/4-scale deflection of the CDI") and the AIM's GPS
terminal CDI scale, +-1 NM full scale (AIM 1-1-17).

Contract:

- Limits (rule 14): load factor and AoA protected, bank and VMO/MMO not trips
  [CFR 25.337, NTSB 1.6.3, BEA447, TCDS].
- Turn physics (A1, A2): rate g tan(bank) / V within 3 %, load 1 / cos(bank)
  within 0.02 g, in a level coordinated turn.
- Readings (rule 16): airspeed and normal acceleration.
- Episode (rule 17): established, wings level, at the assigned level and speed.
- Heading score (rule 15): altitude +-65 ft [RVSM], speed +-10 kt [AIM 4-4-12],
  heading +-10 deg [ACS] about a heading the aircraft turns toward at no less
  than the holding turn's rate, 3 deg/s or 30 deg of bank, whichever is less
  [IFH ch. 10]; past an edge, the squared excess in band widths.
- Holding (B5): right turns, inbound leg 1 min at or below 14 000 ft and 1.5 min
  above, turns at 3 deg/s or 30 deg of bank, speed at most 230 KIAS up to
  14 000 ft and 265 above [IFH ch. 10]; cross-track inside 3/4 of +-1 NM
  [ACS, AIM], headings +-10 deg of the pattern's [ACS].

Hand-derived numbers: 3/4 NM = 1 389 m; 2 778 m off the pattern is 1 389 m past
the edge, (1389 / 1389)^2 = 1.0. Flying a leg backwards is 180 deg off, 170 deg
past the edge, (170 / 10)^2 = 289.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.plane.dynamics import scheduled_airspeed

FT, KT, G = 0.3048, 1852.0 / 3600.0, 9.81
BAND = 65.0 * FT
KEY = jax.random.PRNGKey(11)
HEADING, RACETRACK = "plane3d_heading", "plane3d_racetrack"


def _params(name, **changes):
    _, params = tg.make(name)
    return params.replace(turbulence_sigma=0.0, sensor_noise_scale=0.0, **changes)


def _rollout(name, params, steps, act=None, key=KEY):
    """States and observations under the reference expert, its action passed
    through ``act(obs, action)`` when given."""
    env, _ = tg.make(name)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    expert = tg.make_expert(name)
    memory = expert.reset()
    policy = jax.jit(expert.step)
    states, observations, tripped = [state], [obs], False
    for k in jax.random.split(key, steps):
        action, memory = policy(memory, obs)
        if act is not None:
            action = act(obs, action)
        obs, state, _, _, info = step(k, state, action, params)
        tripped |= bool(info["tripped"])
        states.append(state)
        observations.append(obs)
    return states, observations, tripped


# -- Limits (rule 14) -------------------------------------------------------------


def test_full_pull_in_full_bank_is_protected_not_tripped():
    """A3: full nose-up, full bank and full throttle for 30 s: no trip, load
    within [-1.0, 2.5] g plus 0.1 g, AoA under the stall."""
    params = _params(HEADING, target_altitude_range=(6000.0, 6000.0))
    states, _, tripped = _rollout(
        HEADING, params, 30, act=lambda o, a: jnp.array([1.0, 1.0, 1.0])
    )
    n = np.array([float(s.load_factor) for s in states])
    assert not tripped
    assert n.max() <= 2.6 and n.min() >= -1.1
    assert np.rad2deg([float(s.alpha) for s in states]).max() < params.aoa_stall


@pytest.mark.parametrize("name", (HEADING, RACETRACK))
def test_bank_and_overspeed_are_not_trips(name):
    """Normal law protects bank; VMO is a warning [BEA447, TCDS]."""
    env, _ = tg.make(name)
    params = _params(name)
    _, state = env.reset_env(KEY, params)
    banked = state.replace(phi=jnp.float32(np.deg2rad(62.0)))
    fast = state.replace(x_dot=jnp.float32(300.0), y_dot=jnp.float32(0.0))
    assert not bool(env.is_terminated(banked, params))
    assert not bool(env.is_terminated(fast, params))


@pytest.mark.parametrize("name", (HEADING, RACETRACK))
def test_a_full_bank_demand_is_held_at_the_protected_bank(name):
    """Rule 14, Amendment 3: the autopilot's bank is a protected limit, 25 to 30
    deg in the family [IFH ch. 10's 30 deg holding bank; the 25 deg of ICAO Doc
    8168 and the A320's HDG mode, unconfirmed], for agent and reference alike.
    The limit is on the demand, as the autopilot's own bank limit is; the roll
    loop then holds it. Full bank demand for 20 s: the bank settles within 2 deg
    of the 30 deg chosen (ours, inside that range), and peaks nearer 30 than
    the 45 deg demand the agent had before (37.5)."""
    params = _params(name, target_altitude_range=(6000.0, 6000.0))
    states, _, tripped = _rollout(
        name, params, 20, act=lambda o, a: a.at[2].set(1.0)
    )
    phi = np.rad2deg([float(s.phi) for s in states])
    assert not tripped
    assert phi.max() < 37.5
    assert abs(phi[-5:].mean() - 30.0) < 2.0


# -- Turn physics (A1, A2) --------------------------------------------------------


@pytest.mark.medium
@pytest.mark.parametrize("bank_deg", (25.0, 30.0))
def test_a_level_turn_follows_the_kinematics(bank_deg):
    """A1, A2: with the bank held and TECS on height and speed, the last 30 s of
    a 90 s turn turns at g tan(bank) / V within 3 % and pulls 1 / cos(bank)
    within 0.02 g (measured bank and airspeed)."""
    env, _ = tg.make(HEADING)
    params = _params(HEADING, target_heading_range=(0.0, 0.0))
    bank = bank_deg / 45.0

    def hold_bank(obs, action):
        return action.at[2].set(bank)

    states, obs, tripped = _rollout(HEADING, params, 90, act=hold_bank)
    window = states[60:]
    psi = np.unwrap([float(s.psi) for s in window])
    rate = np.diff(psi).mean()
    phi = np.mean([float(s.phi) for s in window])
    speed = np.mean([float(o[env.obs_index["airspeed"]]) for o in obs[60:]])
    n = np.mean([float(o[env.obs_index["load_factor"]]) for o in obs[60:]])
    assert not tripped
    assert rate == pytest.approx(G * np.tan(phi) / speed, rel=0.03)
    assert n == pytest.approx(1.0 / np.cos(phi), abs=0.02)


# -- Readings (rule 16) and the start (rule 17) -----------------------------------


@pytest.mark.parametrize("name", (HEADING, RACETRACK))
def test_the_aircraft_starts_established_and_reads_air_data(name):
    """At the assigned level, wings level, at the assigned speed, reading 1 g;
    holding the reset's trim keeps it in +-65 ft for 60 s of calm air."""
    env, _ = tg.make(name)
    params = _params(name)
    obs, state = env.reset_env(KEY, params)
    i = env.obs_index
    assert float(state.z) == pytest.approx(float(state.target_altitude), abs=0.5)
    assert float(state.phi) == 0.0
    assert abs(float(obs[i["airspeed"]] - obs[i["target_speed"]])) < 10.0 * KT
    assert abs(float(obs[i["load_factor"]]) - 1.0) < 0.02
    step = jax.jit(env.step_env)
    trim = env.trim_action(state, params)
    worst = 0.0
    for k in jax.random.split(KEY, 60):
        obs, state, *_ = step(k, state, trim, params)
        worst = max(worst, abs(float(state.z - state.target_altitude)))
    assert worst < BAND


# -- The heading score (rule 15) --------------------------------------------------


def _heading_state(time, heading_deg, z_off=0.0):
    """Heading task at 5 000 m, cleared at reset onto 90 deg from 0 deg."""
    env, _ = tg.make(HEADING)
    params = _params(
        HEADING,
        target_altitude_range=(5000.0, 5000.0),
        target_heading_range=(np.pi / 2, np.pi / 2),
    )
    _, state = env.reset_env(KEY, params)
    psi = np.deg2rad(heading_deg)
    v = float(jnp.hypot(state.x_dot, state.y_dot))
    state = state.replace(
        time=jnp.int32(time),
        psi=jnp.float32(psi),
        x_dot=jnp.float32(v * np.cos(psi)),
        y_dot=jnp.float32(v * np.sin(psi)),
        z=state.z + z_off,
    )
    return env, params, state


def test_a_heading_inside_its_bands_costs_nothing():
    """5 deg and 15 m off, long after the turn: inside every band."""
    env, params, state = _heading_state(300, 85.0, z_off=15.0)
    assert float(-env.compute_reward(state, params)) == pytest.approx(0.0, abs=1e-6)


def test_the_clearance_itself_costs_nothing():
    """At the clearance the whole turn is still allowed."""
    env, params, state = _heading_state(0, 0.0)
    assert float(-env.compute_reward(state, params)) == pytest.approx(0.0, abs=1e-6)


def test_not_turning_is_charged_past_the_holding_turn_rate():
    """30 s after a 90 deg clearance, not turned: the turn at min(3 deg/s,
    g tan 30 / V) has closed rate * 30 s, so that much less 10 deg is past the
    edge, in 10 deg widths, squared."""
    env, params, state = _heading_state(30, 0.0)
    v = float(scheduled_airspeed(5000.0, params.target_cas, params.target_mach))
    rate = min(3.0, np.rad2deg(G * np.tan(np.deg2rad(30.0)) / v))
    expected = ((rate * 30.0 - 10.0) / 10.0) ** 2
    assert float(-env.compute_reward(state, params)) == pytest.approx(
        expected, rel=1e-3
    )


@pytest.mark.medium
def test_a_90_deg_change_at_25_deg_bank_takes_its_kinematic_time():
    """B4: 90 deg at 25 deg bank, rolling out with a lead of half the bank
    [IFH ch. 7]: roll-in to wings level within 15 % of 90 / rate. Wings level
    is when the bank first comes back to zero (below 1 deg of the turn's side),
    not when the roll loop's later swing has settled."""
    env, _ = tg.make(HEADING)
    params = _params(
        HEADING,
        target_altitude_range=(5000.0, 5000.0),
        target_heading_range=(np.pi / 2, np.pi / 2),
    )
    lead = np.deg2rad(12.5)

    def turn(obs, action):
        left = float(obs[env.obs_index["target_heading"]] - obs[env.obs_index["psi"]])
        return action.at[2].set(jnp.where(left > lead, 25.0 / 45.0, 0.0))

    states, obs, _ = _rollout(HEADING, params, 150, act=turn)
    phi = np.rad2deg([float(s.phi) for s in states])
    end = 10 + int(np.argmax(phi[10:] < 1.0))
    v = np.mean([float(o[env.obs_index["airspeed"]]) for o in obs[:end]])
    rate = np.rad2deg(G * np.tan(np.deg2rad(25.0)) / v)
    assert end == pytest.approx(90.0 / rate, rel=0.15)


# -- The hold (B5) ----------------------------------------------------------------


@pytest.mark.parametrize("level", (4000.0, 7000.0))
def test_the_pattern_fits_the_holding_rules(level):
    """B5: inbound leg 60 s at or below 14 000 ft (4 267 m) and 90 s above,
    within 5 s; turns no tighter than 3 deg/s and 30 deg of bank; the assigned
    speed's band stays at or under the cap, 230 KIAS to 14 000 ft, 265 above."""
    env, _ = tg.make(RACETRACK)
    params = _params(RACETRACK, target_altitude_range=(level, level))
    _, state = env.reset_env(KEY, params)
    cas = float(env.assigned_cas(state, params))
    tas = float(scheduled_airspeed(level, cas, 0.99))
    leg = 2.0 * float(state.target_half_leg) / tas
    r = float(state.target_radius)
    cap, minute = (230.0, 60.0) if level <= 14000 * FT else (265.0, 90.0)
    assert leg == pytest.approx(minute, abs=5.0)
    assert np.rad2deg(tas / r) <= 3.0 + 1e-3
    assert np.rad2deg(np.arctan(tas**2 / (G * r))) <= 30.0 + 1e-3
    assert cas + 10.0 * KT <= cap * KT + 1e-3


def _on_pattern(offset=0.0, backwards=False):
    """An aircraft abeam on the pattern's leg, ``offset`` m outside it."""
    env, _ = tg.make(RACETRACK)
    params = _params(RACETRACK, target_altitude_range=(4000.0, 4000.0))
    _, state = env.reset_env(KEY, params)
    c, s = np.cos(state.target_heading), np.sin(state.target_heading)
    out = jnp.array([-s, c]) * offset
    state = state.replace(x=state.x + out[0], y=state.y + out[1])
    if backwards:
        state = state.replace(
            x_dot=-state.x_dot, y_dot=-state.y_dot, psi=state.psi + jnp.pi
        )
    return env, params, state


def test_inside_three_quarters_of_a_mile_costs_nothing():
    env, params, state = _on_pattern(offset=1000.0)
    assert float(-env.compute_reward(state, params)) == pytest.approx(0.0, abs=1e-6)


def test_two_widths_off_the_pattern_cost_one_per_second():
    env, params, state = _on_pattern(offset=2778.0)
    assert float(-env.compute_reward(state, params)) == pytest.approx(1.0, rel=2e-3)


def test_flying_the_pattern_backwards_is_charged_on_heading():
    """Right turns only [IFH ch. 10]: on the leg, reversed, 170 deg past +-10."""
    env, params, state = _on_pattern(backwards=True)
    assert float(-env.compute_reward(state, params)) == pytest.approx(289.0, rel=2e-3)


# -- The reference (medium) -------------------------------------------------------


@pytest.mark.medium
@pytest.mark.parametrize("name", (HEADING, RACETRACK))
def test_the_reference_holds_the_rvsm_band_in_still_air(name):
    """B1: +-65 ft in calm air [RVSM] through the task under the reference,
    whose whole-episode cost is printed (the room's numerator), not asserted."""
    env, _ = tg.make(name)
    params = _params(name)
    states, _, tripped = _rollout(name, params, int(params.max_steps_in_episode))
    error = np.array([float(s.z - s.target_altitude) for s in states])
    cost = sum(float(-env.compute_reward(s, params)) for s in states[1:])
    assert not tripped
    assert np.abs(error).max() < BAND
    print(f"{name}: reference cost in calm air {cost:.1f}")


@pytest.mark.medium
@pytest.mark.parametrize("wind_kt", (0.0, 30.0))
def test_the_hold_reference_flies_its_assigned_speed(wind_kt):
    """Rule 3 (Amendment 3): the reference is the family's autopilot. It reads
    airspeed, as TECS does [TECS; REALITY.md, readings], and its slowest speed
    is 1.2 times the level stall, "20% higher than level flight stall speed"
    (ARSPD_FBW_MIN [P409]), raised by the load factor of the bank it demands,
    as STALL_PREVENTION's "roll limits based on aerodynamic load factor in
    turns" [P409]. So on a low hold, 220 KIAS under the 230 cap [IFH ch. 10],
    in still air and in a steady 30 kt wind across the pattern, its airspeed
    stays inside +-10 kt [AIM 4-4-12] after its first 60 s, through 600 s."""
    env, _ = tg.make(RACETRACK)
    params = _params(
        RACETRACK,
        target_altitude_range=(4000.0, 4000.0),
        wind_y=float(wind_kt * KT),
    )
    states, obs, tripped = _rollout(RACETRACK, params, 600)
    speed = np.array([float(o[env.obs_index["airspeed"]]) for o in obs[60:]])
    target = np.array([float(o[env.obs_index["target_speed"]]) for o in obs[60:]])
    assert not tripped
    assert np.abs(speed - target).max() < 10.0 * KT


def test_the_reference_l1_sits_in_its_documented_ranges():
    """C1: NAVL1_DAMPING 0.6 to 1.0 and advised 0.7 to 0.85; NAVL1_PERIOD at
    or above its conservative default of 20 s for a less responsive airframe
    [P409]."""
    from target_gym.plane3d.experts import _PATH

    assert 0.7 <= _PATH.damping <= 0.85
    assert _PATH.period >= 20.0
