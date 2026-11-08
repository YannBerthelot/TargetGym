"""``plane_energy`` against the real job (charter Amendments 1 and 2).

The job is TECS's: hold height and speed by trading pitch against throttle,
throttle on total energy and pitch on the split; the real cases are altitude
changes with speed held and speed changes at constant altitude [TECS; REALITY.md,
missions].

Contract (sources in src/target_gym/plane/REALITY.md and PHYSICS.md):

- Clearances (rule 17). The aircraft starts established. Every 300 s ATC gives
  one clearance, a new level or a new speed: levels move in whole 1 000 ft
  [RVSM, the vertical separation minimum], speeds in 10 kt [AIM 4-4-12, "5 or
  10 knot increments"]. The schedule, documented in PHYSICS.md: +2 000 ft at
  300 s, +10 kt at 600 s, -1 000 ft at 900 s, -10 kt at 1 200 s, -2 000 ft at
  1 500 s.
- Score (rule 15). Speed inside +-10 kt of the assigned speed [AIM 4-4-12]. In a
  level change, climb or descend at no less than 500 fpm [AIM 4-4-10]: the
  altitude band is +-65 ft [RVSM] about a level the aircraft must close at
  500 fpm from the clearance. Within 1 000 ft of the level, no more than
  1 500 fpm [AIM 4-4-10], charged past that edge in 500 fpm widths, half the
  AIM's 500 to 1 500 fpm range.

Hand-derived numbers: 2 000 ft = 609.6 m; 500 fpm = 2.54 m/s, so 60 s after the
clearance the aircraft may still be 609.6 - 152.4 = 457.2 m short. Not having
moved, it is 152.4 m past that, (152.4 / 19.812)^2 = 59.17 per second. 2 000 fpm
within 1 000 ft is 500 fpm past 1 500, (500 / 500)^2 = 1.0 per second.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg

FT, KT, FPM = 0.3048, 1852.0 / 3600.0, 0.3048 / 60.0
KEY = jax.random.PRNGKey(3)
BASE = 5000.0


def _params(**changes):
    _, params = tg.make("plane_energy")
    return params.replace(
        turbulence_sigma=0.0,
        sensor_noise_scale=0.0,
        target_altitude_range=(BASE, BASE),
        **changes,
    )


def _targets(params, steps):
    """The commanded altitude and assigned CAS at each second, from a rollout."""
    env, _ = tg.make("plane_energy")
    obs, state = env.reset_env(KEY, params)
    step = jax.jit(env.step_env)
    action = env.trim_action(state, params)
    levels, speeds = [float(state.target_altitude)], [
        float(env.assigned_cas(state, params))
    ]
    for k in jax.random.split(KEY, steps):
        obs, state, *_ = step(k, state, action, params)
        levels.append(float(state.target_altitude))
        speeds.append(float(env.assigned_cas(state, params)))
    return np.array(levels), np.array(speeds)


# -- Clearances (rule 17) --------------------------------------------------------


def test_levels_change_in_whole_thousands_of_feet_at_clearances():
    """B2: the target is piecewise constant, it moves only at 300 s marks, and
    each move is a whole number of 1 000 ft within 0.5 m [RVSM]."""
    levels, _ = _targets(_params(), 1799)
    moves = np.flatnonzero(np.diff(levels)) + 1
    assert set(moves) <= {300, 600, 900, 1200, 1500}
    steps_ft = np.diff(levels)[moves - 1] / FT
    assert np.allclose(steps_ft, np.round(steps_ft / 1000.0) * 1000.0, atol=0.5 / FT)
    assert len(moves) == 3


def test_speeds_change_in_ten_knot_steps_at_clearances():
    """Speed adjustments in 10 kt increments [AIM 4-4-12], at 300 s marks."""
    _, speeds = _targets(_params(), 1799)
    moves = np.flatnonzero(np.diff(speeds)) + 1
    assert set(moves) <= {300, 600, 900, 1200, 1500}
    assert len(moves) == 2
    assert np.allclose(np.abs(np.diff(speeds)[moves - 1]), 10.0 * KT, atol=1e-3)


def test_the_episode_starts_established():
    """The aircraft starts at its level and assigned speed [rule 17]."""
    env, _ = tg.make("plane_energy")
    params = _params()
    obs, state = env.reset_env(KEY, params)
    assert float(state.z) == pytest.approx(BASE, abs=0.5)
    airspeed = obs[env.obs_index["airspeed"]]
    assert abs(float(airspeed - obs[env.obs_index["target_speed"]])) < 0.5


# -- The trade (B3) --------------------------------------------------------------


def _energy_rate(params, action, seconds=10):
    """Mean rate of specific total energy h + V^2 / 2g, m/s, under ``action``
    from trim, airspeed from the air data channel."""
    env, _ = tg.make("plane_energy")
    obs, state = env.reset_env(KEY, params)
    step = jax.jit(env.step_env)

    def energy(o):
        v = float(o[env.obs_index["airspeed"]])
        return float(o[env.obs_index["z"]]) + v * v / (2.0 * params.gravity)

    start = energy(obs)
    for k in jax.random.split(KEY, seconds):
        obs, state, *_ = step(k, state, action, params)
    return (energy(obs) - start) / seconds


def test_pitch_trades_energy_and_throttle_adds_it():
    """B3: throttle sets total energy, pitch only splits it [TECS]. From trim,
    a pitch-only step of 10 % of its span (1.5 deg) changes the energy rate by
    under 20 % of what a throttle-only step of 10 % of its span (0.2) does
    (20 %: drag changes with speed in the pitch step, set in BEHAVIOUR_TESTS)."""
    env, _ = tg.make("plane_energy")
    params = _params()
    _, state = env.reset_env(KEY, params)
    trim = np.asarray(env.trim_action(state, params))
    base = _energy_rate(params, trim)
    pitch = _energy_rate(params, trim + np.array([0.0, 0.1]))
    throttle = _energy_rate(params, trim + np.array([0.2, 0.0]))
    assert abs(pitch - base) < 0.2 * abs(throttle - base)


# -- Score (rule 15) -------------------------------------------------------------


def _state(params, time, z, z_dot=0.0, cas=None):
    """An aircraft at ``time`` s, height ``z``, level, at assigned CAS ``cas``
    (the assigned one by default), with the clearance's commanded level."""
    from target_gym.plane.dynamics import scheduled_airspeed

    env, _ = tg.make("plane_energy")
    _, state = env.reset_env(KEY, params)
    state = state.replace(time=jnp.int32(time))
    state = state.replace(target_altitude=env.commanded_level(state, params))
    cas = env.assigned_cas(state, params) if cas is None else cas
    tas = scheduled_airspeed(z, cas, 0.99)
    return env, state.replace(z=jnp.float32(z), x_dot=tas, z_dot=jnp.float32(z_dot))


def test_a_climb_at_500_fpm_from_the_clearance_costs_nothing():
    params = _params()
    env, state = _state(params, 360, BASE + 152.4)
    assert float(-env.compute_reward(state, params)) == pytest.approx(0.0, abs=1e-6)


def test_climbing_faster_than_500_fpm_costs_nothing():
    """60 s after the clearance, 400 m up is ahead of the 152.4 m profile."""
    params = _params()
    env, state = _state(params, 360, BASE + 400.0)
    assert float(-env.compute_reward(state, params)) == pytest.approx(0.0, abs=1e-6)


def test_not_starting_a_cleared_climb_is_charged_past_the_500_fpm_profile():
    params = _params()
    env, state = _state(params, 360, BASE)
    assert float(-env.compute_reward(state, params)) == pytest.approx(59.17, rel=1e-3)


def test_more_than_1500_fpm_within_1000_ft_of_the_level_is_charged():
    params = _params()
    level = BASE + 2000.0 * FT
    # At 400 s the 500 fpm profile allows 609.6 - 254 = 355.6 m short; 200 m
    # short is inside it, so only the rate is charged.
    env, state = _state(params, 400, level - 200.0, z_dot=2000.0 * FPM)
    assert float(-env.compute_reward(state, params)) == pytest.approx(1.0, rel=1e-3)


def test_after_a_speed_clearance_the_band_is_about_the_new_speed():
    """At 610 s, 10 kt was added: the old speed sits on the band's edge (free),
    5 kt below it is 5 kt past the edge, (5 / 10)^2 = 0.25."""
    params = _params()
    level = BASE + 2000.0 * FT
    old = params.target_cas
    env, edge = _state(params, 610, level, cas=old)
    _, below = _state(params, 610, level, cas=old - 5.0 * KT)
    assert float(-env.compute_reward(edge, params)) == pytest.approx(0.0, abs=1e-3)
    assert float(-env.compute_reward(below, params)) == pytest.approx(0.25, rel=5e-3)


@pytest.mark.medium
def test_the_reference_flies_the_clearances_inside_the_rules_in_still_air():
    """The reference TECS flies the whole schedule in calm air without a trip.
    Its cost is reported, not asserted: it is the room's numerator."""
    env, _ = tg.make("plane_energy")
    params = _params()
    expert = tg.make_expert("plane_energy")
    memory = expert.reset()
    obs, state = env.reset_env(KEY, params)
    step, act = jax.jit(env.step_env), jax.jit(expert.step)
    tripped, cost = False, 0.0
    for k in jax.random.split(KEY, 1800):
        action, memory = act(memory, obs)
        obs, state, reward, _, info = step(k, state, action, params)
        tripped |= bool(info["tripped"])
        cost -= float(reward)
    assert not tripped
    print(f"reference cost on the calm schedule: {cost:.1f}")
