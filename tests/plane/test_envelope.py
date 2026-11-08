"""The aircraft's envelope (docs/proposals/industrial-fixes.md, PR 7, and
charter Amendment 2).

Contract, each from a published figure:

- The 2D aircraft's flight controls protect the limit load factors of 14 CFR
  25.337, +2.5 and -1, and its AoA; VMO 350 kt and MMO 0.82 [TCDS] are a
  warning (``overspeed``), not a trip (plane/REALITY.md, limits;
  tests/plane/test_plane_reality.py). The 3D aircraft still trips on them and
  on 60 deg of bank until its own Amendment 2 change.
- The wing's lift stops at a buffet line that falls with Mach, instead of rising
  with Prandtl-Glauert as it did. The tail keeps its own limit.
- The ``plane`` family's airspeed demand is a schedule of 290 kt CAS and M 0.78
  (EUROCONTROL's A320 climb), because 230 m/s at 3000 m is past VMO.

Speeds are by the standard atmosphere: rho(3000) = 0.9091, so 230 m/s is
230 sqrt(0.9091 / 1.225) = 198.1 m/s EAS, 385 kt.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane.dynamics import (
    buffet_line,
    compute_acceleration,
    overspeed,
    scheduled_airspeed,
)
from target_gym.plane.env import PlaneParams, airspeed, check_is_terminal
from target_gym.plane.env_jax import Airplane2D
from target_gym.plane3d.env import check_is_terminal_3d
from target_gym.plane3d.env_jax import Plane3DHeading

KEY = jax.random.PRNGKey(0)
KT = 1852.0 / 3600.0


def _state_2d(z, speed, load_factor=1.0):
    env = Airplane2D()
    params = env.default_params.replace(turbulence_sigma=0.0)
    _, state = env.reset_env(KEY, params)
    return params, state.replace(
        z=jnp.float32(z),
        x_dot=jnp.float32(speed),
        z_dot=jnp.float32(0.0),
        load_factor=jnp.float32(load_factor),
    )


def _state_3d(z, speed, phi_deg=0.0, load_factor=1.0):
    env = Plane3DHeading()
    params = env.default_params.replace(turbulence_sigma=0.0)
    _, state = env.reset_env(KEY, params)
    return params, state.replace(
        z=jnp.float32(z),
        x_dot=jnp.float32(speed),
        y_dot=jnp.float32(0.0),
        z_dot=jnp.float32(0.0),
        phi=jnp.float32(np.deg2rad(phi_deg)),
        load_factor=jnp.float32(load_factor),
    )


def _trips(params, state):
    check = (
        check_is_terminal if isinstance(params, PlaneParams) else check_is_terminal_3d
    )
    return bool(check(state, params)[0])


def _overspeed(params, state):
    """The VMO/MMO warning on a 2D state, against the wind it flies in."""
    return bool(overspeed(airspeed(state, params), state.z, params))


# -- Trips --------------------------------------------------------------------


# 5000 m, 180 m/s: rho 0.7361, EAS 139.5 m/s (271 kt), M 0.56: inside everything.
@pytest.mark.parametrize("make", (_state_2d, _state_3d))
def test_a_cruise_inside_the_envelope_does_not_trip(make):
    assert not _trips(*make(5000.0, 180.0))


@pytest.mark.parametrize(
    "load_factor,trips",
    [(2.4, False), (2.6, True), (-0.9, False), (-1.1, True)],
)
@pytest.mark.parametrize("make", (_state_3d,))
def test_the_limit_load_factors_trip(make, load_factor, trips):
    assert _trips(*make(5000.0, 180.0, load_factor=load_factor)) is trips


# At 3000 m, EAS = 0.8615 TAS, so 350 kt (180.06 m/s EAS) is 209 m/s of TAS, where
# M = 0.636: this trips on EAS alone.
@pytest.mark.parametrize("speed,trips", [(205.0, False), (213.0, True)])
def test_vmo_trips_on_equivalent_airspeed(speed, trips):
    assert _trips(*_state_3d(3000.0, speed)) is trips


@pytest.mark.parametrize("speed,warns", [(205.0, False), (213.0, True)])
def test_the_vmo_warning_reads_equivalent_airspeed(speed, warns):
    assert _overspeed(*_state_2d(3000.0, speed)) is warns


# At 8000 m, a = 308.1 m/s, so M 0.82 is 252.6 m/s, whose EAS is 165 m/s: this
# trips on Mach alone.
@pytest.mark.parametrize("speed,trips", [(246.0, False), (258.0, True)])
def test_mmo_trips_on_mach(speed, trips):
    assert _trips(*_state_3d(8000.0, speed)) is trips


@pytest.mark.parametrize("speed,warns", [(246.0, False), (258.0, True)])
def test_the_mmo_warning_reads_mach(speed, warns):
    assert _overspeed(*_state_2d(8000.0, speed)) is warns


@pytest.mark.parametrize("bank,trips", [(58.0, False), (62.0, True), (-62.0, True)])
def test_the_bank_limit_trips(bank, trips):
    assert _trips(*_state_3d(5000.0, 180.0, phi_deg=bank)) is trips


def test_the_old_cruise_at_3000_m_is_past_vmo():
    """The review's 385 KEAS, by the module docstring's arithmetic."""
    params, state = _state_2d(3000.0, 230.0)
    eas = 230.0 * np.sqrt(float(state.rho) / 1.225)
    assert eas / KT == pytest.approx(385.0, abs=1.0)
    assert _overspeed(params, state)


def test_a_steady_headwind_counts_toward_the_airspeed():
    # 180 m/s over the ground into 40 m/s of wind is 220 m/s of air, past 209.
    params, state = _state_2d(3000.0, 180.0)
    assert not _overspeed(params, state)
    assert _overspeed(params.replace(wind_x=-40.0), state)


# Full nose-up no longer trips: normal law protects the load factor
# (test_plane_reality.test_full_pitch_demand_is_protected_not_tripped).


def test_a_new_load_factor_is_read_each_step():
    env = Airplane2D()
    params = env.default_params.replace(turbulence_sigma=0.0)
    _, state = env.reset_env(KEY, params)
    state = state.replace(z=jnp.float32(5000.0), load_factor=jnp.float32(9.0))
    new = env.step_env(KEY, state, jnp.array([0.0, 0.1]), params)[1]
    assert float(new.load_factor) < 2.5


# -- The buffet line ----------------------------------------------------------


def test_the_buffet_line_falls_with_mach():
    params = PlaneParams()
    # CL_max 1.5 up to M 0.3, then 1.4 per unit of Mach.
    assert float(buffet_line(0.2, params)) == pytest.approx(1.5)
    assert float(buffet_line(0.7, params)) == pytest.approx(0.94)
    # The anchor: 65 t at FL390, M 0.78, has CL 0.632, and 1.3 times that is
    # 0.822, the 30% margin a cruise keeps to buffet.
    assert float(buffet_line(0.78, params)) == pytest.approx(0.828)


def test_the_wing_stops_at_the_buffet_line_at_high_mach():
    """At 3500 m and 230 m/s (M 0.704) an incidence of 14 deg asks for
    CL = (0.2 + 0.08786 * 14) / 0.710 = 2.0 of the wing, the old limit. The line
    is 1.5 - 1.4 * 0.404 = 0.934 there, so the line takes off 1.07 of it."""
    params = PlaneParams()
    no_line = params.replace(buffet_slope=-100.0)
    velocities = jnp.array([230.0, 0.0, 0.0])
    positions = jnp.array([0.0, 3500.0, jnp.deg2rad(14.0)])
    action = (0.0, 0.0)
    _, kept = compute_acceleration(velocities, positions, action, params)
    _, free = compute_acceleration(velocities, positions, action, no_line)
    q = 0.5 * 0.8634 * 230.0**2
    lost = (float(free[1]) - float(kept[1])) / (q * params.wings_surface)
    assert lost == pytest.approx(1.07, abs=0.05)


def test_the_tail_keeps_its_authority_at_high_mach():
    """Buffet is the wing's: a nose-down elevator at M 0.75 must still move the
    pitching moment past where the line would have capped a tail."""
    params = PlaneParams()
    velocities = jnp.array([230.0, 0.0, 0.0])
    positions = jnp.array([0.0, 8000.0, 0.02])

    def moment(stick_deg):
        action = (30_000.0, jnp.deg2rad(stick_deg))
        return float(compute_acceleration(velocities, positions, action, params)[0][2])

    # Below the tail's own limit the moment stays linear in the stick.
    near = moment(-2.0) - moment(0.0)
    far = moment(-7.0) - moment(-5.0)
    assert far == pytest.approx(near, rel=0.25)


# -- The speed schedule -------------------------------------------------------


@pytest.mark.parametrize(
    "z,mach",
    [
        # By the compressible pitot relation at 290 kt CAS, qc = 14 299 Pa:
        # M = sqrt(5 ((qc / P + 1)^(2/7) - 1)), with P 70 109 at 3000 m,
        # 50 533 at 5500 m and 35 601 at 8000 m.
        (3000.0, 0.522),
        (5500.0, 0.607),
        (8000.0, 0.712),
    ],
)
def test_the_schedule_holds_cas_below_the_crossover(z, mach):
    a = np.sqrt(1.4 * 287.0 * (288.15 - 0.0065 * z))
    speed = float(scheduled_airspeed(z, 290.0 * KT, 0.78))
    assert speed == pytest.approx(mach * a, rel=2e-3)


def test_the_schedule_holds_mach_above_the_crossover():
    # 290 kt reaches M 0.78 near 9600 m, so 10 000 m is on the Mach branch.
    a = np.sqrt(1.4 * 287.0 * (288.15 - 0.0065 * 10_000.0))
    speed = float(scheduled_airspeed(10_000.0, 290.0 * KT, 0.78))
    assert speed == pytest.approx(0.78 * a, rel=1e-3)


@pytest.mark.parametrize("z", (3000.0, 5500.0, 8000.0))
def test_the_schedule_stays_inside_vmo_and_mmo(z):
    params, state = _state_2d(z, 180.0)
    speed = float(scheduled_airspeed(z, params.target_cas, params.target_mach))
    params, state = _state_2d(z, speed)
    assert not _overspeed(params, state)


def test_the_observation_commands_the_scheduled_airspeed():
    env = Airplane2D()
    params = env.default_params.replace(turbulence_sigma=0.0, sensor_noise_scale=0.0)
    _, state = env.reset_env(KEY, params)
    state = state.replace(z=jnp.float32(3000.0))
    expected = scheduled_airspeed(3000.0, params.target_cas, params.target_mach)
    assert float(env.get_obs(state, params)[9]) == pytest.approx(float(expected))


def test_the_airspeed_term_charges_the_distance_from_the_schedule():
    env = Airplane2D()
    params = env.default_params.replace(turbulence_sigma=0.0)
    _, state = env.reset_env(KEY, params)
    speed = float(scheduled_airspeed(3000.0, params.target_cas, params.target_mach))
    on = state.replace(z=jnp.float32(3000.0), x_dot=jnp.float32(speed))
    off = on.replace(x_dot=jnp.float32(speed + 25.0))  # 169 m/s EAS, inside VMO
    assert float(env.reward_terms(on, params)["running"]) == 0.0
    assert float(env.reward_terms(off, params)["running"]) > 0.0


@pytest.mark.parametrize(
    "name",
    (
        "plane",
        "plane_sine",
        "plane_energy",
    ),
)
def test_the_expert_flies_the_schedule_without_tripping(name):
    """On training seed 100 the expert stays inside the envelope, and after its
    capture sits within 15 m/s of the schedule, a third of the 45 m/s by which the
    old 230 m/s stood above it at 3000 m."""
    import target_gym as tg

    env, params = tg.make(name)
    params = params.replace(max_steps_in_episode=240)
    expert = tg.make_expert(name)
    memory = expert.reset()
    key = jax.random.PRNGKey(100)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    act = jax.jit(expert.step)
    tripped = False
    for k in jax.random.split(key, 240):
        action, memory = act(memory, obs)
        obs, state, _, _, info = step(k, state, action, params)
        tripped |= bool(info["tripped"])
    assert not tripped
    speed = float(jnp.sqrt(state.x_dot**2 + state.z_dot**2))
    assert abs(speed - float(obs[9])) < 15.0
