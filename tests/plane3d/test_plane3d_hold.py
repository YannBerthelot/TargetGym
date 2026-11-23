"""The hold's job on ``plane3d_racetrack``: fuel at the aircraft's own holding
speed, under the maximum holding airspeed (owner's decision, 2026-11-23).

Contract, from src/target_gym/plane3d/REALITY.md:

- Cost. Time in the hold is ATC's, so what the crew pays is fuel: green dot is
  "computed to minimize drag and thus, the fuel consumption (for example during
  the HOLD phase)" [SF24]. The task's ``economy`` term is the fuel burned, in kg
  per second; its other terms are the bands of Amendments 2 to 4, unchanged.
- Assigned speed. Green dot, "an estimate of the speed for best lift-to-drag
  ratio" and "the recommended speed ... for a holding in clean configuration",
  computed from "the aircraft weight ... and the pressure altitude" [SF24]. On
  this plant: the speed of least thrust in level flight at the aircraft's mass,
  never above the maximum holding airspeed.
- Speed limit. A maximum, 230 KIAS from 6 001 to 14 000 ft and 265 KIAS above
  [AIM538, IFH ch. 10], so priced on that side only: past it, the squared excess
  in 10 kt widths per second (the width of AIM 4-4-12's tolerance, as before).
  Under it the task prices nothing: no lower limit is in the record besides the
  aircraft's protection.
- Pattern. An RNAV hold sized for the speed held, turns at 25 deg at the
  greatest ground speed [AIM538; plane3d/PHYSICS.md, Amendment 4].

Hand-derived numbers: on a low hold 240 KIAS is 10 kt past the 230 cap, one
width, (10 / 10)^2 = 1.0 per second; 250 KIAS is two widths, 4.0. A jet's best
lift-to-drag speed at a fixed lift coefficient goes as the square root of its
weight: 78 t against 60 t, sqrt(78 / 60) = 1.140.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.plane import dynamics as D

FT, KT, G = 0.3048, 1852.0 / 3600.0, 9.81
RACETRACK = "plane3d_racetrack"
KEY = jax.random.PRNGKey(11)
LOW, HIGH = 4000.0, 7000.0  # m: under and over 14 000 ft (4 267 m)
MASSES = (60_000.0, 73_500.0, 78_000.0)  # the A320 family's range, plane/PHYSICS.md


def _hold(level=LOW, mass=73_500.0, **changes):
    """The hold in still, calm air at ``level`` for an aircraft of ``mass``."""
    env, params = tg.make(RACETRACK)
    params = params.replace(
        turbulence_sigma=0.0,
        sensor_noise_scale=0.0,
        target_altitude_range=(level, level),
        initial_mass=mass,
        **changes,
    )
    obs, state = env.reset_env(KEY, params)
    return env, params, obs, state


def _at_cas(state, cas):
    """``state`` flying its leg at ``cas`` (m/s), in still air."""
    tas = float(D.scheduled_airspeed(state.z, cas, 0.99))
    psi = float(state.psi)
    return state.replace(
        x_dot=jnp.float32(tas * np.cos(psi)), y_dot=jnp.float32(tas * np.sin(psi))
    )


@jax.jit
def _level_thrust(cas, z, mass):
    """The thrust that holds level flight at ``cas``: the plant's own trim."""
    _, params = tg.make(RACETRACK)
    tas = D.scheduled_airspeed(z, cas, 0.99)
    _, _, power = D.trim_level(tas, z, mass, params)
    return D.compute_thrust_output(
        power=power,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
        rho=D.compute_air_density_from_altitude(z),
        M=tas / D.compute_speed_of_sound_from_altitude(z),
    )


# -- The assigned speed ------------------------------------------------------------


@pytest.mark.parametrize("level", (3700.0, 6500.0))
@pytest.mark.parametrize("mass", MASSES)
def test_the_assigned_speed_is_the_plants_least_fuel_speed(mass, level):
    """Green dot minimises drag, hence fuel, in the hold [SF24]: level flight
    10 kt faster or 10 kt slower than the assigned speed needs more thrust."""
    env, params, _, state = _hold(level, mass)
    cas = float(env.assigned_cas(state, params))
    at = float(_level_thrust(cas, level, mass))
    assert at < float(_level_thrust(cas + 10.0 * KT, level, mass))
    assert at < float(_level_thrust(cas - 10.0 * KT, level, mass))


def test_the_assigned_speed_follows_the_aircraft_weight():
    """Green dot is computed from the aircraft's weight [SF24]; at a fixed lift
    coefficient the speed goes as its square root."""
    speeds = []
    for mass in MASSES:
        env, params, _, state = _hold(LOW, mass)
        speeds.append(float(env.assigned_cas(state, params)))
    assert speeds[0] < speeds[1] < speeds[2]
    assert speeds[2] / speeds[0] == pytest.approx(np.sqrt(78.0 / 60.0), rel=0.02)


@pytest.mark.parametrize("level, cap", ((LOW, 230.0), (HIGH, 265.0)))
@pytest.mark.parametrize("mass", MASSES)
def test_the_assigned_speed_is_at_or_under_the_maximum_holding_airspeed(
    mass, level, cap
):
    env, params, _, state = _hold(level, mass)
    assert float(env.assigned_cas(state, params)) <= cap * KT + 1e-3


def test_a_cap_under_green_dot_is_the_assigned_speed():
    """Where the best lift-to-drag speed is over the maximum holding airspeed
    the aircraft holds at the maximum: a 200 KIAS cap, as under 6 000 ft
    [AIM538], is under any A320's green dot."""
    env, params, _, state = _hold(LOW, 78_000.0, holding_cap_low=200.0 * KT)
    assert float(env.assigned_cas(state, params)) == pytest.approx(200.0 * KT)


def test_the_observation_gives_the_assigned_speed_as_a_true_airspeed():
    env, params, obs, state = _hold(HIGH)
    cas = env.assigned_cas(state, params)
    tas = float(D.scheduled_airspeed(state.z, cas, 0.99))
    assert float(obs[env.obs_index["target_speed"]]) == pytest.approx(tas, abs=0.05)
    assert float(obs[env.obs_index["airspeed"]]) == pytest.approx(tas, abs=0.05)


def test_the_pattern_is_sized_for_the_speed_held():
    """An RNAV hold's turns take 25 deg at the greatest ground speed, here the
    assigned airspeed in still air, and its inbound leg is that speed's minute
    [AIM538]."""
    env, params, _, state = _hold(LOW, 60_000.0)
    tas = float(D.scheduled_airspeed(LOW, env.assigned_cas(state, params), 0.99))
    radius = tas**2 / (G * np.tan(np.deg2rad(25.0)))
    assert float(state.target_radius) == pytest.approx(radius, rel=1e-3)
    assert 2.0 * float(state.target_half_leg) / tas == pytest.approx(60.0, abs=0.5)


# -- The speed limit ---------------------------------------------------------------


@pytest.mark.parametrize("under", (0.0, 10.0, 30.0))
def test_holding_at_or_under_the_assigned_speed_costs_nothing_on_speed(under):
    """The sources give a maximum holding airspeed and no minimum [AIM538]."""
    env, params, _, state = _hold(HIGH)
    cas = float(env.assigned_cas(state, params)) - under * KT
    terms = env.reward_terms(_at_cas(state, cas), params)
    assert float(terms["running"]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("past, cost", ((0.0, 0.0), (10.0, 1.0), (20.0, 4.0)))
def test_past_the_maximum_holding_airspeed_costs_its_squared_excess(past, cost):
    """230 KIAS at or below 14 000 ft [AIM538]: 10 kt past it is one width."""
    env, params, _, state = _hold(LOW)
    terms = env.reward_terms(_at_cas(state, (230.0 + past) * KT), params)
    assert float(terms["running"]) == pytest.approx(cost, rel=0.03, abs=1e-3)


# -- The cost ----------------------------------------------------------------------


@pytest.mark.medium
def test_the_holds_cost_is_the_fuel_it_burns():
    """Two minutes of the reference's hold in calm air: inside every band, so
    the whole cost is ``economy``, and it is the kilograms of fuel burned."""
    env, params, obs, state = _hold(LOW)
    expert = tg.make_expert(RACETRACK)

    def body(carry, key):
        obs, state, memory = carry
        action, memory = expert.step(memory, obs)
        obs, state, reward, _, _ = env.step_env(key, state, action, params)
        return (obs, state, memory), (env.reward_terms(state, params), reward)

    run = jax.jit(lambda carry, keys: jax.lax.scan(body, carry, keys))
    (_, last, _), (terms, reward) = run(
        (obs, state, expert.reset()), jax.random.split(KEY, 120)
    )
    burned = float(state.fuel - last.fuel)
    assert 60.0 < burned < 120.0  # the unit: tens of kg a minute, not g or t
    assert float(terms["economy"].sum()) == pytest.approx(burned, rel=0.01)
    assert float(terms["tracking"].sum() + terms["running"].sum()) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(-reward.sum()) == pytest.approx(burned, rel=0.01)
