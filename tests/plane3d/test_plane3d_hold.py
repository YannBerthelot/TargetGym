"""The hold's job on ``plane3d_racetrack``: fuel at the aircraft's own holding
speed, under the maximum holding airspeed (owner's decision, 2026-11-23).

Contract, from src/target_gym/plane3d/REALITY.md:

- Cost. Time in the hold is ATC's, so what the crew pays is fuel: green dot is
  "computed to minimize drag and thus, the fuel consumption (for example during
  the HOLD phase)" [SF24]. The task's ``economy`` term is the fuel burned, in kg
  per second, by the A320's fuel law (``plane.dynamics.fuel_flow``; owner,
  2026-11-24); its other terms are the bands of Amendments 2 to 4, unchanged.
- Assigned speed. Green dot, "an estimate of the speed for best lift-to-drag
  ratio" and "the recommended speed ... for a holding in clean configuration",
  computed from "the aircraft weight ... and the pressure altitude" [SF24]. On
  this plant: the speed of least thrust, so of least drag, in level flight at
  the aircraft's mass, never above the maximum holding airspeed. Since the fuel
  law, consumption per newton rises with speed, so the least-fuel speed is
  under green dot; the reference stays at green dot, as the aircraft holds,
  and the speed is the agent's (owner, 2026-11-24).
- Speed limit. A maximum, 230 KIAS from 6 001 to 14 000 ft and 265 KIAS above
  [AIM538, IFH ch. 10], so priced on that side only: past it, the squared excess
  in 10 kt widths per second (the width of AIM 4-4-12's tolerance, as before).
  Under it the task prices nothing: no lower limit is in the record besides the
  aircraft's protection.
- Pattern. An RNAV hold sized for the speed held, turns at 25 deg at the
  greatest ground speed [AIM538; plane3d/PHYSICS.md, Amendment 4].
- Reference. Where green dot is at or over the maximum holding airspeed the
  reference rests on a priced limit, and holds under it by the rule for that
  case (owner, 2026-11-23, "lowest with no exit"): on a 1 kt grid, the
  smallest margin at which it passes the limit on none of the nominal seeds,
  100 to 109, in the normal day.

On the A320-233 plant of 2026-11-23 (plane/PHYSICS.md) green dot is near
231 kt at 73.5 t, so the 230 KIAS limit binds on a low hold from about 73 t.

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


@pytest.mark.parametrize(
    "mass, level",
    # Where green dot is under the maximum holding airspeed: every mass under
    # the 265 KIAS limit, and the light aircraft under the 230 KIAS one.
    ((60_000.0, 3700.0), (60_000.0, 6500.0), (73_500.0, 6500.0), (78_000.0, 6500.0)),
)
def test_the_assigned_speed_is_the_plants_least_drag_speed(mass, level):
    """Green dot is the best lift-to-drag speed [SF24]: level flight 10 kt
    faster or 10 kt slower than the assigned speed needs more thrust."""
    env, params, _, state = _hold(level, mass)
    cas = float(env.assigned_cas(state, params))
    at = float(_level_thrust(cas, level, mass))
    assert at < float(_level_thrust(cas + 10.0 * KT, level, mass))
    assert at < float(_level_thrust(cas - 10.0 * KT, level, mass))


@pytest.mark.parametrize("mass, level", ((60_000.0, 6500.0), (73_500.0, 6500.0)))
def test_least_fuel_is_under_green_dot_once_consumption_rises_with_speed(mass, level):
    """At the least-drag speed the thrust is flat in speed, and the A320's
    consumption per newton rises with speed, so level flight 10 kt under green
    dot burns less fuel for a little more thrust. That speed is the agent's to
    find: the reference holds at green dot, as the aircraft does [SF24]."""
    env, params, _, state = _hold(level, mass)
    cas = float(env.assigned_cas(state, params))

    def burned(speed):
        thrust = _level_thrust(speed, level, mass)
        tas = D.scheduled_airspeed(level, speed, 0.99)
        return float(D.fuel_flow(thrust, tas, level, params))

    assert burned(cas - 10.0 * KT) < burned(cas)
    assert float(_level_thrust(cas - 10.0 * KT, level, mass)) > float(
        _level_thrust(cas, level, mass)
    )


@pytest.mark.parametrize("mass", (73_500.0, 78_000.0))
def test_a_heavy_aircraft_on_a_low_hold_is_assigned_the_maximum(mass):
    """From about 73 t the least-drag speed is at or over 230 KIAS, the maximum
    holding airspeed to 14 000 ft [AIM538]: the aircraft is assigned the
    maximum, the allowed speed nearest its green dot, so 10 kt slower needs
    more thrust; at 78 t, 10 kt faster would need less."""
    env, params, _, state = _hold(3700.0, mass)
    cas = float(env.assigned_cas(state, params))
    assert cas == pytest.approx(230.0 * KT)
    at = float(_level_thrust(cas, 3700.0, mass))
    assert at < float(_level_thrust(cas - 10.0 * KT, 3700.0, mass))
    if mass == 78_000.0:
        assert float(_level_thrust(cas + 10.0 * KT, 3700.0, mass)) < at


def test_the_assigned_speed_follows_the_aircraft_weight():
    """Green dot is computed from the aircraft's weight [SF24]; at a fixed lift
    coefficient the speed goes as its square root. Read on a high hold, whose
    265 KIAS limit no A320's green dot reaches."""
    speeds = []
    for mass in MASSES:
        env, params, _, state = _hold(HIGH, mass)
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


# -- The reference at the limit ----------------------------------------------------


@pytest.mark.medium
def test_the_reference_holds_under_the_limit_by_the_no_exit_rule():
    """73.5 t on low holds, the normal day, the nominal seeds 100 to 109: the
    reference never passes 230 KIAS, and with its margin one knot of the grid
    smaller it does on at least one seed. That is the rule: the smallest margin
    with no exit."""
    from target_gym import scenarios
    from target_gym.plane3d import env as E
    from target_gym.registry import REGISTRY

    spec = REGISTRY[RACETRACK]
    env, ctrl = spec.make_env(), spec.make_pid_controller(None)
    (normal,) = [s for s in scenarios.suite(spec) if s.name == "normal"]
    params = scenarios.params_for(
        spec, normal, target_altitude_range=(3_000.0, 14_000.0 * FT)
    )

    def furthest_past_the_limit(seed, margin):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        gains = ctrl.params.replace(cap_margin=margin)

        def body(carry, k):
            obs, state, memory = carry
            action, memory = ctrl.step(gains, memory, obs, jnp)
            obs, state, *_ = env.step_env(k, state, action, params)
            cas = D.calibrated_airspeed(state.z, E.airspeed_3d(state, params))
            return (obs, state, memory), cas - 230.0 * KT

        memory = jax.tree.map(jnp.asarray, ctrl.init_state)
        keys = jax.random.split(key, int(params.max_steps_in_episode))
        return jax.lax.scan(body, (obs, state, memory), keys)[1].max()

    run = jax.jit(jax.vmap(furthest_past_the_limit, in_axes=(0, None)))
    seeds = jnp.arange(100, 110)
    shipped = float(ctrl.params.cap_margin)
    assert float(run(seeds, shipped).max()) <= 0.0
    assert float(run(seeds, shipped - 1.0 * KT).max()) > 0.0
