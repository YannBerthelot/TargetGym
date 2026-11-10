"""The autopilot's own limits and what the operator pays inside the bands
(owner's decisions of 2026-11-10, plane/PHYSICS.md).

Contract:

- Every controller flies through the autopilot's manoeuvre limit, 0.2 g either
  side of the load the flight path needs (ours, the top of the 0.1 to 0.2 g the
  record's autopilots hold a capture to [KFC 500, TM 1-1510-225-10]). A load
  factor below zero is a Hazardous effect [EASA AMC 25.1329, A4.2]: no demand
  reaches it. The attitude loop overshoots a little, so the check allows 0.05 g
  past the limit.
- The economy term is fuel plus time by the cost index per air distance,
  normalised so the reference cruise (FL330, M 0.78, 73.5 t) costs 1 a second.
- The heavy corner of ``plane``'s scenarios keeps 1.3 g to the buffet line, the
  recommended maximum altitude's margin.
"""

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.plane.dynamics import (
    buffet_line,
    compute_air_density_from_altitude,
    compute_speed_of_sound_from_altitude,
    scheduled_airspeed,
)
from target_gym.plane.env import PlaneParams

FT, KT = 0.3048, 1852.0 / 3600.0


def _worst_loads(pitch_action, seconds=10):
    env, params = tg.make("plane", turbulence_sigma=0.0, sensor_noise_scale=0.0)
    _, state = env.reset(jax.random.PRNGKey(3), params)
    action = env.trim_action(state, params).at[1].set(pitch_action)
    loads = []
    for k in range(seconds):
        _, state, _, _, _, _ = env.step(jax.random.PRNGKey(k), state, action, params)
        loads.append(float(state.load_factor))
    return np.array(loads)


def test_a_full_nose_up_demand_is_held_to_the_autopilots_limit():
    loads = _worst_loads(1.0)
    assert loads.max() <= 1.2 + 0.05, f"worst {loads.max():.3f} g"
    assert loads.max() >= 1.1, "the demand should still reach the limit"


def test_a_full_nose_down_demand_never_unloads_below_zero_g():
    loads = _worst_loads(-1.0)
    assert loads.min() >= 0.8 - 0.05, f"worst {loads.min():.3f} g"


def test_the_reference_cruise_costs_one_a_second_on_economy():
    """At trim on the schedule, FL330 and 73.5 t, the economy term is its
    normalisation's 1.0: the fuel flow it was scaled by is the model's own."""
    env, params = tg.make(
        "plane",
        turbulence_sigma=0.0,
        sensor_noise_scale=0.0,
        target_altitude_range=(33_000 * FT, 33_000 * FT),
        initial_altitude_range=(33_000 * FT, 33_000 * FT),
    )
    _, state = env.reset(jax.random.PRNGKey(0), params)
    economy = float(env.reward_terms(state, params)["economy"])
    assert economy == pytest.approx(1.0, abs=0.02)


def test_the_heavy_corner_keeps_one_point_three_g_to_buffet():
    from target_gym.plane.scenarios import suite

    heavy = next(s for s in suite() if s.name == "heavy_high")
    params = PlaneParams()
    top = heavy.overrides["target_altitude_range"][1]
    v = float(scheduled_airspeed(top, params.target_cas, params.target_mach))
    mach = v / float(compute_speed_of_sound_from_altitude(top))
    rho = float(compute_air_density_from_altitude(top))
    cl = (
        heavy.overrides["initial_mass"]
        * params.gravity
        / (0.5 * rho * v * v * params.wings_surface)
    )
    assert float(buffet_line(mach, params)) / cl >= 1.3
