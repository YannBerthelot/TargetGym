"""The drum boiler's limits, score, observation and episode (charter rules
14 to 17), each from ``REALITY.md``.

Contract:
- Trips (rule 14): the level past -330 mm [S9] or +254 mm [S20] for more than
  10 s [S9] trips the plant; a shorter excursion does not. Pressure has no
  sourced limit, so it never trips.
- Score (rule 15): tonnes of steam delivered off the load plan per step, and a
  trip costs the nominal steam of a 2.5 h hot restart [S17]: 93.35 kg/s *
  9000 s = 840.15 t.
- Observation (rule 16): the plan is read: load now, load planned, and the
  seconds to the ramp.
- Episode (rule 17): established operation at a planned load between 40 % and
  100 % of nominal, balanced at normal water level, one ramp at 1 to 2 %/min.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.boiler_drum.env import (
    BoilerDrumParams,
    check_is_terminal,
    compute_reward,
    compute_velocity,
)
from target_gym.boiler_drum.env_jax import BoilerDrum

PARAMS = BoilerDrumParams(sensor_noise_scale=0.0)


@pytest.fixture(scope="module")
def env():
    return BoilerDrum()


def _reset(env, seed=0, params=PARAMS):
    return env.reset_env(jax.random.PRNGKey(seed), params)


def _held(state, level, seconds, params=PARAMS):
    """``state`` after ``seconds`` with the level held at ``level``: the trip
    timers count what the plant reports, step by step."""
    high = jnp.asarray(0.0)
    low = jnp.asarray(0.0)
    for _ in range(int(round(seconds / params.delta_t))):
        high = jnp.where(level >= params.level_trip_high, high + params.delta_t, 0.0)
        low = jnp.where(level <= params.level_trip_low, low + params.delta_t, 0.0)
    return state.replace(level=jnp.asarray(level), time_high=high, time_low=low)


@pytest.mark.parametrize("level", [0.26, -0.34])
def test_level_past_a_trip_for_ten_seconds_or_less_does_not_trip(env, level):
    _, state = _reset(env)
    tripped, _ = check_is_terminal(_held(state, level, 10.0), PARAMS)
    assert not bool(tripped)


@pytest.mark.parametrize("level", [0.26, -0.34])
def test_level_past_a_trip_for_more_than_ten_seconds_trips(env, level):
    _, state = _reset(env)
    tripped, _ = check_is_terminal(_held(state, level, 12.0), PARAMS)
    assert bool(tripped)


def test_trips_are_asymmetric(env):
    """-0.30 m is inside the low trip at -330 mm; +0.30 m is past the high
    trip at +254 mm."""
    _, state = _reset(env)
    low, _ = check_is_terminal(_held(state, -0.30, 20.0), PARAMS)
    high, _ = check_is_terminal(_held(state, 0.30, 20.0), PARAMS)
    assert not bool(low) and bool(high)


@pytest.mark.parametrize("pressure", [55.0, 115.0])
def test_pressure_alone_never_trips(env, pressure):
    _, state = _reset(env)
    tripped, _ = check_is_terminal(
        state.replace(pressure=jnp.asarray(pressure)), PARAMS
    )
    assert not bool(tripped)


def test_steam_on_plan_costs_nothing(env):
    _, state = _reset(env)
    assert float(compute_reward(state, PARAMS)) == 0.0


def test_steam_off_plan_costs_its_tonnes(env):
    """Valve passing 80 kg/s at nominal; 2 bar low passes 80 * 83/85: 1.882
    kg/s short, 3.765 kg in a 2 s step."""
    _, state = _reset(env)
    q = 80.0 * 83.0 / 85.0
    state = state.replace(q_demand=jnp.asarray(80.0), q_steam=jnp.asarray(q))
    np.testing.assert_allclose(
        -float(compute_reward(state, PARAMS)), (80.0 - q) * 2.0 / 1000.0, rtol=1e-5
    )


def test_a_trip_costs_a_hot_restarts_steam(env):
    _, state = _reset(env)
    tripped = _held(state, 0.30, 12.0)
    np.testing.assert_allclose(
        -float(compute_reward(tripped, PARAMS)), 93.35 * 9000.0 / 1000.0, rtol=1e-5
    )


@pytest.mark.parametrize("seed", range(8))
def test_episode_starts_in_established_operation_on_a_plan(env, seed):
    obs, state = _reset(env, seed)
    nominal = PARAMS.q_steam_nominal
    assert 0.4 * nominal - 1e-4 <= float(state.load_from) <= nominal + 1e-4
    assert 0.4 * nominal - 1e-4 <= float(state.load_to) <= nominal + 1e-4
    rate = float(state.ramp_rate) * 60.0 / nominal * 100.0  # %/min
    assert 1.0 - 1e-4 <= rate <= 2.0 + 1e-4
    assert float(state.level) == 0.0
    # Balanced: no derivative at the reset's firing, feedwater and load.
    d, _ = compute_velocity(
        jnp.array([state.pressure, state.V_wt, state.alpha_r, state.V_sd]),
        (state.Q_fuel, state.q_feed),
        state.q_demand,
        PARAMS,
    )
    np.testing.assert_allclose(np.asarray(d), 0.0, atol=1e-5)
    # The plan is read: load now, load planned, seconds to the ramp.
    np.testing.assert_allclose(
        np.asarray(obs[7:]),
        [float(state.load_from), float(state.load_to), float(state.ramp_start)],
        rtol=1e-5,
    )


def test_the_valve_follows_the_plan(env):
    """The load moves at the planned rate after the ramp starts, and stops at
    the planned load."""
    _, state = _reset(env, 3)
    action = env.initial_action(state, PARAMS)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    loads = []
    for _ in range(PARAMS.max_steps_in_episode):
        _, state, _, _, _ = step(key, state, action, PARAMS)
        loads.append(float(state.q_demand))
    span = float(state.load_to - state.load_from)
    expected_end = float(state.ramp_start) + abs(span) / float(state.ramp_rate)
    assert expected_end < PARAMS.max_steps_in_episode * PARAMS.delta_t
    np.testing.assert_allclose(loads[-1], float(state.load_to), rtol=1e-5)
