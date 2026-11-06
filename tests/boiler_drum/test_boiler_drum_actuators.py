"""Firing, feedwater and the turbine valve, against Bell and Astrom (1987).

Contract, from the report on this 160 MW unit (TFRT-3192) and from the task:

- The fuel actuator moves at most 0.007 of its span per second, the feedwater at
  most 0.05 (section 4). Actuator positions are read by the observation.
- Fuel reaches the risers through a 20 s lag, ours (PHYSICS.md, Actuators); the
  feedwater's lag is the report's 1 s.
- A reset plant is balanced: holding what ``fuel_pct`` and ``feed_pct`` read moves
  neither actuator.
- For a fixed turbine valve the steam flow is proportional to drum pressure, the
  report's ``q_s = (1.1 u2 - 0.19) P`` (eq. 14), so pressure regulates itself.

Examples: ``Q_max`` 240 MW and ``q_feed_max`` 200 kg/s, 2 s steps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.boiler_drum.env import BoilerDrumParams
from target_gym.boiler_drum.env_jax import BoilerDrum

STILL = BoilerDrumParams(q_steam_noise_std=0.0, sensor_noise_scale=0.0)
ENV = BoilerDrum()
RESET, STEP = jax.jit(ENV.reset_env), jax.jit(ENV.step_env)


def _reset(params=STILL, seed=0):
    return RESET(jax.random.PRNGKey(seed), params)


def _roll(state, action, steps, params=STILL):
    key, action = jax.random.PRNGKey(1), jnp.asarray(action, jnp.float32)
    for _ in range(steps):
        _, state, _, _, _ = STEP(key, state, action, params)
    return state


def _held(obs):
    """The raw action that repeats what the actuators read, in percent."""
    return np.asarray(obs)[[3, 4]] / 50.0 - 1.0


def test_full_fuel_demand_climbs_at_the_units_rate():
    """0.007 of the 240 MW span per second is 1.68 MW/s, 33.6 MW in 10 steps."""
    obs, state = _reset()
    last = _roll(state, [1.0, _held(obs)[1]], 10)
    assert float(last.Q_fuel - state.Q_fuel) == pytest.approx(33.6e6, rel=1e-3)


def test_full_feed_demand_climbs_at_the_units_rate():
    """0.05 of the 200 kg/s span per second is 10 kg/s/s, 60 kg/s in 3 steps."""
    obs, state = _reset()
    last = _roll(state, [_held(obs)[0], 1.0], 3)
    assert float(last.q_feed - state.q_feed) == pytest.approx(60.0, rel=1e-3)


def test_a_small_firing_step_follows_a_20_s_lag():
    """Inside the rate limit, 20 s closes 1 - exp(-1) = 0.632 of the step."""
    obs, state = _reset()
    held = _held(obs)
    last = _roll(state, [held[0] + 0.01, held[1]], 10)
    moved = float(last.Q_fuel - state.Q_fuel) / (0.005 * STILL.Q_max)
    assert moved == pytest.approx(0.632, abs=0.005)


def test_a_small_feed_step_follows_a_1_s_lag():
    """One 2 s step closes 1 - exp(-2) = 0.865 of it."""
    obs, state = _reset()
    held = _held(obs)
    last = _roll(state, [held[0], held[1] + 0.01], 1)
    moved = float(last.q_feed - state.q_feed) / (0.005 * STILL.q_feed_max)
    assert moved == pytest.approx(0.865, abs=0.005)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_holding_the_readings_leaves_a_reset_plant_alone(seed):
    """Steam draws differ by seed, and so does the reset's feedwater."""
    params = BoilerDrumParams(sensor_noise_scale=0.0)
    obs, state = _reset(params, seed)
    next_state = _roll(state, _held(obs), 1, params)
    assert float(next_state.Q_fuel) == pytest.approx(float(state.Q_fuel), rel=1e-5)
    assert float(next_state.q_feed) == pytest.approx(float(state.q_feed), rel=1e-5)


def test_steam_flow_is_proportional_to_pressure_at_a_fixed_valve():
    """At 90 bar the valve set for 93.35 kg/s at 85 bar passes 98.84 kg/s."""
    obs, state = _reset()
    high = _roll(state.replace(pressure=jnp.float32(90.0)), _held(obs), 1)
    assert float(high.q_steam / high.pressure) == pytest.approx(93.35 / 85.0, rel=1e-5)
    assert float(high.q_steam) == pytest.approx(98.84, rel=2e-3)


@pytest.mark.parametrize("offset", [-5.0, 5.0])
def test_pressure_regulates_itself_at_fixed_fuel_and_valve(offset):
    """Off its balance, drum pressure returns: a boiler draws less steam at low
    pressure and more at high. 10 minutes (300 steps) spans Bell and Astrom's
    time constant: from eq. 3, ``1 / (0.0018 u2 (9/8) P^(1/8))`` is about 400 s
    at ``u2`` 0.7, so a real unit is back by 3/4 of the offset. The model's
    storage differs, so this asks only for half."""
    obs, state = _reset()
    start = state.replace(pressure=state.pressure + offset)
    last = _roll(start, _held(obs), 300)
    assert abs(float(last.pressure) - STILL.p_nominal) < 0.5 * abs(offset)
