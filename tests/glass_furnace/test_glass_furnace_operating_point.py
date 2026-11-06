"""The furnace's settled reset, fuel slew, observed pull and hidden gas.

Contract, from docs/proposals/industrial-fixes.md (PR 5) and the task:

- Reset is a settled operating point. Held at the firing it starts on, with the
  disturbances off, neither the crown nor the melt drifts: the review measured
  the melt at -0.9 K/h with the crown held. The bound is 0.1 K/h, a kelvin over
  the 13 h episode, the crown thermocouple's resolution.
- Fuel slews at 1/1800 of its span per second, ours (PHYSICS.md, Actuators).
- A float line's pull is steady and known, so the observation reports it.
- The gas's heating value is hidden. It moves by 1 % (one sigma) about its mean
  over a day, the size from Leicher's 4 % between supply gases and the time from
  Glass Service's "weekly and sometimes even daily". A furnace is found on the
  gas of the day: reset's heat does not depend on the draw.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.glass_furnace.env import GlassFurnaceParams
from target_gym.glass_furnace.env_jax import GlassFurnace

STILL = GlassFurnaceParams(
    m_pull_noise_std=0.0, lhv_noise_std=0.0, sensor_noise_scale=0.0
)
ENV = GlassFurnace()
RESET, STEP = jax.jit(ENV.reset_env), jax.jit(ENV.step_env)
CYCLE = 100  # steps in a full reversal cycle: 2 x 1500 s at 30 s


def _raw(fuel, params):
    return 2.0 * (fuel - params.fuel_min) / (params.fuel_max - params.fuel_min) - 1.0


@jax.jit
def _hold(state, action):
    def body(s, _):
        _, s, _, _, _ = ENV.step_env(jax.random.PRNGKey(1), s, action, STILL)
        return s, (s.T_crown, s.T_melt)

    return jax.lax.scan(body, state, None, length=1600)[1]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_reset_sits_on_the_steady_state_of_its_firing(seed):
    """Cycle means, first against last, 1500 steps or 12.5 h apart."""
    obs, state = RESET(jax.random.PRNGKey(seed), STILL)
    action = jnp.array([_raw(state.fuel_pipeline[-1], STILL)])
    for name, trace in zip(("crown", "melt"), _hold(state, action)):
        trace = np.asarray(trace)
        rate = (trace[-CYCLE:].mean() - trace[:CYCLE].mean()) / 12.5
        assert abs(rate) < 0.1, f"{name} drifts {rate:.3f} K/h"


def test_holding_the_fuel_reading_leaves_a_reset_furnace_alone():
    _, state = RESET(jax.random.PRNGKey(0), STILL)
    action, last = jnp.array([_raw(state.fuel_flow, STILL)]), state
    for _ in range(5):
        _, last, _, _, _ = STEP(jax.random.PRNGKey(1), last, action, STILL)
    assert float(last.fuel_flow) == pytest.approx(float(state.fuel_flow), rel=1e-5)


def test_full_fuel_demand_slews_at_its_limit():
    """1/1800 of the 0.185 kg/s span per second is 3.08 g/s per 30 s step. The
    first two steps burn the pipeline, so ten moves show after 12 steps."""
    _, state = RESET(jax.random.PRNGKey(0), STILL)
    start, last = state.fuel_flow, state
    for _ in range(12):
        _, last, _, _, _ = STEP(jax.random.PRNGKey(1), last, jnp.ones(1), STILL)
    assert float(last.fuel_flow - start) == pytest.approx(0.03083, rel=1e-3)


def test_the_pull_is_observed():
    """5.79 kg/s and a 0.3 kg/s deviation read 6.09."""
    _, state = RESET(jax.random.PRNGKey(0), STILL)
    obs = ENV.get_obs(state.replace(m_pull_disturbance=jnp.float32(0.3)), STILL)
    assert obs.shape == ENV.obs_shape == (6,)
    assert float(obs[5]) == pytest.approx(6.09, rel=1e-5)


def test_the_gas_is_hidden_and_a_richer_gas_burns_hotter():
    _, state = RESET(jax.random.PRNGKey(0), STILL)
    richer = state.replace(lhv_deviation=jnp.float32(0.02))
    np.testing.assert_array_equal(ENV.get_obs(state, STILL), ENV.get_obs(richer, STILL))
    action = jnp.array([_raw(state.fuel_pipeline[-1], STILL)])
    key = jax.random.PRNGKey(1)
    flame, rich_flame = STEP(key, state, action, STILL), STEP(
        key, richer, action, STILL
    )
    assert float(rich_flame[1].T_gas) > float(flame[1].T_gas) + 1.0


def test_reset_burns_the_same_heat_on_any_gas():
    key = jax.random.PRNGKey(3)
    _, base = RESET(key, STILL)
    _, state = RESET(key, GlassFurnaceParams())
    assert float(state.T_crown) == pytest.approx(float(base.T_crown))
    assert abs(float(state.lhv_deviation)) > 1e-4
    assert float(state.fuel_flow * (1.0 + state.lhv_deviation)) == pytest.approx(
        float(base.fuel_flow), rel=1e-5
    )


def test_the_gas_wanders_one_percent_over_a_day():
    """Stationary from reset; one 30 s step moves it sqrt(1 - rho^2) of its
    sigma, with rho = exp(-30 / 86400): 2.63e-4. 256 draws give 4.4 % error."""
    params, keys = GlassFurnaceParams(), jax.random.split(jax.random.PRNGKey(5), 256)
    _, states = jax.vmap(ENV.reset_env, (0, None))(keys, params)
    assert float(states.lhv_deviation.std()) == pytest.approx(0.01, rel=0.15)
    _, moved, _, _, _ = jax.vmap(ENV.step_env, (0, 0, None, None))(
        keys, states, jnp.zeros(1), params
    )
    step = moved.lhv_deviation - states.lhv_deviation
    assert float(step.std()) == pytest.approx(2.63e-4, rel=0.15)
