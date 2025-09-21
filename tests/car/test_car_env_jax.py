import jax
import jax.numpy as jnp
import pytest

from target_gym.car.env import CarParams, CarState
from target_gym.car.env_jax import Car2D

# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture
def env():
    return Car2D()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# -------------------------------
# Reset / initialization tests
# -------------------------------


def test_reset_env(env, key):
    obs, state = env.reset_env(key)
    assert isinstance(state, CarState)
    assert isinstance(obs, jnp.ndarray)
    assert obs.ndim == 1
    assert obs.shape[0] == env.obs_shape[0]
    # velocity within initial range
    assert (
        env.default_params.initial_velocity_range[0]
        <= state.velocity
        <= env.default_params.initial_velocity_range[1]
    )
    # target_velocity within range
    assert (
        env.default_params.target_velocity_range[0]
        <= state.target_velocity
        <= env.default_params.target_velocity_range[1]
    )


# -------------------------------
# Step tests
# -------------------------------


def test_step_env_returns_correct_types(env, key):
    obs, state = env.reset_env(key)
    action = jnp.array(0.5, dtype=jnp.float32)
    next_obs, next_state, reward, done, info = env.step_env(key, state, action)
    assert isinstance(next_state, CarState)
    assert isinstance(next_obs, jnp.ndarray)
    assert next_obs.shape[0] == env.obs_shape[0]
    assert isinstance(reward, jnp.ndarray) or isinstance(reward, float)
    assert isinstance(done, jnp.ndarray) or isinstance(done, bool)
    assert isinstance(info, dict)
    assert "last_state" in info


# -------------------------------
# Observation tests
# -------------------------------


def test_get_obs_shape_and_type(env, key):
    obs, state = env.reset_env(key)
    obs_array = env.get_obs(state)
    assert isinstance(obs_array, jnp.ndarray)
    assert obs_array.ndim == 1
    assert obs_array.shape[0] == env.obs_shape[0]


# -------------------------------
# Terminal state tests
# -------------------------------


def test_is_terminal_logic(env):
    params = env.default_params
    # velocity below min -> terminated
    state_low = CarState(x=0.0, velocity=-1.0, t=0, target_velocity=10.0, throttle=0.0)
    terminated, truncated = env.is_terminal(state_low, params)
    assert terminated
    # velocity above max -> terminated
    state_high = CarState(
        x=0.0,
        velocity=params.max_velocity + 1.0,
        t=0,
        target_velocity=10.0,
        throttle=0.5,
    )
    terminated, truncated = env.is_terminal(state_high, params)
    assert terminated
    # t >= max_steps -> truncated
    state_trunc = CarState(
        x=0.0,
        velocity=10.0,
        t=params.max_steps_in_episode,
        target_velocity=10.0,
        throttle=0.5,
    )
    terminated, truncated = env.is_terminal(state_trunc, params)
    assert truncated


# -------------------------------
# Action / observation space tests
# -------------------------------


def test_action_space(env):
    space = env.action_space()
    assert space.low.shape == (1,)
    assert space.high.shape == (1,)
    assert jnp.all(space.low == -1.0)
    assert jnp.all(space.high == 1.0)
