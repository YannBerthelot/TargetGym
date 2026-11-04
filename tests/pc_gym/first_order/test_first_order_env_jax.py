import jax
import jax.numpy as jnp
import pytest

from target_gym.pc_gym.first_order.env import FirstOrderState
from target_gym.pc_gym.first_order.env_jax import FirstOrderSystem


@pytest.fixture
def env():
    return FirstOrderSystem()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_reset_initial_values_in_range(env, key):
    _, state = env.reset_env(key)
    p = env.default_params
    assert p.initial_x_range[0] <= state.x <= p.initial_x_range[1]
    assert p.target_x_range[0] <= state.target_x <= p.target_x_range[1]


def test_obs_contains_x_and_target(env, key):
    _, state = env.reset_env(key)
    obs = env.get_obs(state)
    assert jnp.allclose(obs[0], state.x)
    assert jnp.allclose(obs[1], state.target_x)


def test_terminal_when_x_out_of_bounds(env, key):
    p = env.default_params
    state = FirstOrderState(time=0, x=p.x_max + 1.0, target_x=1.0, u=0.0)
    term = env.is_terminated(state, p)
    assert term
