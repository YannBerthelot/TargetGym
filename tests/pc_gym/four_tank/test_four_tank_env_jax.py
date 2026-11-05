import jax
import jax.numpy as jnp
import pytest

from target_gym.pc_gym.four_tank.env_jax import FourTank


@pytest.fixture
def env():
    return FourTank()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_reset_initial_levels_in_range(env, key):
    _, state = env.reset_env(key)
    p = env.default_params
    assert p.initial_h1_range[0] <= state.h1 <= p.initial_h1_range[1]
    assert p.initial_h2_range[0] <= state.h2 <= p.initial_h2_range[1]
    assert p.initial_h3_range[0] <= state.h3 <= p.initial_h3_range[1]
    assert p.initial_h4_range[0] <= state.h4 <= p.initial_h4_range[1]


def test_reset_target_levels_in_range(env, key):
    _, state = env.reset_env(key)
    p = env.default_params
    assert p.target_h1_range[0] <= state.target_h1 <= p.target_h1_range[1]
    assert p.target_h2_range[0] <= state.target_h2 <= p.target_h2_range[1]


def test_obs_contains_all_levels_and_targets(env, key):
    _, state = env.reset_env(key)
    obs = env.get_obs(state)
    assert jnp.allclose(obs[0], state.h1)
    assert jnp.allclose(obs[1], state.h2)
    assert jnp.allclose(obs[2], state.h3)
    assert jnp.allclose(obs[3], state.h4)
    assert jnp.allclose(obs[4], state.target_h1)
    assert jnp.allclose(obs[5], state.target_h2)


def test_terminal_when_tank_overflows(env, key):
    p = env.default_params
    _, state = env.reset_env(key)
    state = state.replace(h1=p.h_max + 0.1)
    term = env.is_terminated(state, p)
    assert term
