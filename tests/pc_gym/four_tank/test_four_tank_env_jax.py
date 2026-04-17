import jax
import jax.numpy as jnp
import pytest

from target_gym.pc_gym.four_tank.env import FourTankParams, FourTankState
from target_gym.pc_gym.four_tank.env_jax import FourTank


@pytest.fixture
def env():
    return FourTank()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_reset_returns_correct_shapes(env, key):
    obs, state = env.reset_env(key)
    assert obs.shape == env.obs_shape
    assert isinstance(state, FourTankState)
    assert state.time == 0


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


def test_reset_different_seeds_differ(env):
    _, s1 = env.reset_env(jax.random.PRNGKey(0))
    _, s2 = env.reset_env(jax.random.PRNGKey(1))
    h1 = jnp.array([s1.h1, s1.h2, s1.target_h1])
    h2 = jnp.array([s2.h1, s2.h2, s2.target_h1])
    assert not jnp.allclose(h1, h2)


@pytest.mark.parametrize("method", ["euler_1", "rk2_1", "rk4_1"])
def test_step_advances_state(env, key, method):
    env2 = FourTank(integration_method=method)
    _, state = env2.reset_env(key)
    obs, new_state, reward, done, info = env2.step_env(
        key, state, jnp.array([0.0, 0.0])
    )
    assert obs.shape == env2.obs_shape
    assert isinstance(new_state, FourTankState)
    assert new_state.time == state.time + 1
    assert jnp.isfinite(reward)
    assert done.dtype == jnp.bool_


def test_obs_contains_all_levels_and_targets(env, key):
    _, state = env.reset_env(key)
    obs = env.get_obs(state)
    assert jnp.allclose(obs[0], state.h1)
    assert jnp.allclose(obs[1], state.h2)
    assert jnp.allclose(obs[2], state.h3)
    assert jnp.allclose(obs[3], state.h4)
    assert jnp.allclose(obs[4], state.target_h1)
    assert jnp.allclose(obs[5], state.target_h2)


def test_action_space(env):
    space = env.action_space()
    assert space.shape == (2,)
    assert jnp.all(space.low == -1.0)
    assert jnp.all(space.high == 1.0)


def test_observation_space(env):
    assert env.observation_space(env.default_params).shape == env.obs_shape


def test_state_space(env):
    assert env.state_space(env.default_params).shape == len(
        FourTankState.__dataclass_fields__
    )


def test_terminal_when_tank_overflows(env, key):
    p = env.default_params
    _, state = env.reset_env(key)
    state = state.replace(h1=p.h_max + 0.1)
    term, _ = env.is_terminal(state, p)
    assert term


def test_not_terminal_at_reset(env, key):
    _, state = env.reset_env(key)
    term, trunc = env.is_terminal(state, env.default_params)
    assert not term
    assert not trunc
