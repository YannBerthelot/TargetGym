import jax
import jax.numpy as jnp
import pytest

from target_gym.pc_gym.first_order.env import FirstOrderParams, FirstOrderState
from target_gym.pc_gym.first_order.env_jax import FirstOrderSystem


@pytest.fixture
def env():
    return FirstOrderSystem()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_reset_returns_correct_shapes(env, key):
    obs, state = env.reset_env(key)
    assert obs.shape == env.obs_shape
    assert isinstance(state, FirstOrderState)
    assert state.time == 0


def test_reset_initial_values_in_range(env, key):
    _, state = env.reset_env(key)
    p = env.default_params
    assert p.initial_x_range[0] <= state.x <= p.initial_x_range[1]
    assert p.target_x_range[0] <= state.target_x <= p.target_x_range[1]


def test_reset_produces_different_initial_conditions(env):
    _, s1 = env.reset_env(jax.random.PRNGKey(0))
    _, s2 = env.reset_env(jax.random.PRNGKey(1))
    # Different seeds should (almost certainly) give different initial states
    assert not jnp.allclose(
        jnp.array([s1.x, s1.target_x]), jnp.array([s2.x, s2.target_x])
    )


@pytest.mark.parametrize("method", ["euler_1", "rk2_1", "rk4_1"])
def test_step_advances_state(env, key, method):
    env2 = FirstOrderSystem(integration_method=method)
    _, state = env2.reset_env(key)
    obs, new_state, reward, done, info = env2.step_env(key, state, jnp.array([0.5]))

    assert obs.shape == env2.obs_shape
    assert isinstance(new_state, FirstOrderState)
    assert new_state.time == state.time + 1
    assert jnp.isfinite(reward)
    assert done.dtype == jnp.bool_
    assert "last_state" in info


def test_obs_contains_x_and_target(env, key):
    _, state = env.reset_env(key)
    obs = env.get_obs(state)
    assert jnp.allclose(obs[0], state.x)
    assert jnp.allclose(obs[1], state.target_x)


def test_get_obs_consistent(env, key):
    _, state = env.reset_env(key)
    obs1 = env.get_obs(state)
    obs2 = env.get_obs(state, env.default_params)
    assert jnp.allclose(obs1, obs2)


def test_action_space(env):
    space = env.action_space()
    assert space.shape == (1,)
    assert jnp.all(space.low == -1.0)
    assert jnp.all(space.high == 1.0)


def test_observation_space(env):
    space = env.observation_space(env.default_params)
    assert space.shape == env.obs_shape


def test_state_space(env):
    space = env.state_space(env.default_params)
    assert space.shape == len(FirstOrderState.__dataclass_fields__)


def test_terminal_when_x_out_of_bounds(env, key):
    p = env.default_params
    state = FirstOrderState(time=0, x=p.x_max + 1.0, target_x=1.0, u=0.0)
    term, _ = env.is_terminal(state, p)
    assert term


def test_not_terminal_at_reset(env, key):
    _, state = env.reset_env(key)
    term, trunc = env.is_terminal(state, env.default_params)
    assert not term
    assert not trunc
