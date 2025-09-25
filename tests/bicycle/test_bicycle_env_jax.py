# test_env_jax.py
import jax
import jax.numpy as jnp
import pytest

from target_gym.bicycle.env_jax import RandlovBicycle


@pytest.fixture
def env():
    return RandlovBicycle()


def test_reset_returns_obs_and_state(env):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)
    assert obs.shape == env.obs_shape
    # check state has expected attributes
    assert hasattr(state, "omega")
    assert hasattr(state, "x_f")
    assert state.time == 0


def test_step_returns_correct_tuple(env):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    action = jnp.array([0.1, 0.0])
    obs2, state2, reward, done, info = env.step_env(key, state, action)

    # Shapes
    assert obs2.shape == env.obs_shape
    assert isinstance(reward, jnp.ndarray)
    assert reward.shape == ()
    assert isinstance(done, jnp.ndarray)
    assert done.shape == ()

    # Info dict should contain state and metrics
    assert "last_state" in info
    assert "metrics" in info


def test_observation_consistency(env):
    key = jax.random.PRNGKey(1)
    obs, state = env.reset_env(key)
    obs2 = env.get_obs(state)
    assert jnp.allclose(obs, obs2)


def test_action_space(env):
    space = env.action_space()
    assert space.shape == (2,)
    assert jnp.all(space.low == -1.0)
    assert jnp.all(space.high == 1.0)


def test_observation_space(env):
    space = env.observation_space()
    assert space.shape == env.obs_shape
    # assert jnp.isinf(space.low).all()
    # assert jnp.isinf(space.high).all()


def test_state_space(env):
    space = env.state_space()
    # EnvState has 11 fields
    assert space.shape == (11,)
    # assert jnp.isinf(space.low).all()
    # assert jnp.isinf(space.high).all()


def test_terminal_conditions(env):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    # Construct a bad state with huge tilt
    bad_state = state.replace(omega=jnp.deg2rad(90.0))
    assert env.is_terminal(bad_state, env.default_params)

    # Construct a good state
    good_state = state.replace(omega=0.0)
    assert not env.is_terminal(good_state, env.default_params)
