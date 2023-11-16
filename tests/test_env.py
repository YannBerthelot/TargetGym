from plane.env import Airplane2D, EnvParams, compute_reward
import jax
import pytest


def test_init():
    env = Airplane2D()


def test_reset():
    env = Airplane2D()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    assert obs.shape == env.obs_shape


def test_compute_reward():
    env = Airplane2D()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    env_params = EnvParams()
    reward = compute_reward(state=env_state, params=env_params)
    assert reward.shape == ()
    assert -1 < reward < 0


def test_sample_action():
    env = Airplane2D()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    env_params = EnvParams()
    action = env.action_space(env_params).sample(key)
    assert 0 < action < 1
    assert action.shape == ()


def test_step():
    env = Airplane2D()
    key = jax.random.PRNGKey(seed=42)
    obs, state = env.reset(key)
    env_params = EnvParams()
    action = 1.0
    # Perform the step transition.
    n_obs, new_state, reward, done, _ = env.step(key, state, action, env_params)
    assert new_state.x > state.x
    assert new_state.x_dot == pytest.approx(state.x_dot, rel=0.1)
    assert new_state.z < state.z
    assert new_state.z_dot < state.z_dot
    assert new_state.power > state.power
    assert new_state.t == state.t + 1
    assert new_state.theta == state.theta
    assert new_state.alpha > state.alpha
    assert new_state.gamma < state.gamma


def test_render():
    pass
