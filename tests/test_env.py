import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plane.env import compute_reward, get_env_classes
from plane.env_gymnasium import Airplane2D as GymEnv
from plane.env_jax import Airplane2D as JaxEnv

EnvState, EnvParams, EnvMetrics = get_env_classes(use_jax=True)


def test_init():
    env = JaxEnv()


def test_reset():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    assert obs.shape == env.obs_shape


def test_compute_reward():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    env_params = EnvParams()
    reward = compute_reward(state=env_state, params=env_params)
    assert isinstance(reward, (jnp.ndarray, np.ndarray))
    assert 1 >= reward >= 0


def test_sample_action():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    env_params = EnvParams()
    action = env.action_space(env_params).sample(key)
    assert 0 <= action[0] <= 1
    assert -1 <= action[1] <= 1
    assert action.shape == (2,)


def test_step():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, state = env.reset(key)
    action = (
        1.0,
        0,
    )  # Sample a valid action (e.g., maximum throttle and no pitch change)
    # Perform the step transition.
    n_obs, new_state, reward, terminated, truncated, _ = env.step(key, state, action)
    assert new_state.x > state.x
    assert new_state.x_dot == pytest.approx(state.x_dot, rel=0.1)
    assert new_state.z < state.z
    assert new_state.z_dot < state.z_dot
    assert new_state.power >= state.power
    assert new_state.t == state.t + 1
    assert new_state.alpha > state.alpha
    assert new_state.gamma < state.gamma


def test_is_terminal():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, state = env.reset(key)
    env_params = EnvParams()
    terminal_state = EnvState(
        x=0,
        x_dot=0,
        z=env_params.max_alt + 0.01,
        z_dot=0,
        theta_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        stick=0,
        fuel=0,
        t=0,
        target_altitude=0,
    )
    assert env.is_terminal(terminal_state, env_params)
    terminal_state = EnvState(
        x=0,
        x_dot=0,
        z=env_params.max_alt + 0.01,
        z_dot=0,
        theta_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        stick=0,
        fuel=0,
        t=0,
        target_altitude=0,
    )
    assert env.is_terminal(terminal_state, env_params)
    terminal_state = EnvState(
        x=0,
        x_dot=0,
        z=env_params.max_alt + 0.01,
        z_dot=0,
        theta_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        stick=0,
        fuel=0,
        t=0,
        target_altitude=0,
    )
    assert env.is_terminal(terminal_state, env_params)


def test_render():
    pass


def test_environments_compatible():
    """Test that both environments produce similar results"""
    jax_env = JaxEnv()
    gym_env = GymEnv()

    # Reset environments
    key = jax.random.PRNGKey(0)
    gym_obs, gym_state = gym_env.reset(seed=0)
    jax_obs, jax_state = gym_obs, gym_state  # jax_env.reset(key)

    # Test same action in both environments
    action = (0.8, 0.0)  # power, stick

    # JAX step
    jax_obs, jax_next_state, jax_reward, jax_terminated, jax_truncated, _ = (
        jax_env.step(key, jax_state, action, jax_env.default_params)
    )

    # Gym step

    gym_obs, gym_reward, gym_terminated, gym_truncated, _ = gym_env.step(action)

    gym_next_state = gym_env.state

    # Compare results
    assert np.allclose(jax_next_state.x, gym_next_state.x, rtol=1e-2)
    assert np.allclose(jax_next_state.z, gym_next_state.z, rtol=1e-2)
    assert np.allclose(jax_reward, gym_reward, rtol=1e-2)
    assert jax_terminated == gym_terminated
    assert jax_truncated == gym_truncated
