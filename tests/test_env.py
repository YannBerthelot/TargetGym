import jax
import pytest

from plane.env import Airplane2D, EnvParams, EnvState, compute_reward


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


def test_is_terminal():
    env = Airplane2D()
    key = jax.random.PRNGKey(seed=42)
    obs, state = env.reset(key)
    env_params = EnvParams()
    terminal_state = EnvState(
        x=0,
        x_dot=0,
        z=env_params.max_alt + 0.01,
        z_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        fuel=0,
        rho=0,
        t=0,
        target_altitude=0,
    )
    assert env.is_terminal(terminal_state, env_params)
    terminal_state = EnvState(
        x=0,
        x_dot=0,
        z=env_params.min_alt - 0.01,
        z_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        fuel=0,
        rho=0,
        t=0,
        target_altitude=0,
    )
    assert env.is_terminal(terminal_state, env_params)
    terminal_state = EnvState(
        x=0,
        x_dot=0,
        z=env_params.max_alt - 0.01,
        z_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        fuel=0,
        rho=0,
        t=env_params.max_steps_in_episode + 1,
        target_altitude=0,
    )
    assert env.is_terminal(terminal_state, env_params)


def test_render():
    pass
