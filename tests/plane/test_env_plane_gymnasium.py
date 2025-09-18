import pytest

from target_gym.plane.env_gymnasium import (
    Airplane2D,
    EnvParams,
    EnvState,
    compute_reward,
)


def test_init():
    env = Airplane2D()


def test_reset():
    env = Airplane2D()
    obs, info = env.reset()
    assert obs.shape == env.obs_shape


def test_compute_reward():
    env = Airplane2D()
    obs, info = env.reset()
    env_params = EnvParams()
    reward = compute_reward(state=env.state, params=env_params)
    assert reward.shape == ()
    assert 1 > reward > 0


def test_sample_action():
    env = Airplane2D()
    obs, info = env.reset()
    env_params = EnvParams()
    action = env.action_space.sample()
    for i in range(len(action)):
        assert env.action_space.low[i] <= action[i] <= env.action_space.high[i]
    assert action.shape == (2,)


def test_step():
    env = Airplane2D()
    obs, info = env.reset(seed=42)
    env_params = EnvParams()
    action = (0.5, 0)
    state = env.state
    # Perform the step transition.
    n_obs, reward, terminated, truncated, new_info = env.step(action)
    new_state = env.state
    assert new_state.x > state.x
    assert new_state.x_dot == pytest.approx(state.x_dot, rel=0.1)
    assert new_state.z_dot < state.z_dot
    if new_state.z_dot <= 0:
        assert new_state.z <= state.z
    else:
        assert new_state.z > state.z

    assert new_state.power < state.power
    assert new_state.t == state.t + 1
    # assert new_state.theta == state.theta
    assert new_state.alpha > state.alpha
    assert new_state.gamma < state.gamma


def test_is_terminal():
    env = Airplane2D()
    obs, info = env.reset()
    env_params = EnvParams()
    terminal_state = EnvState(
        x=0,
        x_dot=0,
        z=env_params.max_alt + 0.01,
        z_dot=0,
        theta=0,
        theta_dot=0,
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
        z=env_params.min_alt - 0.01,
        z_dot=0,
        theta=0,
        theta_dot=0,
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
        z=env_params.max_alt - 0.01,
        z_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        theta_dot=0,
        stick=0,
        fuel=0,
        t=env_params.max_steps_in_episode + 1,
        target_altitude=0,
    )
    assert env.is_terminal(terminal_state, env_params)


def test_render():
    env = Airplane2D(render_mode="human")
    env.reset()
    env.render()
