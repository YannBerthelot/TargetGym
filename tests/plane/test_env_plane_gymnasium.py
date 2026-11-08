import pytest

from target_gym.plane.env import (
    PlaneParams,
    PlaneState,
    compute_reward,
)
from target_gym.plane.env_jax import Airplane2D as _Plane
from target_gym.wrapper import GymnasiumEnv


def Airplane2D(env_params=None, render_mode=None):
    return GymnasiumEnv(_Plane(), env_params, render_mode)


def test_compute_reward():
    env = Airplane2D()
    obs, info = env.reset()
    env_params = PlaneParams(reward_version=1)
    reward = compute_reward(state=env.state, params=env_params)
    assert reward.shape == ()
    # 1 exactly when the hold starts on its level (Amendment 2 rule 17).
    assert 1 >= reward > 0


def test_step():
    # Still air: the assertion is about the elevator, not a gust.
    env = Airplane2D(env_params=PlaneParams(turbulence_sigma=0.0))
    obs, info = env.reset(seed=42)
    # Idle and a level pitch demand, below the reset's trim pitch (the hold
    # starts established, Amendment 2 rule 17).
    action = (-1.0, 0)
    state = env.state
    # Perform the step transition.
    n_obs, reward, terminated, truncated, new_info = env.step(action)
    new_state = env.state
    assert new_state.x > state.x
    assert float(new_state.x_dot) == pytest.approx(float(state.x_dot), rel=0.1)
    assert new_state.z_dot < state.z_dot
    if new_state.z_dot <= 0:
        assert new_state.z <= state.z
    else:
        assert new_state.z > state.z

    assert new_state.power < state.power
    assert new_state.time == state.time + 1
    assert new_state.theta < state.theta
    assert new_state.gamma < state.gamma


def test_is_terminal():
    """Only the ground ends a flight: the ceiling is an operating limit, not a
    trip (Amendment 2 rule 14; REALITY.md, limits, TCDS item 11)."""
    env_params = PlaneParams(reward_version=1)

    def state_at(z):
        return PlaneState(
            x=0,
            x_dot=0,
            z=z,
            z_dot=0,
            theta=0,
            theta_dot=0,
            alpha=0,
            gamma=0,
            m=0,
            power=0,
            stick=0,
            fuel=0,
            time=0,
            target_altitude=0,
        )

    assert not Airplane2D().is_terminated(
        state_at(env_params.max_alt + 0.01), env_params
    )
    assert Airplane2D().is_terminated(state_at(env_params.min_alt - 0.01), env_params)


def test_render():
    env = Airplane2D(render_mode="human")
    env.reset()
    env.render()
