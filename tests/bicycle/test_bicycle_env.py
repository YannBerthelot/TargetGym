# test_bike_env.py

import jax
import jax.numpy as jnp
import pytest

from target_gym.bicycle.env import (  # replace "your_module" with actual file name
    BikeParams,
    BikeState,
    check_is_terminal,
    compute_acceleration_bike,
    compute_next_state,
    compute_reward,
    get_obs,
)


@pytest.fixture
def default_params():
    return BikeParams()


@pytest.fixture
def default_state(default_params):
    return BikeState(
        omega=0.0,
        omega_dot=0.0,
        theta=0.01,  # small steering angle so psi evolves
        theta_dot=0.0,
        psi=0.0,
        x_f=0.0,
        y_f=0.0,
        x_b=-default_params.l,
        y_b=0.0,
        last_d=0.0,
        t=0,
    )


def test_compute_acceleration_zero_action(default_params):
    velocities = jnp.array([0.0, 0.0, 0.0])
    positions = jnp.array([0.0, 0.0, 0.0])
    action = (0.0, 0.0)

    acc, metrics = compute_acceleration_bike(
        velocities, positions, action, default_params
    )

    # Should be finite
    assert jnp.all(jnp.isfinite(acc))
    # At zero everything, psi_ddot should be zero
    assert abs(acc[2]) < 1e-6


def test_compute_acceleration_with_torque(default_params):
    velocities = jnp.array([0.0, 0.0, 0.0])
    positions = jnp.array([0.0, 0.01, 0.0])  # small steer
    action = (1.0, 0.0)

    acc, metrics = compute_acceleration_bike(
        velocities, positions, action, default_params
    )
    assert jnp.all(jnp.isfinite(acc))
    # Expect nonzero theta_ddot with torque
    assert abs(acc[1]) > 0


def test_next_state_runs(default_state, default_params):
    action = jnp.array([0.5, 0.0])
    new_state, metrics = compute_next_state(
        action, default_state, default_params, "rk4_1"
    )

    assert isinstance(new_state, type(default_state))
    assert jnp.isfinite(new_state.omega)
    assert jnp.isfinite(new_state.psi)
    # psi should increase if steering is nonzero
    state2, _ = compute_next_state(action, new_state, default_params, "rk4_1")
    assert abs(state2.psi) > abs(new_state.psi)


def test_terminal_and_reward(default_state, default_params):
    # Not terminal at init
    terminated, truncated = check_is_terminal(default_state, default_params)
    assert not bool(terminated)
    assert not bool(truncated)

    # Exceed tilt -> terminal
    bad_state = default_state.replace(
        omega=jnp.deg2rad(default_params.max_tilt_deg + 5)
    )
    terminated, _ = check_is_terminal(bad_state, default_params)
    assert bool(terminated)

    # # Reward should be -1 if terminated
    # r = compute_reward(bad_state, default_params)
    # assert r == -1.0


def test_get_obs_shape(default_state, default_params):
    obs = get_obs(default_state, default_params)
    assert obs.shape == (5,)
    assert jnp.all(jnp.isfinite(obs))
