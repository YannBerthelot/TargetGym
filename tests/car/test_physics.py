import jax.numpy as jnp
import pytest

from target_gym.car.env import (
    EnvParams,
    EnvState,
    check_is_terminal,
    compute_acceleration,
    compute_next_state,
    compute_reward,
    compute_theta_from_position,
    compute_thrust,
    engine_torque_from_rpm,
    get_obs,
    road_profile,
)


@pytest.fixture
def params():
    return EnvParams()


@pytest.fixture
def state(params):
    return EnvState(
        x=0.0,
        velocity=20.0,  # ~72 km/h
        throttle=0.5,
        t=0,
        target_velocity=25.0,
    )


# ------------------------------
# Torque curve
# ------------------------------
def test_engine_torque_curve_monotonic(params):
    low = engine_torque_from_rpm(1000, 1.0, params)
    mid = engine_torque_from_rpm(params.peak_rpm, 1.0, params)
    high = engine_torque_from_rpm(params.redline_rpm, 1.0, params)

    assert mid > low
    assert mid > high
    assert jnp.all(low >= 0.0)
    assert jnp.all(high >= 0.0)


# ------------------------------
# Road profile
# ------------------------------
def test_road_profile_smooth():
    x = jnp.linspace(0, 100, 50)
    z = road_profile(x)
    assert z.shape == x.shape
    assert jnp.isfinite(z).all()


def test_theta_computation():
    angle = compute_theta_from_position(10.0, road_profile)
    assert jnp.isscalar(angle)
    assert -jnp.pi / 2 < angle < jnp.pi / 2


# ------------------------------
# Terminal + Reward
# ------------------------------
def test_check_is_terminal_velocity(params, state):
    # Normal speed -> not terminated
    term, trunc = check_is_terminal(state, params)
    assert term is False
    assert trunc is False

    # Low velocity -> terminated
    state_low_v = state.replace(velocity=0.0)
    term, _ = check_is_terminal(state_low_v, params)
    assert term is True


def test_compute_reward(params, state):
    r = compute_reward(state, params)
    assert jnp.isscalar(r)
    # Reward should be between -max_penalty and 1
    assert r <= 1.0
    assert r >= -params.max_steps_in_episode


# ------------------------------
# Observations
# ------------------------------
def test_get_obs_shape(state):
    obs = get_obs(state)
    assert obs.shape == (5,)
    assert jnp.isfinite(obs).all()


# ------------------------------
# Dynamics
# ------------------------------
def test_compute_thrust_increases_with_throttle(params):
    v = 20.0
    f_low = compute_thrust(0.2, v, params)
    f_high = compute_thrust(1.0, v, params)
    assert f_high > f_low


def test_compute_acceleration_against_drag(params):
    theta = 0.0
    v = 30.0
    a_throttle = compute_acceleration(1.0, v, theta, params)
    a_zero = compute_acceleration(0.0, v, theta, params)
    assert a_throttle > a_zero


def test_compute_next_state_progress(params, state):
    s_next = compute_next_state(1.0, state, params, n_substeps=5, xp=jnp)
    # Time should advance
    assert s_next.t == state.t + 1
    # X should advance forward
    assert s_next.x > state.x
    # Velocity should remain finite
    assert jnp.isfinite(s_next.velocity)
